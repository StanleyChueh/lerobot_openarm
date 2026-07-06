import time
import torch

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

from teleoperators.umeow_openarm_leader import OpenArmConfig, OpenArmLeader
from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower

FPS        = 30
TASK       = 'pick up the white block from the red area'
ROBOT_TYPE = 'openarm_follower'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pretrained_model_path = "ethanCSL/0422_stanley_red_cube_gr00t"
    policy = GrootPolicy.from_pretrained(
        pretrained_model_path,
        tune_visual=True,
        tune_llm=False,
        use_bf16=False,
        strict=False,
    )
    policy.to(device)
    policy.eval()
    
    dataset_id = "ethanCSL/0422_stanley_red_cube"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)

    preprocess, postprocess = make_groot_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats, # type: ignore
    ) 

    camera_config = {
        "side_camera": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
        "left_camera": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=FPS),
        "body_camera": OpenCVCameraConfig(index_or_path=16, width=640, height=480, fps=FPS),
    }

    robot_cfg = OpenArmFollowerConfig(
        right_port="can2",
        left_port="can3",
        enable_fd=True,
        model_path="/home/csl/lerobot_openarm/model/openarm_description.urdf",
        cameras=camera_config,  # type: ignore
    )
    robot = OpenArmFollower(robot_cfg)

    robot.connect()
    time.sleep(1.0)

    action_features = hw_to_dataset_features(robot.action_features, "action") # type: ignore
    obs_features    = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    first = True

    try:
        while True:
            obs = robot.get_observation()

            if first:
                first = False
                continue
                
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=TASK,
                robot_type=ROBOT_TYPE,
            )
            obs_processed = preprocess(obs_frame)
            action = policy.select_action(obs_processed)
            action = postprocess(action)
            action = make_robot_action(action, dataset_features)
            robot.send_action(action)
            print("[INFO] Action sent:", action)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, stopping loop...")


if __name__ == "__main__":
    main()