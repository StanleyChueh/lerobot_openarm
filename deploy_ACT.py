import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower


MAX_EPISODES = 30
MAX_STEPS_PER_EPISODE = 999999999
FPS = 30

def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    # pretrained_model_path = "/home/csl/lerobot_openarm/outputs/train/act_20260120-must-success/checkpoints/last/pretrained_model"
    pretrained_model_path = "ethanCSL/0422_stanley_red_cube_act"
    model = ACTPolicy.from_pretrained(pretrained_model_path)

    dataset_id = "eval_ethanCSL/0422_stanley_red_cube_act"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)


    # # find ports using lerobot-find-port
    # # something like "/dev/tty.usbmodem58760431631"

    # # the robot ids are used the load the right calibration files
    # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    
    camera_config = {
    "side_camera": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
    "left_camera": OpenCVCameraConfig(index_or_path=16, width=640, height=480, fps=FPS), 
    "body_camera": OpenCVCameraConfig(index_or_path=15, width=640, height=480, fps=FPS),
    # Optional: fourcc="MJPG" for troubleshooting OpenCV async error.
}
    
    robot_cfg = OpenArmFollowerConfig(
    right_port = 'can2',
    left_port  = 'can3',
    
    enable_fd = True,
    
    model_path='/home/csl/lerobot_openarm/model/openarm_description.urdf',
    
    cameras=camera_config # type: ignore
)
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)

            action = make_robot_action(action, dataset_metadata.features)

            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()