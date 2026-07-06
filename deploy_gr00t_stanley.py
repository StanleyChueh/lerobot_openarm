import time
import torch
import matplotlib.pyplot as plt

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.utils import build_inference_frame, make_robot_action
from teleoperators.umeow_openarm_leader import OpenArmConfig, OpenArmLeader
from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower
from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

MAX_EPISODES = 30
MAX_STEPS_PER_EPISODE = 999999999
FPS = 30
RESET_TIME_SEC = 10
task = "pick up the white block from the red area"
robot_type = "openarm_follower"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 改成載入 GR00T / Groot policy checkpoint
    dataset_id = "ethanCSL/0422_stanley_red_cube"
    pretrained_path = "ethanCSL/0422_stanley_red_cube_gr00t"
    
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    
    model = GrootPolicy.from_pretrained(
        pretrained_path,
        tune_visual=True,
        tune_llm=False,
        use_bf16=False,
        strict=False,
    )
    model.to(device)
    model.eval()

    # 讓 preprocessor / model 使用同一個 device
    model.config.device = device

    preprocess, postprocess = make_groot_pre_post_processors(
        model.config,
        dataset_stats=dataset_metadata.stats,
    )

    camera_config = {
        "side_camera": OpenCVCameraConfig(index_or_path=18, width=640, height=480, fps=FPS),
        "left_camera": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
        "body_camera": OpenCVCameraConfig(index_or_path=14, width=640, height=480, fps=FPS),
    }

    robot_cfg = OpenArmFollowerConfig(
        right_port="can2",
        left_port="can3",
        enable_fd=True,
        model_path="/home/csl/lerobot_openarm/model/openarm_description.urdf",
        cameras=camera_config,  # type: ignore
    )
    robot = OpenArmFollower(robot_cfg)

    teleop_cfg = OpenArmConfig(
        right_port="can0",
        left_port="can1",
        enable_fd=True,
        model_path="/home/csl/lerobot_openarm/model/openarm_description.urdf",
    )
    teleop = OpenArmLeader(teleop_cfg)

    robot.connect()
    teleop.connect()
    time.sleep(1.0)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    action_keys = list(robot.action_features.keys())
    
    model_dt = 0.027
    interp_steps = 10
    control_dt = model_dt / interp_steps

    first = True

    time_log = []
    action_log = []
    start_time = time.perf_counter()

    try:
        for ep in range(MAX_EPISODES):
            print(f"Starting episode {ep}...")
            episode_has_frames = False
            first = True
            interp_start = time.perf_counter()

            for step in range(MAX_STEPS_PER_EPISODE):
                model_start = time.perf_counter()

                obs = robot.get_observation()
                obs_dict = obs.copy()

                if first:
                    first = False
                    continue

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=task,
                    robot_type=robot_type,
                )
                obs_processed = preprocess(obs_frame)

                action = model.select_action(obs_processed)
                action = postprocess(action)
                action = make_robot_action(action, dataset_features)

                model_time = time.perf_counter() - model_start

                time_log.append(time.perf_counter() - start_time)

                prev_action = obs_dict
                next_time = time.perf_counter()

                for i in range(interp_steps):
                    next_time += control_dt
                    alpha = (i + 1) / interp_steps

                    interp_action = {
                        joint: prev_action[joint] + (action[joint] - prev_action[joint]) * alpha
                        for joint in action.keys()
                    }
                    robot.send_action(interp_action)

                    now = time.perf_counter()
                    sleep_time = next_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_time = now

                interp_time = time.perf_counter() - interp_start
                print(
                    f"[Timing] model: {model_time*1000:.2f} ms | "
                    f"interp(total): {interp_time*1000:.2f} ms | "
                    f"per step: {(interp_time/interp_steps)*1000:.2f} ms"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, stopping loop...")

    finally:
        print("[INFO] Finalizing dataset...")

        try:
            robot.disconnect()
        except Exception as e:
            print(f"[WARN] Robot disconnect failed: {e}")
        try:
            teleop.disconnect()
        except Exception as e:
            print(f"[WARN] Teleop disconnect failed: {e}")

        print("[INFO] Plotting results...")
        if len(time_log) > 0:
            plt.figure()
            plt.plot(time_log, action_log)
            plt.xlabel("Time (s)")
            plt.ylabel(action_keys[0] if action_keys else "action[0]")
            plt.title("Action vs Time")
            plt.grid()
            plt.show()
        else:
            print("[WARN] No data to plot.")


if __name__ == "__main__":
    main()