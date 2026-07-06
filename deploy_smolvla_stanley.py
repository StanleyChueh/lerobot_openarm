import time
import torch
import matplotlib.pyplot as plt

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower
from teleoperators.umeow_openarm_leader import OpenArmConfig, OpenArmLeader
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors

MAX_EPISODES = 30
MAX_STEPS_PER_EPISODE = 999999999
FPS = 30
RESET_TIME_SEC = 10
task = "pick up the white block from the red area"
robot_type = "openarm_follower"


def main():
    device = torch.device("cuda")
    pretrained_model_path = "ethanCSL/0422_stanley_red_cube_large_steps"
    model = SmolVLAPolicy.from_pretrained(pretrained_model_path)

    dataset_id = "ethanCSL/0422_stanley_red_cube"
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        pretrained_model_path,
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

    camera_keys = list(camera_config.keys())
    state_keys = [k for k in robot.observation_features.keys() if k not in camera_keys]
    action_keys = list(robot.action_features.keys())

    record_dataset = LeRobotDataset.create(
        repo_id="ethanCSL/eval_0422_stanley_red_cube",
        robot_type=robot_type,
        fps=FPS,
        features=dataset_features,
        use_videos=True,
        image_writer_threads=4,
        video_backend="torchcodec",
    )

    _, events = init_keyboard_listener()
    init_rerun(session_name="recording")
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    model_dt = 0.027
    interp_steps = 10
    control_dt = model_dt / interp_steps

    first = True
    episode_has_frames = False

    def clear_decision_flags():
        events["rerecord_episode"] = False
        events["exit_early"] = False

    def wait_episode_decision():
        print("[EPISODE END] Press RIGHT to save, LEFT to discard, Ctrl+C to stop.")
        while True:
            if events["stop_recording"]:
                return "stop"
            if events["rerecord_episode"]:
                clear_decision_flags()
                return "discard"
            if events["exit_early"]:
                clear_decision_flags()
                return "save"
            time.sleep(0.05)

    def run_reset_mode():
        clear_decision_flags()
        log_say("Reset the environment with the leader arm")
        print("[RESET MODE] Use the leader arm to reset. Press RIGHT when reset is done.")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=task,
            display_data=True,
        )
        clear_decision_flags()

    time_log = []
    action_log = []
    start_time = time.perf_counter()

    try:
        for ep in range(MAX_EPISODES):
            print(f"Starting episode {ep}...")

            clear_decision_flags()
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

                action_target = model.select_action(obs_processed)
                action_target = postprocess(action_target)
                action_target = make_robot_action(action_target, dataset_features)

                obs_state = torch.tensor(
                    [float(obs_dict[k]) for k in state_keys],
                    dtype=torch.float32,
                )
                action_vec = torch.tensor(
                    [float(action_target[k]) for k in action_keys],
                    dtype=torch.float32,
                )

                frame = {
                    "observation.state": obs_state,
                    "action": action_vec,
                    **{f"observation.images.{cam_key}": obs_dict[cam_key] for cam_key in camera_keys},
                    "task": task,
                }
                record_dataset.add_frame(frame)
                episode_has_frames = True

                model_time = time.perf_counter() - model_start

                if events["stop_recording"]:
                    raise KeyboardInterrupt
                if events["rerecord_episode"] or events["exit_early"]:
                    break
                if model_time > model_dt:
                    continue

                prev_action = obs_dict
                next_time = time.perf_counter()

                for i in range(interp_steps):
                    next_time += control_dt
                    alpha = (i + 1) / interp_steps

                    interp_action = {
                        joint: prev_action[joint] + (action_target[joint] - prev_action[joint]) * alpha
                        for joint in action_target.keys()
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

            decision = wait_episode_decision()

            if decision == "stop":
                break
            if decision == "discard":
                if record_dataset.has_pending_frames():
                    record_dataset.clear_episode_buffer()
                episode_has_frames = False
                run_reset_mode()
                continue
            if decision == "save":
                if episode_has_frames:
                    record_dataset.save_episode()
                episode_has_frames = False
                run_reset_mode()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, stopping loop...")
        if episode_has_frames and record_dataset.has_pending_frames():
            try:
                record_dataset.save_episode()
            except Exception as e:
                print(f"[WARN] Failed to safely save current episode: {e}")

    finally:
        print("[INFO] Finalizing dataset...")
        try:
            record_dataset.finalize()
        except Exception as e:
            print(f"[WARN] Finalize failed: {e}")

        # Re-enable this after local testing if desired.
        # try:
        #     record_dataset.push_to_hub()
        # except Exception as e:
        #     print(f"[WARN] Push failed: {e}")

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
            plt.ylabel("LJ1.pos")
            plt.title("Action LJ1.pos vs Time")
            plt.grid()
            plt.show()
        else:
            print("[WARN] No data to plot.")


if __name__ == "__main__":
    main()