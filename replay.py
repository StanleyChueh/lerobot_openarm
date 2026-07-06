import time
import logging
from pprint import pformat

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say, init_logging
from lerobot.utils.constants import ACTION

# your robot imports (same as record script)
from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower


# ========= CONFIG =========
REPO_ID = "ethanCSL/0422_stanley_red_cube"
EPISODE_IDX = 0
FPS = 30

robot_config = OpenArmFollowerConfig(
    right_port='can2',
    left_port='can3',
    enable_fd=True,
    model_path='/home/csl/lerobot_openarm/model/openarm_description_leader.urdf',
)

# ==========================


def main():
    init_logging()

    print("Loading dataset...")
    dataset = LeRobotDataset(REPO_ID, episodes=[EPISODE_IDX], force_cache_sync=True)

    print(f"Dataset loaded. Num frames: {dataset.num_frames}")

    actions = dataset.select_columns(ACTION)

    robot = OpenArmFollower(robot_config)
    robot.connect()

    robot_action_processor = make_default_robot_action_processor()

    try:
        log_say("Replaying episode")

        for idx in range(dataset.num_frames):
            start_t = time.perf_counter()

            # --- get action ---
            action_array = actions[idx][ACTION]

            action = {}
            for i, name in enumerate(dataset.features[ACTION]["names"]):
                action[name] = action_array[i]

            # --- get robot obs (needed for processor) ---
            robot_obs = robot.get_observation()

            # --- process action ---
            processed_action = robot_action_processor((action, robot_obs))

            # --- send to robot ---
            robot.send_action(processed_action)

            # --- keep FPS consistent ---
            dt = time.perf_counter() - start_t
            precise_sleep(max(1 / FPS - dt, 0))

    except KeyboardInterrupt:
        print("Replay interrupted by user.")

    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()