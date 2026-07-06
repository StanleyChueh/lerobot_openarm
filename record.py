from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
import os


# Change this to your dataset name
REPO_ID = "ethanCSL/" + "0522-6767"

hf_token = os.environ.get("HF_TOKEN") 

if hf_token:
    login(token=hf_token)
    print("Logged in successfully!")
else:
    print("HF_TOKEN environment variable not set. Cannot log in.")

import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import hw_to_dataset_features
from teleoperators.umeow_openarm_leader import OpenArmConfig, OpenArmLeader
from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower
from lerobot.common.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from pathlib import Path

# change this to your parameters
NUM_EPISODES = 30
FPS = 30
EPISODE_TIME_SEC = 999
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Place your left arm on top of the white area."  # something like "pick the red block"
RESUME = False
# make sure the camera ports is correct for your setup
camera_config = {
    
    "right_camera": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=FPS),
    "left_camera": OpenCVCameraConfig(index_or_path=17, width=640, height=480, fps=FPS), 
    "body_camera": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=FPS),
    #"side_camera": OpenCVCameraConfig(index_or_path=10, width=640, height=480, fps=FPS),
        # 5:  side_camera
        # 20: body_camera
        # 11: left wrist camera
    
}

robot_config = OpenArmFollowerConfig(
    right_port = 'can2',
    left_port  = 'can3',
    
    enable_fd = True,
    
    model_path='/home/csl/lerobot_openarm/model/openarm_description_leader.urdf',
    
    cameras= camera_config 
)

teleop_config = OpenArmConfig(
    right_port = 'can0',
    left_port  = 'can1',
    
    enable_fd = True,
    
    model_path='/home/csl/lerobot_openarm/model/openarm_description.urdf',
)

robot = OpenArmFollower(robot_config)
teleop = OpenArmLeader(teleop_config)


robot.connect()
teleop.connect()

time.sleep(1.0)  

action_features = hw_to_dataset_features(robot.action_features, "action") # type: ignore
# -> q -> [joints]
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# resume dataset with RESUME=True
DATASET_ROOT = Path(f"/home/csl/.cache/huggingface/lerobot/{REPO_ID}")

episode_idx = 0

if RESUME:
    try:
        print("Trying to resume dataset from local or Hugging Face Hub...")
        dataset = LeRobotDataset.resume(
            repo_id=REPO_ID,
            root=DATASET_ROOT,
            image_writer_threads=4,
            video_backend="torchcodec"
        )
        episode_idx = dataset.meta.total_episodes
        print(f"Resumed successfully. Existing episodes: {episode_idx}")
    except Exception as e:
        print(f"Resume failed: {e}")
        print("No existing dataset found locally or on Hub, creating a new dataset...")
        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=FPS,
            features=dataset_features,
            root=DATASET_ROOT,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
            video_backend="torchcodec",
        )
        episode_idx = 0
else:
    print("Creating new dataset...")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=dataset_features,
        root=DATASET_ROOT,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
        video_backend="torchcodec",
    )
    episode_idx = 0
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

try:
    target_episodes = episode_idx + NUM_EPISODES

    while episode_idx < target_episodes and not events["stop_recording"]:
        print(f"Recording episode {episode_idx + 1} of {target_episodes}")

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
        )

        if not events["stop_recording"] and (episode_idx < target_episodes - 1 or events["rerecord_episode"]):
            log_say("Reset the environment")
            record_loop(

                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                control_time_s=RESET_TIME_SEC,
                single_task=TASK_DESCRIPTION,
        
                display_data=True,
            )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        episode_idx += 1

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    print("Finalizing dataset...")

    try:
        dataset.finalize()
    except Exception as e:
        print(f"Finalize failed: {e}")

    try:
        dataset.push_to_hub()
    except Exception as e:
        print(f"Push failed: {e}")

    robot.disconnect()
    teleop.disconnect()

    print("Cleanup complete.")