"""Auto-resume: the same lerobot-record command twice, no --resume / --dataset.root (mocked robot, scripted Quest).

    rm -rf /tmp/lrhome; for i in 1 2; do HF_LEROBOT_HOME=/tmp/lrhome python chain_auto_resume.py <this directory>; done
    (run 1 creates 3 episodes, run 2 continues to 6)
"""
import sys
HERE = sys.argv[1]
sys.argv = ["lerobot-record", "--robot.type=openarm_umeow", "--robot.assume_yes=true",
    "--robot.cameras={body_cam: {type: opencv, index_or_path: 90, width: 64, height: 48, fps: 30}, wrist_cam: {type: opencv, index_or_path: 91, width: 64, height: 48, fps: 30}, right_wrist_cam: {type: opencv, index_or_path: 92, width: 64, height: 48, fps: 30}}",
    "--teleop.type=openarm_quest", "--teleop.port=5995", "--dataset.repo_id=local/chain_auto", "--dataset.no_stamp=true",
    "--dataset.push_to_hub=false", "--dataset.single_task=reach out and back", "--dataset.num_episodes=3",
    "--dataset.episode_time_s=10", "--dataset.reset_time_s=3", "--dataset.fps=30", "--play_sounds=false", "--display_data=false"]
sys.path.insert(0, HERE)
import json, math, os, socket, threading, time
import mock_hw  # noqa: F401  (imports the robot plugin -> auto_resume sees the argv above)
import lerobot.scripts.lerobot_record as rec
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot_robot_openarm_umeow.rerun_status import BOARD
EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
rec.init_keyboard_listener = lambda: (None, EVENTS)
Q = {"rx": 0.25, "x": False, "run": True}
def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose = lambda x: {"x": x, "y": 1.1, "z": 0.3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    while Q["run"]:
        s.sendto(json.dumps({"rc": pose(Q["rx"]), "lc": pose(-0.25), "rf": pose(0.0) | {"y": 1.5}, "rt": 0.0, "lt": 0.0,
                             "x": Q["x"], "y": False, "a": False, "b": False, "v": 0, "vr": 0, "vl": 0}).encode(), ("127.0.0.1", 5995))
        time.sleep(1 / 72)
def press_x():
    Q["x"] = True; time.sleep(0.15); Q["x"] = False
def operator():
    for _ in range(3):
        while BOARD.phase != "WAITING" and Q["run"]:
            time.sleep(0.02)
        time.sleep(0.5); press_x(); t0 = time.perf_counter()
        while time.perf_counter() - t0 < 2.0:
            Q["rx"] = 0.25 + 0.05 * math.sin(math.pi * (time.perf_counter() - t0) / 2.0); time.sleep(0.01)
        press_x()
threading.Thread(target=sender, daemon=True).start(); threading.Thread(target=operator, daemon=True).start()
print("ARGV seen by lerobot:", [a for a in sys.argv if "resume" in a or "dataset.root" in a])
rec.main(); Q["run"] = False
info = json.load(open(os.path.join(HF_LEROBOT_HOME, "local/chain_auto/meta/info.json")))
print("DATASET NOW:", info["total_episodes"], "episodes,", info["total_frames"], "frames")
