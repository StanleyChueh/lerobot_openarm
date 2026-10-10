"""chain_record.py against Isaac Sim: record 3 short episodes with lerobot-record, --robot.type=openarm_isaac.

Needs IsaacLab's scripts/tools/lerobot_sim_server.py running (port 5710). Same scripted Quest as chain_record.py.

    python chain_record_sim.py <dataset root> <this directory>
"""
import json, math, shutil, socket, sys, threading, time
sys.path.insert(0, sys.argv[2])
import lerobot.scripts.lerobot_record as rec
from lerobot.utils.keyboard_input import apply_recording_control
from lerobot_robot_openarm_umeow.rerun_status import BOARD
from lerobot_teleoperator_openarm_quest import OpenArmQuest

EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
rec.init_keyboard_listener = lambda: (None, EVENTS)
OpenArmQuest._press_episode_key = lambda self, key: apply_recording_control(key, EVENTS)
PORT = 5996
Q = {"rx": 0.25, "x": False, "run": True}

def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose = lambda x: {"x": x, "y": 1.1, "z": 0.3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    while Q["run"]:
        s.sendto(json.dumps({"rc": pose(Q["rx"]), "lc": pose(-0.25), "rf": pose(0.0) | {"y": 1.5}, "rt": 0.0, "lt": 0.0,
                             "x": Q["x"], "y": False, "a": False, "b": False, "v": 0, "vr": 0, "vl": 0}).encode(), ("127.0.0.1", PORT))
        time.sleep(1 / 72)

def press_x():
    Q["x"] = True; time.sleep(0.15); Q["x"] = False

def operator():
    for ep in range(3):
        while BOARD.phase != "WAITING" and Q["run"]:
            time.sleep(0.02)
        time.sleep(0.5); press_x(); t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:  # a reach: 6 cm out and back
            Q["rx"] = 0.25 + 0.06 * math.sin(math.pi * (time.perf_counter() - t0) / 3.0); time.sleep(0.01)
        press_x()  # save

threading.Thread(target=sender, daemon=True).start()
threading.Thread(target=operator, daemon=True).start()
root = sys.argv[1]; shutil.rmtree(root, ignore_errors=True)
sys.argv = ["lerobot-record", "--robot.type=openarm_isaac",
    "--teleop.type=openarm_quest", f"--teleop.port={PORT}", "--dataset.repo_id=local/chain_sim", f"--dataset.root={root}",
    "--dataset.push_to_hub=false", "--dataset.single_task=reach out and back", "--dataset.num_episodes=3",
    "--dataset.episode_time_s=10", "--dataset.reset_time_s=1", "--dataset.fps=30",
    "--dataset.streaming_encoding=true", "--dataset.encoder_threads=2", "--play_sounds=false", "--display_data=false"]
rec.main(); Q["run"] = False
info = json.load(open(f"{root}/meta/info.json"))
print("CHAIN RECORD SIM:", info["robot_type"], info["total_episodes"], "episodes,", info["total_frames"], "frames, features:",
      {k: v["shape"] for k, v in info["features"].items()})
