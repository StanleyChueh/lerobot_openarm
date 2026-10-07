"""Run the OFFICIAL lerobot-teleoperate main() end to end with the CAN hardware mocked out.

The follower's hardware methods are replaced by a perfect-tracking fake (state = last command), the
Quest by synthetic UDP packets. Everything between them -- plugin discovery, the teleop loop, the
openarm_umeow wrapper (approach, step limit, squeeze, return to rest) -- is the real code. No CAN
traffic: the follower's hardware methods are patched out.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_teleoperate_mock.py
"""
import json, socket, sys, threading, time
from unittest import mock

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
import robots.umeow_openarm_follower.openarm_follower as fol

fol.oa.OpenArm = mock.MagicMock()
KEYS = [f"{p}J{i}.pos" for i in range(1, 9) for p in ("R", "L")]
fake = {"state": {k: 0.0 for k in KEYS}, "sends": [], "connected": False}

def connect(self, calibrate=False): self._is_connected = True
def get_observation(self): return dict(fake["state"])
def send_action(self, action, target_vel):
    fake["sends"].append((time.perf_counter(), dict(action), dict(self.gripper_squeeze_tau)))
    fake["state"] = {k: float(action[k]) for k in KEYS}
    return action
def disconnect(self): self._is_connected = False; fake["disconnected"] = True
for name, fn in dict(connect=connect, get_observation=get_observation, send_action=send_action, disconnect=disconnect).items():
    setattr(fol.OpenArmFollower, name, fn)

from lerobot_robot_openarm_umeow import OpenArmUmeow
_orig_connect = OpenArmUmeow.connect
def _connect(self, calibrate=True):
    _orig_connect(self, calibrate); fake["ready_t"] = time.time()
OpenArmUmeow.connect = _connect
PORT = 5997
q = {"rx": 0.25, "x": False, "rt": 0.0, "run": True}
def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose = lambda x: {"x": x, "y": 1.1, "z": 0.3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    while q["run"]:
        el = time.time() - fake.get("ready_t", float("inf"))  # operator script starts once the arm is at home
        # X at 1 s (anchor), then sweep the right hand 10 cm and close the right trigger.
        msg = {"rc": pose(0.25 + (0.10 * min(1.0, max(0.0, el - 1.5) / 1.5))), "lc": pose(-0.25), "rf": pose(0) | {"y": 1.5},
               "rt": 1.0 if el > 3.0 else 0.0, "lt": 0.0, "x": 1.0 <= el < 1.1, "y": False, "a": False, "b": False, "v": 0}
        s.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT)); time.sleep(1 / 72)
threading.Thread(target=sender, daemon=True).start()

sys.argv = ["lerobot-teleoperate", "--robot.type=openarm_umeow", "--robot.assume_yes=true",
    "--teleop.type=openarm_quest", f"--teleop.port={PORT}", "--teleop.episode_buttons=false",
    "--fps=30", "--teleop_time_s=5"]
from lerobot.scripts.lerobot_teleoperate import main
main()
q["run"] = False
sends = fake["sends"]
print(f"TELEOP: {len(sends)} commands, disconnected={fake.get('disconnected')}, squeeze seen={any(sq['R'] for _, _, sq in sends)}, ended at rest={all(abs(v) < 1e-6 for v in sends[-1][1].values())}")
