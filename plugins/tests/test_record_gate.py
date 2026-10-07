"""Episodes start on the operator's first X: a mocked 2-episode lerobot-record with a scripted operator.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_record_gate.py /tmp/openarm_gate_ds

No hardware (the follower's CAN methods are patched out), no key presses on the desktop (the Quest's
episode keys are applied straight to lerobot-record's event flags). The operator:
  episode 0: waits 2 s at HELD (must NOT be recorded), X, drives 2 s, X (= save) -> return home;
  reset:     X 1 s after the arms are home, inside a 10 s reset phase (must end the reset early);
  episode 1: drives until lerobot's 3 s episode timer ends it (the arms must still return home).
"""

import json
import math
import shutil
import socket
import sys
import threading
import time
from unittest import mock

import numpy as np

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
import robots.umeow_openarm_follower.openarm_follower as fol
from lerobot_robot_openarm_umeow.rerun_status import BOARD
from lerobot_robot_openarm_umeow.shared import TELEOP_STATE
from lerobot_teleoperator_openarm_quest import OpenArmQuest

fol.oa.OpenArm = mock.MagicMock()
KEYS = [f"{p}J{i}.pos" for i in range(1, 9) for p in ("R", "L")]
fake = {"state": {k: 0.0 for k in KEYS}}
fol.OpenArmFollower.connect = lambda self, calibrate=False: setattr(self, "_is_connected", True)
fol.OpenArmFollower.disconnect = lambda self: setattr(self, "_is_connected", False)
fol.OpenArmFollower.get_observation = lambda self: dict(fake["state"])


def _send(self, action, vel):
    fake["state"] = {k: float(action[k]) for k in KEYS}
    return action


fol.OpenArmFollower.send_action = _send

# lerobot-record's event flags, and the Quest's episode keys applied to them directly.
import lerobot.scripts.lerobot_record as rec
from lerobot.utils.keyboard_input import apply_recording_control

EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
rec.init_keyboard_listener = lambda: (None, EVENTS)
OpenArmQuest._press_episode_key = lambda self, key: apply_recording_control(key, EVENTS)

# ── scripted operator on a synthetic Quest ──────────────────────────────────────
PORT = 5999
Q = {"rx": 0.25, "x": False, "run": True}
log = []  # (time, event)


def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose = lambda x: {"x": x, "y": 1.1, "z": 0.3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    while Q["run"]:
        msg = {"rc": pose(Q["rx"]), "lc": pose(-0.25), "rf": pose(0.0) | {"y": 1.5}, "rt": 0.0, "lt": 0.0,
               "x": Q["x"], "y": False, "a": False, "b": False, "v": 0, "vr": 0, "vl": 0}
        s.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT))
        time.sleep(1 / 72)


def press_x(tag):
    log.append((time.perf_counter(), tag))
    Q["x"] = True
    time.sleep(0.15)
    Q["x"] = False


def operator():
    def wait_state(s, timeout=60):
        t0 = time.perf_counter()
        while TELEOP_STATE.snapshot()["state"] != s and time.perf_counter() - t0 < timeout:
            time.sleep(0.02)

    while BOARD.phase != "WAITING":  # the robot is up and lerobot is in episode 0, waiting for X
        time.sleep(0.02)
    time.sleep(2.0)  # episode 0 is "running" in lerobot's terms, but nothing must be recorded yet
    press_x("X start ep0")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:  # drive
        Q["rx"] = 0.25 + 0.04 * math.sin(3 * (time.perf_counter() - t0))
        time.sleep(0.01)
    press_x("X save ep0")
    wait_state("RETURNING", 5)
    wait_state("HELD")
    log.append((time.perf_counter(), "home after ep0"))
    time.sleep(1.0)
    press_x("X start ep1 (during reset)")
    t0 = time.perf_counter()
    while Q["run"] and TELEOP_STATE.snapshot()["state"] == "LIVE":  # drive until lerobot's timer ends it
        Q["rx"] = 0.25 + 0.04 * math.sin(3 * (time.perf_counter() - t0))
        time.sleep(0.01)
    log.append((time.perf_counter(), f"after ep1: {TELEOP_STATE.snapshot()['state']}"))


threading.Thread(target=sender, daemon=True).start()
threading.Thread(target=operator, daemon=True).start()

root = sys.argv[1]
shutil.rmtree(root, ignore_errors=True)
sys.argv = ["lerobot-record",
    "--robot.type=openarm_umeow", "--robot.assume_yes=true",
    "--teleop.type=openarm_quest", f"--teleop.port={PORT}",
    "--dataset.repo_id=local/openarm_gate_mock", f"--dataset.root={root}", "--dataset.push_to_hub=false",
    "--dataset.single_task=gate test", "--dataset.num_episodes=2", "--dataset.episode_time_s=3",
    "--dataset.reset_time_s=10", "--dataset.fps=30", "--play_sounds=false", "--display_data=false"]
t_start = time.perf_counter()
rec.main()
t_total = time.perf_counter() - t_start
Q["run"] = False

# ── checks ─────────────────────────────────────────────────────────────────────
import glob

import pandas as pd

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


info = json.load(open(f"{root}/meta/info.json"))
df = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{root}/data/**/*.parquet", recursive=True))
n = df.groupby("episode_index").size().to_dict()
print("\noperator log:", [(round(t - t_start, 1), e) for t, e in log])
print("frames per episode:", n, "| session", round(t_total, 1), "s")
check("2 episodes saved", info["total_episodes"] == 2, str(info["total_episodes"]))
check("episode 0 = X to 2nd X only (~2 s = ~60 frames), the 2 s wait before X not recorded",
      50 <= n.get(0, 0) <= 75, f"{n.get(0)} frames")
check("episode 1 = lerobot's 3 s timer from its X (~90 frames)", 80 <= n.get(1, 0) <= 95, f"{n.get(1)} frames")
names = info["features"]["action"]["names"]
arm = [j for j, k in enumerate(names) if not k.endswith("8.pos")]
for ep in (0, 1):
    a = np.stack(df[df.episode_index == ep]["action"])
    check(f"episode {ep} starts at the home pose and then moves",
          np.abs(a[0, arm] - a[1, arm]).max() < 0.02 and np.ptp(a[:, arm], axis=0).max() > 0.02,
          f"first-step change {np.abs(a[0, arm] - a[1, arm]).max():.3f}, range {np.ptp(a[:, arm], axis=0).max():.3f} rad")
check("X during the 10 s reset ended it early", t_total < 30, f"session {t_total:.0f} s")
after = [e for _, e in log if e.startswith("after ep1")]
check("episode 1 ended by the timer: the arms returned home anyway", bool(after) and after[0].endswith(("RETURNING", "HELD")),
      after[0] if after else "no log")
print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
