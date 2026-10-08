"""lerobot-record episodes driven by the Quest, through the official lerobot-record main(). No hardware.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_record_gate.py /tmp/openarm_gate_ds

The follower's CAN methods are patched out (perfect tracking) and the keyboard listener is replaced by
lerobot-record's own event flags with no key ever pressed: the Quest's save / discard reach them through
record_gate.py, as in a real session. The scripted operator:
  episode 0      waits 2 s (not recorded), X, drives 2 s with the right gripper closed, a 30 cm snap ->
                 PAUSED, X -> back home still recording, X -> continues, drives, X -> the return home is
                 recorded and the episode saved on arrival, grippers still closed in its last frame;
  reset          X 1 s into it must be ignored (the reset runs its full length);
  episode 1      X, drives, Y -> discarded, re-recorded;
  episode 1 again X, drives until episode_time_s -> return home and save, the return included.
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

import lerobot.scripts.lerobot_record as rec  # noqa: E402

EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
rec.init_keyboard_listener = lambda: (None, EVENTS)  # no keyboard; the Quest's decisions set these flags

# lerobot's phase announcements, independent of the rerun panel (which only follows them while rerun runs).
PHASE = {"name": None}
_say = rec.log_say


def _log_say(text, *args, **kwargs):
    if text.startswith("Reset the environment"):
        PHASE["name"] = "RESETTING"
    elif text.startswith("Recording episode"):
        PHASE["name"] = "EPISODE"
    return _say(text, *args, **kwargs)


rec.log_say = _log_say

PORT = 5999
Q = {"rx": 0.25, "x": False, "y": False, "rt": 0.0, "run": True}
log = []  # (time, event)


def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose = lambda x: {"x": x, "y": 1.1, "z": 0.3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    while Q["run"]:
        msg = {"rc": pose(Q["rx"]), "lc": pose(-0.25), "rf": pose(0.0) | {"y": 1.5}, "rt": Q["rt"], "lt": 0.0,
               "x": Q["x"], "y": Q["y"], "a": False, "b": False, "v": 0, "vr": 0, "vl": 0}
        s.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT))
        time.sleep(1 / 72)


def press(k, tag):
    log.append((time.perf_counter(), tag))
    Q[k] = True
    time.sleep(0.15)
    Q[k] = False
    time.sleep(0.1)


def wait_phase(phase, timeout=60):
    """WAITING: the record gate waits for X (it sets BOARD.phase itself); RESETTING: lerobot's reset phase."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if (BOARD.phase == phase) if phase == "WAITING" else (PHASE["name"] == phase):
            return
        time.sleep(0.02)
    raise SystemExit(f"operator: timed out waiting for {phase}")


def drive(seconds, until_not_live=False):
    t0 = time.perf_counter()
    while Q["run"]:
        el = time.perf_counter() - t0
        if (not until_not_live and el > seconds) or (until_not_live and TELEOP_STATE.snapshot()["state"] != "LIVE"):
            break
        Q["rx"] = 0.25 + 0.05 * math.sin(2.0 * el)
        time.sleep(0.01)


def operator():
    wait_phase("WAITING")
    time.sleep(2.0)  # lerobot is in episode 0, but nothing must be recorded yet
    press("x", "X start ep0")
    Q["rt"] = 1.0  # close the right gripper
    drive(2.0)
    Q["rx"] += 0.30  # a 30 cm snap in one packet -> PAUSED
    time.sleep(0.4)
    log.append((time.perf_counter(), f"after snap: {TELEOP_STATE.snapshot()['state']}"))
    press("x", "X recover ep0")
    t0 = time.perf_counter()
    while TELEOP_STATE.snapshot()["reason"] != "in episode" and time.perf_counter() - t0 < 20:
        time.sleep(0.02)
    log.append((time.perf_counter(), f"recovered: {TELEOP_STATE.snapshot()['state']} {TELEOP_STATE.snapshot()['reason']}"))
    Q["rx"] = 0.25
    time.sleep(0.3)
    press("x", "X continue ep0")
    drive(1.0)
    press("x", "X save ep0")
    wait_phase("RESETTING")
    log.append((time.perf_counter(), f"reset starts, teleop {TELEOP_STATE.snapshot()['state']}"))
    time.sleep(1.0)
    press("x", "X during reset")
    log.append((time.perf_counter(), f"after X during reset: {TELEOP_STATE.snapshot()['state']}"))
    Q["rt"] = 0.0
    wait_phase("WAITING")
    log.append((time.perf_counter(), "waiting for ep1"))
    press("x", "X start ep1")
    drive(1.0)
    press("y", "Y discard ep1")
    wait_phase("WAITING")
    log.append((time.perf_counter(), "waiting for ep1 (re-record)"))
    press("x", "X start ep1 again")
    drive(0, until_not_live=True)  # until the time limit sends the arms home
    log.append((time.perf_counter(), f"ep1 time limit: {TELEOP_STATE.snapshot()['state']}"))


threading.Thread(target=sender, daemon=True).start()
threading.Thread(target=operator, daemon=True).start()

root = sys.argv[1]
shutil.rmtree(root, ignore_errors=True)
sys.argv = ["lerobot-record",
    "--robot.type=openarm_umeow", "--robot.assume_yes=true",
    "--teleop.type=openarm_quest", f"--teleop.port={PORT}",
    "--dataset.repo_id=local/openarm_gate_mock", f"--dataset.root={root}", "--dataset.push_to_hub=false",
    "--dataset.single_task=gate test", "--dataset.num_episodes=2", "--dataset.episode_time_s=6",
    "--dataset.reset_time_s=4", "--dataset.fps=30", "--play_sounds=false", "--display_data=false"]
t_start = time.perf_counter()
rec.main()
Q["run"] = False

# ── checks ─────────────────────────────────────────────────────────────────────
import glob  # noqa: E402

import pandas as pd  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def at(tag):
    return next((t for t, e in log if e.startswith(tag)), None)


info = json.load(open(f"{root}/meta/info.json"))
df = pd.concat(pd.read_parquet(f) for f in glob.glob(f"{root}/data/**/*.parquet", recursive=True))
names = info["features"]["action"]["names"]
arm = [j for j, k in enumerate(names) if not k.endswith("8.pos")]
rj8 = names.index("RJ8.pos")
eps = {e: np.stack(df[df.episode_index == e]["action"]) for e in sorted(df.episode_index.unique())}
print("\noperator log:", [(round(t - t_start, 1), e) for t, e in log])
print("frames per episode:", {e: len(a) for e, a in eps.items()})

check("2 episodes saved (the discarded attempt is not)", info["total_episodes"] == 2, str(info["total_episodes"]))
a0, a1 = eps.get(0), eps.get(1)
home = a0[0, arm]
check("ep0 starts at home (the 2 s wait before X not recorded)", np.abs(a0[1, arm] - home).max() < 0.02)
check("ep0 includes the return: it ends back at home", np.abs(a0[-1, arm] - home).max() < 0.01,
      f"last frame {np.abs(a0[-1, arm] - home).max():.4f} rad from home")
t_x0, t_s0 = at("X start ep0"), at("X save ep0")
check("ep0 lasts past the saving X (drive, pause, recovery, drive, return), not cut at it",
      len(a0) / 30 > (t_s0 - t_x0) + 0.3, f"{len(a0) / 30:.1f} s recorded, saving X at {t_s0 - t_x0:.1f} s")
snap = next((e for _, e in log if e.startswith("after snap")), "")
rec_ = next((e for _, e in log if e.startswith("recovered")), "")
check("ep0: the snap PAUSED the arms, X recovered home with the episode still recording",
      snap.endswith("PAUSED") and rec_ == "recovered: HELD in episode", f"{snap} | {rec_}")
check("ep0 last frame: right gripper still closed (opens only after the save)",
      abs(a0[-1, rj8] - 0.0) < 0.05, f"RJ8 {a0[-1, rj8]:.3f} (closed 0.0, open -1.298)")
after_reset_x = next((e for _, e in log if e.startswith("after X during reset")), "")
check("X during the reset is ignored", after_reset_x.endswith("HELD"), after_reset_x)
t_rs, t_w1 = at("reset starts"), at("waiting for ep1")
check("the reset ran its full 4 s (X did not end it early)", t_rs is not None and t_w1 is not None and t_w1 - t_rs > 3.5,
      f"{(t_w1 - t_rs) if t_rs and t_w1 else float('nan'):.1f} s")
check("ep1 = the re-recorded attempt: time limit 6 s, then the return, ending at home",
      len(a1) / 30 > 6.0 and np.abs(a1[-1, arm] - home).max() < 0.01,
      f"{len(a1) / 30:.1f} s, last frame {np.abs(a1[-1, arm] - home).max():.4f} rad from home")
print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
