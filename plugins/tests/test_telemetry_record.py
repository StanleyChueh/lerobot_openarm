"""The OFFICIAL lerobot-record main() with --robot.telemetry_dir, on a fake DM bus: every recorded dataset frame
must be found in the telemetry, on the policy clock. No hardware, no CAN socket (telemetry.make_tap is faked).

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_telemetry_record.py <scratch dir>

Checks, per saved episode: align_to_dataset() finds exactly one window of consecutive telemetry rows whose q
equals observation.state bit-for-bit; dataset action[k] == desired on the next row (the request lerobot
records); the wire command on the next row equals the host's sent command within one LSB; and every frame of
the episode is traced to a wire frame (no stale rows on a healthy bus).
"""
import glob
import json
import socket
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np

import telemetry_fakes as tf
from telemetry_fakes import LIMITS, FakeWire, lsb

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
import robots.umeow_openarm_follower.openarm_follower as fol
from lerobot_robot_openarm_umeow import telemetry as tel
from lerobot_robot_openarm_umeow.common import MOTOR_KEYS

ROOT = Path(sys.argv[1])
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


WIRE = FakeWire()
fol.oa.OpenArm = mock.MagicMock(side_effect=lambda *a, **k: mock.MagicMock())
fol.OpenArmFollower.connect = lambda self, calibrate=False: setattr(self, "_is_connected", True)
fol.OpenArmFollower.disconnect = lambda self: setattr(self, "_is_connected", False)
fol.OpenArmFollower.get_observation = lambda self: WIRE.read()
fol.OpenArmFollower.send_action = lambda self, action, vel: (
    WIRE.command({k: float(action[k]) for k in MOTOR_KEYS}, vel, {}, 50.0, 2.0), action)[1]
fol.OpenArmFollower._motor_channels = lambda self: WIRE.channels()
tel.make_tap = lambda port, ids, fd=True: WIRE.taps["R" if port == "can0" else "L"]

from lerobot_robot_openarm_umeow import OpenArmUmeow  # noqa: E402

_orig_connect = OpenArmUmeow.connect
READY = {}


def _connect(self, calibrate=True):
    _orig_connect(self, calibrate)
    READY["t"] = time.time()


OpenArmUmeow.connect = _connect
PORT = 5995
RUN = {"on": True}


def quest():
    """test_record_mock.py's scripted operator, but X every 6 s (the record gate waits for X before each episode)
    and the right hand keeps sweeping +-5 cm so every episode moves -- alignment by state needs motion (see align_to_dataset)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose = lambda x: {"x": x, "y": 1.1, "z": 0.3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}  # noqa: E731
    while RUN["on"]:
        el = time.time() - READY.get("t", float("inf"))
        msg = {"rc": pose(0.25 + 0.05 * np.sin(max(0.0, el - 1.5))), "lc": pose(-0.25), "rf": pose(0) | {"y": 1.5},
               "rt": 1.0 if el > 3.0 else 0.0, "lt": 0.0, "x": el >= 1.0 and (el - 1.0) % 6.0 < 0.1, "y": False, "a": False, "b": False, "v": 0}
        s.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT))
        time.sleep(1 / 72)


threading.Thread(target=quest, daemon=True).start()
ds_root, tel_root = ROOT / "ds", ROOT / "telemetry"
sys.argv = ["lerobot-record",
            "--robot.type=openarm_umeow", "--robot.assume_yes=true", f"--robot.telemetry_dir={tel_root}",
            "--teleop.type=openarm_quest", f"--teleop.port={PORT}", "--teleop.episode_buttons=false",
            "--dataset.repo_id=local/openarm_telemetry_mock", f"--dataset.root={ds_root}", "--dataset.push_to_hub=false",
            "--dataset.single_task=mock telemetry", "--dataset.num_episodes=2", "--dataset.episode_time_s=4",
            "--dataset.reset_time_s=1", "--dataset.fps=30", "--display_data=false", "--play_sounds=false"]
from lerobot.scripts.lerobot_record import main  # noqa: E402

main()
RUN["on"] = False

import pandas as pd  # noqa: E402

sessions = sorted(tel_root.iterdir())
df, meta = tel.load(sessions[-1])
info = json.load(open(ds_root / "meta/info.json"))
names = info["features"]["observation.state"]["names"]
frames = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{ds_root}/data/**/*.parquet", recursive=True)))
check("A  one telemetry session, closed cleanly, no snapshot or writer errors",
      len(sessions) == 1 and meta["closed"] and meta["errors"] == {"snapshot": 0, "writer": 0},
      f"{meta['rows']} rows, {info['total_episodes']} episodes, {len(frames)} frames")
for ep, g in frames.groupby("episode_index"):
    g = g.sort_values("frame_index")
    states, actions = np.stack(g["observation.state"]), np.stack(g["action"])
    try:
        start = tel.align_to_dataset(df, states, names)
    except ValueError as e:
        check(f"B  episode {ep}: aligned to telemetry", False, str(e))
        continue
    rows = df.iloc[start:start + len(g) + 1]
    contiguous = rows["tick"].diff().iloc[1:].eq(1).all() and (rows["kind"] == 0).all()
    jn = [n.removesuffix(".pos") for n in names]
    desired = rows[[f"{j}.desired" for j in jn]].to_numpy()[1:].astype(np.float32)
    cmd_gap = max((rows[f"{j}.cmd_q"].iloc[1:] - rows[f"{j}.sent"].iloc[1:]).abs().max() / lsb(LIMITS[j], "q") for j in jn)
    stale = int(rows[[f"{j}.stale" for j in jn]].iloc[:-1].to_numpy().sum())
    check(f"B  episode {ep}: its {len(g)} frames are ONE run of consecutive telemetry rows (from row {start})",
          contiguous)
    check(f"B  episode {ep}: action[k] == desired on row k+1 (the request lerobot records, before the step limit)",
          np.array_equal(desired, actions))
    check(f"B  episode {ep}: the wire command on row k+1 equals the host's sent command, within one LSB",
          cmd_gap <= 1.0, f"max {cmd_gap:.2f} LSB")
    check(f"B  episode {ep}: every frame traced to a wire frame on a healthy bus (stale = 0 throughout)", stale == 0)

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
