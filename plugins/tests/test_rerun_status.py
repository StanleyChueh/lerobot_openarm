"""The rerun status panel during a mocked 2-episode lerobot-record. No hardware, no window.

Runs lerobot-record's own main() with --display_data=true against a headless rerun gRPC server this
script starts, and checks what the status panel showed and that lerobot's layout carries it.

The follower's hardware methods are replaced by a perfect-tracking fake (state = last command), the
Quest by synthetic UDP packets. Everything between them -- plugin discovery, the record loop, the
openarm_umeow wrapper (approach, step limit, squeeze, return to rest) and the LeRobot dataset
writer -- is the real code. No CAN traffic: the follower's hardware methods are patched out.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_rerun_status.py /tmp/openarm_mock_ds_rerun
"""
import json, socket, sys, threading, time

import numpy as np
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
def _waiting_long() -> bool:
    """Press X once lerobot has been waiting for it for 0.5 s (every episode, like an operator)."""
    from lerobot_robot_openarm_umeow.rerun_status import BOARD

    return BOARD.phase == "WAITING" and time.perf_counter() - BOARD.phase_t > 0.5


def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose = lambda x: {"x": x, "y": 1.1, "z": 0.3, "qx": 0, "qy": 0, "qz": 0, "qw": 1}
    while q["run"]:
        el = time.time() - fake.get("ready_t", float("inf"))  # operator script starts once the arm is at home
        # X at 1 s (anchor), then sweep the right hand 10 cm and close the right trigger.
        msg = {"rc": pose(0.25 + (0.10 * min(1.0, max(0.0, el - 1.5) / 1.5))), "lc": pose(-0.25), "rf": pose(0) | {"y": 1.5},
               "rt": 1.0 if el > 3.0 else 0.0, "lt": 0.0, "x": _waiting_long(), "y": False, "a": False, "b": False, "v": 0}
        s.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT)); time.sleep(1 / 72)
threading.Thread(target=sender, daemon=True).start()

import os
import shutil
import signal
import subprocess

import rerun as rr

from lerobot_robot_openarm_umeow.rerun_status import BOARD

root = sys.argv[1]
shutil.rmtree(root, ignore_errors=True)
RR_PORT = 9915
server = subprocess.Popen([os.path.join(os.path.dirname(sys.executable), "rerun"), "--serve-grpc", "--port", str(RR_PORT)],
                          env={**os.environ, "DISPLAY": ""}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          start_new_session=True)  # its own group: the launcher spawns the real server binary
time.sleep(3)

panels, blueprints = [], []
_render = BOARD._render


def render():
    text = _render()
    panels.append(text)
    return text


BOARD._render = render
_send = rr.send_blueprint
rr.send_blueprint = lambda bp, *a, **k: (blueprints.append(bp), _send(bp, *a, **k))[1]

sys.argv = ["lerobot-record",
    "--robot.type=openarm_umeow", "--robot.assume_yes=true",
    "--teleop.type=openarm_quest", f"--teleop.port={PORT}", "--teleop.episode_buttons=false",
    "--dataset.repo_id=local/openarm_quest_mock", f"--dataset.root={root}", "--dataset.push_to_hub=false",
    "--dataset.single_task=mock plate wipe", "--dataset.num_episodes=2", "--dataset.episode_time_s=4",
    "--dataset.reset_time_s=2", "--dataset.fps=30", "--play_sounds=false",
    "--display_data=true", "--display_ip=127.0.0.1", f"--display_port={RR_PORT}"]
from lerobot.scripts.lerobot_record import main

try:
    main()
finally:
    q["run"] = False
    os.killpg(server.pid, signal.SIGTERM)

shown = []
for p in panels:
    h = p.splitlines()[0].rsplit(" · ", 1)[0]  # drop the running seconds
    if not shown or shown[-1] != h:
        shown.append(h)
print("\nstatus panel headlines, in order:")
for h in shown:
    print("   ", h)

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


expected = ["starting", "WAITING for X", "RECORDING · episode 0 of 2", "RESETTING", "SAVING · episode 0",
            "WAITING for X", "RECORDING · episode 1 of 2",
            "SAVING · episode 1", "FINALIZING", "STOPPING", "returning to rest"]
it = iter(shown)
check("phases appear in order: " + " -> ".join(expected), all(any(e in h for h in it) for e in expected))
notes = {l for p in panels for l in p.splitlines() if l.startswith("last:")}
check("saved counts shown", {"last: episode 0 SAVED (1 in the dataset)", "last: episode 1 SAVED (2 in the dataset)"} <= notes, str(sorted(notes)))
check("Quest state shown", any("**Quest:** ▶ LIVE" in p for p in panels) and any("**Quest:** ⏸ HELD" in p for p in panels))
def _views(c):
    out = []
    for x in [x for x in (getattr(c, "contents", None) or []) if not isinstance(x, str)]:
        out += [type(x).__name__ + ":" + str(getattr(x, "origin", ""))] + _views(x)
    return out


top = blueprints[0].root_container.contents[0] if blueprints else None
names = _views(top) if top is not None else []
check("layout: status panel top-left, network panel + latency plot top-right, lerobot's views below",
      len(blueprints) == 1 and type(top).__name__ == "Horizontal"
      and names[0].startswith("TextDocumentView") and any(n.startswith("TimeSeriesView") for n in names), str(names))
print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
