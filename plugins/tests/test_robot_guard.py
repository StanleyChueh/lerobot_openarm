"""openarm_umeow's safety guard against a mocked follower. No hardware, no CAN traffic.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_robot_guard.py
"""

import sys
import time
from unittest import mock

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
import robots.umeow_openarm_follower.openarm_follower as fol
from lerobot_robot_openarm_umeow import OpenArmUmeow, OpenArmUmeowConfig
from lerobot_robot_openarm_umeow.common import MOTOR_KEYS
from lerobot_robot_openarm_umeow.shared import ROBOT_STATE

fol.oa.OpenArm = mock.MagicMock()
fake = {"state": {k: 0.0 for k in MOTOR_KEYS}, "offset": 0.0, "sent": []}
fol.OpenArmFollower.connect = lambda self, calibrate=False: setattr(self, "_is_connected", True)
fol.OpenArmFollower.disconnect = lambda self: setattr(self, "_is_connected", False)
fol.OpenArmFollower.get_observation = lambda self: {
    k: v + (fake["offset"] if k == "RJ4.pos" else 0.0) for k, v in fake["state"].items()
}


def _send(self, action, vel):
    fake["sent"].append(dict(action))
    fake["state"] = {k: float(action[k]) for k in MOTOR_KEYS}
    return action


fol.OpenArmFollower.send_action = _send

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


r = OpenArmUmeow(OpenArmUmeowConfig(start_pose="none", return_to_rest=False))
r.connect()
cmd = {k: 0.0 for k in MOTOR_KEYS}


def tick(**changes):
    time.sleep(1 / 30)  # lerobot's loops tick at 30 Hz; the speed clamp is per elapsed time
    cmd.update(changes)
    r.get_observation()
    return r.send_action(dict(cmd))


for i in range(30):  # normal motion, 0.6 rad/s
    sent = tick(**{"RJ4.pos": 0.02 * (i + 1)})
check("1  normal motion is executed", abs(sent["RJ4.pos"] - 0.6) < 1e-6 and ROBOT_STATE.fault() is None)

held = dict(sent)
sent = tick(**{"RJ2.pos": 0.5})  # 0.5 rad jump in one tick
check("2  0.5 rad jump in one tick is REFUSED (arm holds)", sent == held and ROBOT_STATE.fault() is not None,
      ROBOT_STATE.fault() or "")
for _ in range(5):
    sent = tick()
check("2  stays held while requests stay far", sent == held and ROBOT_STATE.fault() is not None)
sent = tick(**{"RJ2.pos": held["RJ2.pos"] + 0.05})
check("3  released once requests come back near the held pose", ROBOT_STATE.fault() is None
      and abs(sent["RJ2.pos"] - held["RJ2.pos"]) > 0, f"RJ2 sent {sent['RJ2.pos']:.3f}")

fake["offset"] = -0.7  # measured RJ4 0.7 rad away from the command (blocked / dragged arm)
sent_before = dict(fake["sent"][-1])
sent = tick(**{"RJ4.pos": cmd["RJ4.pos"] + 0.01})
check("4  request 0.7 rad from the measured joint is REFUSED", ROBOT_STATE.fault() is not None
      and sent == sent_before, ROBOT_STATE.fault() or "")
fake["offset"] = 0.0

r.umeow_config.max_command_jump = 10.0  # isolate the speed clamp
r._fault = None
sent = tick(**{"LJ1.pos": cmd["LJ1.pos"] + 0.2})
check("5  speed clamp: one tick moves <= max_joint_speed * dt", sent["LJ1.pos"] <= r.umeow_config.max_joint_speed / 30 * 1.3,
      f"moved {sent['LJ1.pos']:.3f} rad of 0.2")

# Motor watchdog: a motor that trips its protection switches itself off, while its last position is still
# reported. The status nibble is the only sign.
import contextlib  # noqa: E402
import io  # noqa: E402

r.umeow_config.max_command_jump = 0.25
r._fault = None
for _ in range(3):
    tick()
STATUS = {k.replace(".pos", ""): 0x1 for k in MOTOR_KEYS}
r.get_feedback_status = lambda: dict(STATUS)
r.get_motor_health = lambda: {"RJ7": {"t_mos": 41, "t_rotor": 38}}
out = io.StringIO()
with contextlib.redirect_stdout(out):
    STATUS["RJ7"] = 0xA  # OVERCURRENT
    time.sleep(0.6)
    held = dict(fake["sent"][-1])
    sent = tick(**{"RJ1.pos": cmd["RJ1.pos"] + 0.02})
check("6  a motor reporting OVERCURRENT -> the arm HOLDS, the terminal names the motor, fault and temperatures",
      ROBOT_STATE.fault() is not None and "RJ7 OVERCURRENT" in ROBOT_STATE.fault() and sent == held
      and "MOTOR FAULT" in out.getvalue() and "MOS 41 C" in out.getvalue(), ROBOT_STATE.fault() or "")
for _ in range(3):
    sent = tick(**{"RJ1.pos": held["RJ1.pos"]})  # requests at the held pose: a normal hold would release
check("6  ... and the hold is NOT released while the motor stays faulted", ROBOT_STATE.fault() is not None)
with contextlib.redirect_stdout(out):
    STATUS["RJ7"] = 0x1
    time.sleep(0.6)
    tick(**{"RJ1.pos": held["RJ1.pos"]})
    sent = tick(**{"RJ1.pos": held["RJ1.pos"]})
check("6  ... released once the motor reports enabled and requests are at the held pose",
      ROBOT_STATE.fault() is None and "enabled again" in out.getvalue(), ROBOT_STATE.fault() or "")

# A joint that does not reach a steady command (blocked, or limp without a status report).
out = io.StringIO()
with contextlib.redirect_stdout(out):
    fake["offset"] = 0.3  # RJ4 measured 0.3 rad off, under max_tracking_error: no hold, but not following
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.5:
        tick()
    fake["offset"] = 0.0
    tick()
check("7  a joint 0.3 rad from a steady command for 2 s is reported as NOT following, then as following again",
      "RJ4.pos is NOT following" in out.getvalue() and "RJ4.pos is following its command again" in out.getvalue(),
      out.getvalue().strip().splitlines()[0] if out.getvalue().strip() else "nothing printed")

r.disconnect()
print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
