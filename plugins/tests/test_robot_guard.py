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
check("5  speed clamp: one tick moves <= max_joint_speed * dt", sent["LJ1.pos"] <= 1.0 / 30 * 1.3,
      f"moved {sent['LJ1.pos']:.3f} rad of 0.2")

r.disconnect()
print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
