"""Safety scenarios for the openarm_quest teleop, driven by a synthetic Quest. No hardware.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_quest_safety.py

Every check prints PASS/FAIL; the script exits non-zero if any fails.
"""

import json
import socket
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_openarm_umeow.common import motor_action_to_sim_joints, sim_joints_to_driver16
from lerobot_robot_openarm_umeow.shared import ROBOT_STATE
from lerobot_teleoperator_openarm_quest import OpenArmQuest, OpenArmQuestConfig
from lerobot_teleoperator_openarm_quest.ik_driver import ARM, GRIP, build_kinematics

PORT = 5998
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


# ── synthetic Quest: Unity left-handed poses, headset (rf) worn at the neck ───────
Q = {"rc": [0.25, 1.1, 0.3], "lc": [-0.25, 1.1, 0.3], "rf": [0.0, 1.3, 0.0], "rf_yaw": 0.0,
     "rt": 0.0, "lt": 0.0, "x": False, "y": False, "a": False, "b": False, "v": 0, "vr": 0, "vl": 0, "run": True, "jump": None, "silent": False}


def pose(p, yaw_deg=0.0):
    q = Rotation.from_euler("y", yaw_deg, degrees=True).as_quat()
    return {"x": p[0], "y": p[1], "z": p[2], "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3]}


def sender():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while Q["run"]:
        if Q["silent"]:  # the Quest (or the IK thread) goes quiet for a moment
            time.sleep(1 / 72)
            continue
        rc = list(Q["rc"])
        if Q["jump"] is not None:  # one-packet tracking glitch
            rc = [rc[0] + Q["jump"], rc[1], rc[2]]
            Q["jump"] = None
        msg = {"t": time.perf_counter(), "rc": pose(rc), "lc": pose(Q["lc"]), "rf": pose(Q["rf"], Q["rf_yaw"]), "rt": Q["rt"],
               "lt": Q["lt"], "x": Q["x"], "y": Q["y"], "a": Q["a"], "b": Q["b"], "v": Q["v"], "vr": Q["vr"], "vl": Q["vl"]}
        s.sendto(json.dumps(msg).encode(), ("127.0.0.1", PORT))
        time.sleep(1 / 72)


threading.Thread(target=sender, daemon=True).start()

keys = []
t = OpenArmQuest(OpenArmQuestConfig(port=PORT, episode_buttons=False))
t.connect()
t._press_episode_key = keys.append  # record episode keys instead of pressing them
t.driver._on_episode_key = keys.append
d = t.driver
home = d.home_driver.copy()
fk = build_kinematics(t.config.ik_xml, "home", t.config.ik_args)

# 100 Hz monitor of the shipped command, for speed checks.
log = []


def monitor():
    while Q["run"]:
        log.append((time.perf_counter(), d.command()))
        time.sleep(0.01)


threading.Thread(target=monitor, daemon=True).start()


def press(k):
    Q[k] = True
    time.sleep(0.12)
    Q[k] = False
    time.sleep(0.12)


def peak_speed(t0, t1, span=5):
    """Peak joint speed over `span` monitor samples (~50 ms): single 10 ms samples carry the timing jitter
    of reading a 500 Hz command from a 100 Hz thread, not real speed."""
    seg = [(ts, c) for ts, c in log if t0 <= ts <= t1]
    v = [np.abs(b[1][ARM] - a[1][ARM]).max() / (b[0] - a[0]) for a, b in zip(seg, seg[span:]) if b[0] > a[0]]
    return max(v) if v else 0.0


def state():
    return d.status()["state"]


def wait_home(timeout=20):
    t0 = time.perf_counter()
    while state() == "RETURNING" and time.perf_counter() - t0 < timeout:
        time.sleep(0.05)


def move(key, axis, dist, secs):
    start = list(Q[key])
    n = int(secs * 100)
    for i in range(n):
        Q[key] = list(start)
        Q[key][axis] = start[axis] + dist * (i + 1) / n
        time.sleep(0.01)


time.sleep(1.0)
c = d.command()
check("A  HELD at start: command is home", state() == "HELD" and np.allclose(c[ARM], home[ARM]))
check("A  HELD at start: grippers open", all(np.isclose(c[GRIP[s]], d.grip[s][0]) for s in ("right", "left")))

press("x")
time.sleep(0.5)
c = d.command()
check("B  X -> LIVE with no jump", state() == "LIVE" and np.abs(c[ARM] - home[ARM]).max() < 1e-3,
      f"max {np.abs(c[ARM] - home[ARM]).max():.4f} rad")

c0 = d.command()
t_a = time.perf_counter()
for i in range(100):  # headset at the neck swings 15 deg and sways 6 cm; hands still
    Q["rf_yaw"] = 15.0 * (i + 1) / 100
    Q["rf"] = [0.06 * (i + 1) / 100, 1.3, 0.0]
    time.sleep(0.01)
time.sleep(1.0)
c = d.command()
check("C  headset swings 15 deg / 6 cm, hands still: arms do not move",
      np.abs(c[ARM] - c0[ARM]).max() < 0.005 and state() == "LIVE", f"moved {np.abs(c[ARM] - c0[ARM]).max():.4f} rad")

pr0, _ = fk.fk_bimanual(d.command()[:8], d.command()[8:])
t_b = time.perf_counter()
move("rc", 0, 0.05, 1.0)  # right hand 5 cm along Unity x
time.sleep(1.5)
c = d.command()
pr1, pl1 = fk.fk_bimanual(c[:8], c[8:])
check("D  hand moves 5 cm -> EE follows ~5 cm", abs(np.linalg.norm(pr1[:3] - pr0[:3]) - 0.05) < 0.005,
      f"EE moved {np.linalg.norm(pr1[:3] - pr0[:3]) * 100:.1f} cm")
check("D  command speed <= max_joint_speed", peak_speed(t_b, time.perf_counter()) <= t.config.max_joint_speed * 1.15,
      f"peak {peak_speed(t_b, time.perf_counter()):.2f} rad/s")

c0 = d.command()
Q["jump"] = 0.15  # one packet 15 cm off, then back
time.sleep(0.5)
c = d.command()
check("E  15 cm tracking glitch -> PAUSED", state() == "PAUSED", d.status()["reason"])
check("E  arms frozen while PAUSED", np.abs(c[ARM] - c0[ARM]).max() < 0.02, f"moved {np.abs(c[ARM] - c0[ARM]).max():.3f} rad")
move("rc", 1, 0.05, 0.5)  # hand moves while paused: must not move the arm
c1 = d.command()
check("E  hand motion while PAUSED is ignored", np.abs(c1[ARM] - c[ARM]).max() < 1e-6)

keys.clear()
c_p = d.command()
press("x")
check("F  X in PAUSED -> RETURNING home, the episode keeps recording (no save / discard)",
      state() == "RETURNING" and keys == [], f"{state()}, keys {keys}")
wait_home()
c_h = d.command()
check("F  ... then HELD 'in episode' at home, grippers as they were",
      state() == "HELD" and d.status()["reason"] == "in episode" and np.allclose(c_h[ARM], home[ARM])
      and all(np.isclose(c_h[GRIP[s_]], c_p[GRIP[s_]]) for s_ in ("right", "left")))
d.x_allowed = lambda: False  # as between episodes: X must still continue an episode in progress
press("x")
time.sleep(0.5)
check("F  X continues the same episode from home, no jump", state() == "LIVE"
      and np.abs(d.command()[ARM] - home[ARM]).max() < 0.01)
d.x_allowed = lambda: True

c0 = d.command()
Q["vr"] = 2  # right controller loses tracking ...
move("rc", 2, 0.10, 0.5)  # ... while the hand moves 10 cm
c_lost = d.command()
Q["vr"] = 0  # tracked again, 10 cm away
time.sleep(0.5)
c = d.command()
check("G  lost tracking: arm holds", np.abs(c_lost[ARM] - c0[ARM]).max() < 0.01, f"{np.abs(c_lost[ARM] - c0[ARM]).max():.4f}")
check("G  tracking back 10 cm away: re-clutched, no jump", state() == "LIVE" and np.abs(c[ARM] - c0[ARM]).max() < 0.01,
      f"jump {np.abs(c[ARM] - c0[ARM]).max():.4f} rad")

move("rc", 0, -0.08, 1.5)  # drive somewhere, close the right gripper
Q["rt"] = 1.0
time.sleep(0.8)
c = d.command()
check("H  trigger closes the right gripper", np.isclose(c[GRIP["right"]], d.grip["right"][1]))
keys.clear()
press("x")
t_r = time.perf_counter()  # measure the return itself (the ramp is linear: its peak speed is the same throughout)
check("H  2nd X -> RETURNING, save NOT sent yet (the return is part of the episode)", state() == "RETURNING" and keys == [],
      f"keys {keys}")
press("x")
check("H  X ignored while RETURNING", state() == "RETURNING")
c_mid = d.command()
check("H  gripper stays closed during the return", np.isclose(c_mid[GRIP["right"]], d.grip["right"][1]))
while state() == "RETURNING" and time.perf_counter() - t_r < 20:
    time.sleep(0.05)
Q["rt"] = 0.0
c = d.command()
check("H  return speed <= return_speed", peak_speed(t_r, time.perf_counter()) <= t.config.return_speed * 1.15,
      f"peak {peak_speed(t_r, time.perf_counter()):.2f} rad/s")
check("H  arrives HELD at home", state() == "HELD" and np.allclose(c[ARM], home[ARM]))
check("H  save ('right') sent on arrival", keys == ["right"], f"keys {keys}")
check("H  grippers open after the return", all(np.isclose(c[GRIP[s]], d.grip[s][0]) for s in ("right", "left")))

# With a recorder (defer_gripper_open): the grippers stay as they were until the episode has ended.
d.defer_gripper_open = True
press("x")
Q["rt"] = 1.0
time.sleep(0.8)
keys.clear()
press("x")
wait_home()
c = d.command()
check("H2 recorder: save on arrival, grippers still closed", keys == ["right"] and np.isclose(c[GRIP["right"]], d.grip["right"][1]),
      f"keys {keys}")
d.release_grippers()
Q["rt"] = 0.0
time.sleep(0.3)
check("H2 recorder: grippers open after release_grippers()", np.isclose(d.command()[GRIP["right"]], d.grip["right"][0]))
d.defer_gripper_open = False

keys.clear()
press("x")
move("rc", 1, 0.05, 1.0)
press("y")
check("I  Y -> RETURNING, episode key 'left' (discard)", state() == "RETURNING" and keys == ["left"], f"keys {keys}")
while state() == "RETURNING":
    time.sleep(0.05)
check("I  Y return ends HELD at home", state() == "HELD" and np.allclose(d.command()[ARM], home[ARM]))

Q["vr"] = 2  # controller asleep / untracked while HELD
time.sleep(0.3)
press("x")
check("J  X refused while a controller is untracked (no stale anchor)", state() == "HELD")
Q["vr"] = 0
time.sleep(0.3)

press("x")
time.sleep(0.3)
held = d.command().copy()
held[ARM] += 0.05
from lerobot_robot_openarm_umeow.common import driver16_to_sim_joints, sim_joints_to_motor_action

ROBOT_STATE.publish(sim_joints_to_motor_action(driver16_to_sim_joints(held), t.calib), "test refusal")
time.sleep(0.3)
c = d.command()
check("K  robot refuses a command -> PAUSED at the robot's held pose",
      state() == "PAUSED" and np.abs(c[ARM] - held[ARM]).max() < 1e-4, d.status()["reason"])
ROBOT_STATE.clear()

time.sleep(0.2)
press("x")  # recover home, still recording
wait_home()
keys.clear()
press("y")
check("K  Y while HELD in episode -> discard ('left'), grippers open",
      keys == ["left"] and state() == "HELD" and d.status()["reason"] == ""
      and all(np.isclose(d.command()[GRIP[s_]], d.grip[s_][0]) for s_ in ("right", "left")), f"keys {keys}")
press("x")
time.sleep(0.3)
c0 = d.command()
Q["v"] = 2  # the headset itself loses tracking: controller poses are unreliable
move("lc", 0, 0.10, 0.5)
c_lost = d.command()
Q["v"] = 0
time.sleep(0.5)
c = d.command()
check("L  headset lost tracking: both arms hold", state() == "LIVE" and np.abs(c_lost[ARM] - c0[ARM]).max() < 0.01,
      f"{np.abs(c_lost[ARM] - c0[ARM]).max():.4f}")
check("L  headset tracked again: no jump", np.abs(c[ARM] - c0[ARM]).max() < 0.01, f"{np.abs(c[ARM] - c0[ARM]).max():.4f}")

keys.clear()
Q["a"] = True
time.sleep(0.15)
Q["a"] = False
Q["b"] = True
time.sleep(0.15)
Q["b"] = False
time.sleep(0.2)
check("M  A / B do nothing", keys == [] and state() == "LIVE", f"keys {keys}, {state()}")
press("y")
wait_home()
d.x_allowed = lambda: False
press("x")
check("N  X refused while the recorder does not allow it (reset phase)", state() == "HELD")
d.x_allowed = lambda: True

# Your failure: a gap in packets while the hand moves at a normal speed must not look like a glitch.
press("x")
time.sleep(0.5)
c0 = d.command()
Q["silent"] = True
move("rc", 0, 0.30, 0.3)  # 30 cm ...
time.sleep(0.3)            # ... over a 0.6 s gap = 0.5 m/s
Q["silent"] = False
time.sleep(0.5)
check("O  normal motion across a 0.6 s packet gap does NOT pause", state() == "LIVE", d.status()["reason"])
check("O  ... and the arm does not rush to catch up (re-anchored)", np.abs(d.command()[ARM] - c0[ARM]).max() < 0.02,
      f"moved {np.abs(d.command()[ARM] - c0[ARM]).max():.3f} rad")

from lerobot_robot_openarm_umeow.shared import TELEOP_STATE  # noqa: E402

# Faster than the joints may follow (but far below the 4 m/s glitch speed): the arm stops, and says which
# joint and what to do.
press("y")
wait_home()
press("x")
time.sleep(0.5)
move("rc", 1, 0.60, 0.25)  # 60 cm in 0.25 s (2.4 m/s): faster than the joints may follow
time.sleep(0.5)
reason = d.status()["reason"]
check("Q  too fast -> PAUSED, saying why (joint named + fix, or the hand-speed limit)",
      state() == "PAUSED" and (("fell" in reason and " J" in reason and "max_joint_speed" in reason) or "too fast" in reason),
      reason)
press("y")
wait_home()
press("x")
time.sleep(0.5)

REACH_LOG = []
_log_orig = TELEOP_STATE.log
TELEOP_STATE.log = lambda text: (REACH_LOG.append(text), _log_orig(text))[1]
move("rc", 2, -1.00, 4.0)  # push the right hand 1 m forward (Unity -z), slowly: past the arm's reach
time.sleep(1.0)
msg = TELEOP_STATE.snapshot()["message"]
reached_msg = any("cannot reach" in m for m in REACH_LOG)
check("P  out of reach: says so (terminal + rerun panel)", reached_msg, f"{REACH_LOG[-1:] or msg}")
check("P  ... and if the IK then jumps at the stretched-out pose, the arm stops (PAUSED)",
      state() == "LIVE" or "IK solution jumped" in d.status()["reason"], f"{state()} | {d.status()['reason']}")

Q["run"] = False
t.disconnect()
print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
