"""The reset ramp (RETURNING) must not wait for good on a robot that runs but cannot follow.

Seen 2026-10-21: recording in Isaac Sim, the session froze -- no motion, and X / Y did nothing. The ramp waited while
its command led the robot's last executed command by more than RETURN_LEAD_RAD, with no time limit; a robot that keeps
executing but cannot reach the command (an arm blocked by the can or the other arm, a guard hold) kept the teleop in
RETURNING, where X and Y are ignored. Checks, on the ramp logic alone (no Quest, no robot):

  A  a robot that stops sending (lerobot between phases) pauses the ramp -- the original fix still holds;
  B  a running robot that lags is waited for RETURN_LAG_WAIT_S, then the ramp carries on;
  C  a running robot that follows lets the ramp run at full speed.

  python plugins/tests/test_return_lag.py
"""

import threading

import numpy as np

from lerobot_teleoperator_openarm_quest.ik_driver import RETURN_LAG_WAIT_S, QuestIKDriver


def driver(held):
    d = QuestIKDriver.__new__(QuestIKDriver)  # only what _advance_return reads before the ramp ends
    d._lock = threading.Lock()
    d.home_driver = np.zeros(16)
    start = np.full(16, 0.5)
    d._command = start.copy()
    d._return = (start, 0.0, 0.0, 2.0)  # 2 s ramp
    d._return_robot = True
    d._return_lag_t0 = None
    d._held_command = held
    return d


def run(d, until, dt=0.02):
    t, a = 0.0, 0.0
    while t < until:
        t += dt
        d._advance_return(t)
        a = d._return[1]
        with d._lock:
            d._command = d._return[0] + a * (d.home_driver - d._return[0])
    return a


fails = 0


def check(name, ok, detail):
    global fails
    print(("PASS " if ok else "FAIL ") + name + f"  ({detail})")
    fails += not ok


a = run(driver(lambda: None), 1.5)
check("A  robot silent: ramp paused", a == 0.0, f"progress {a:.2f} after 1.5 s")

blocked = np.full(16, 0.5)  # executed, but the arm never moves off the start pose
a = run(driver(lambda: blocked.copy()), RETURN_LAG_WAIT_S + 1.0)
check("B  robot runs but cannot follow: ramp carries on after the wait", a > 0.3,
      f"progress {a:.2f} after {RETURN_LAG_WAIT_S + 1.0:.1f} s (waited {RETURN_LAG_WAIT_S} s)")

d = driver(None)
d._held_command = lambda: d._command.copy()  # executes every command at once
a = run(d, 1.0)
check("C  robot follows: ramp at full speed", abs(a - 0.5) < 0.02, f"progress {a:.2f} after 1.0 s of a 2 s ramp")

raise SystemExit(1 if fails else 0)
