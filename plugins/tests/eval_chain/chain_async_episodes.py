"""Episodic async inference (lerobot_robot_openarm_umeow.robot_client) on the mocked robot, with scripted keys.

    python chain_async_episodes.py <smolvla|groot> <checkpoint> <actions_per_chunk> <this directory>

CHAIN_SERVER_ARGS / CHAIN_CLIENT_ARGS as in chain_async.py (e.g. --rtc=true / --aggregate_fn_name=latest_only).
3 episodes of 6 s with 2 s resets; the first attempt is discarded with Left arrow after 2 s (run again), and
episode 2 is ended with Right arrow after 3 s. Checks: every episode starts at the start pose and the policy
moves the arm in it, every reset starts at the start pose, the server starts each episode clean, timings.
"""
import os
import re
import signal
import subprocess
import sys
import threading
import time

policy_type, policy, chunk, here = sys.argv[1:5]
sys.path.insert(0, here)
from mock_hw import CAMERAS_ARG, FAKE, KEYS  # noqa: E402

PORT = 8098
server_log = os.path.join(os.environ.get("CHAIN_OUT", "/tmp"), f"policy_server_episodes_{policy_type}.log")
log = open(server_log, "w")
server = subprocess.Popen([sys.executable, "-u", "-m", "lerobot_robot_openarm_umeow.policy_server",
                           "--host=127.0.0.1", f"--port={PORT}", *os.environ.get("CHAIN_SERVER_ARGS", "").split()],
                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
time.sleep(8)

import lerobot_robot_openarm_umeow.robot_client as rce  # noqa: E402

EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
rce.init_keyboard_listener = lambda: (None, EVENTS)  # no real keyboard: scripted below
PLAN = {1: ("rerecord_episode", 2.0), 3: ("exit_early", 3.0)}  # run index -> key, seconds into it
ARM = [k for k in KEYS if not k.endswith("8.pos")]
RUNS, RESETS = [], []  # (pose at start, max move during, seconds, actions) / pose when a reset starts

_run, _reset = rce._run_episode, rce._reset_phase


def run_episode(client, task, episode_time_s, events):
    index = len(RUNS) + 1
    if index in PLAN:
        key, after = PLAN[index]
        threading.Timer(after, lambda: events.__setitem__(key, True)).start()
    start = dict(FAKE["state"])
    n0 = len(FAKE["sends"])
    result = _run(client, task, episode_time_s, events)
    moved = max((abs(s[1][k] - start[k]) for s in FAKE["sends"][n0:] for k in ARM), default=0.0)
    RUNS.append((start, moved, result[1], result[2]))
    return result


def reset_phase(reset_time_s, events):
    RESETS.append(dict(FAKE["state"]))
    return _reset(reset_time_s, events)


rce._run_episode, rce._reset_phase = run_episode, reset_phase
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
sys.argv = ["robot_client", f"--server_address=127.0.0.1:{PORT}", "--robot.type=openarm_umeow",
            "--robot.assume_yes=true", f"--robot.cameras={CAMERAS_ARG}", "--task=reach out and back",
            f"--policy_type={policy_type}", f"--pretrained_name_or_path={policy}", "--policy_device=cuda",
            f"--actions_per_chunk={chunk}", "--chunk_size_threshold=0.5", "--aggregate_fn_name=weighted_average",
            "--fps=30", *os.environ.get("CHAIN_CLIENT_ARGS", "").split(),
            "--num_episodes=3", "--episode_time_s=6", "--reset_time_s=2"]
try:
    signal.alarm(180)
    rce.main()
finally:
    signal.alarm(0)
    os.killpg(server.pid, signal.SIGTERM)
    time.sleep(1)

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def off(a, b):
    return max(abs(a[k] - b[k]) for k in ARM)


home = RUNS[0][0] if RUNS else {}
print("runs (start offset from run 1, max move, s, actions):",
      [(round(off(r[0], home), 3), round(r[1], 3), round(r[2], 1), r[3]) for r in RUNS])
check("4 runs: the discarded attempt + 3 counted episodes", len(RUNS) == 4, str(len(RUNS)))
check("every run starts at the start pose", RUNS and all(off(r[0], home) < 0.1 for r in RUNS),
      str([round(off(r[0], home), 3) for r in RUNS]))
check("policy actions reach the robot in every run", RUNS and all(r[3] > 10 for r in RUNS), str([r[3] for r in RUNS]))
# (A barely trained test policy can request a jump the robot's safety guard refuses: the arm then holds.)
check("the policy moves the arm in most runs", sum(r[1] > 0.02 for r in RUNS) >= 3,
      str([round(r[1], 3) for r in RUNS]))
check("3 resets (none after the last episode), each starting at the start pose",
      len(RESETS) == 3 and all(off(p, home) < 0.1 for p in RESETS), str([round(off(p, home), 3) for p in RESETS]))
durations = [r[2] for r in RUNS]
check("timings: Left arrow at 2 s, full 6 s, Right arrow at 3 s, full 6 s",
      len(durations) == 4 and abs(durations[0] - 2) < 0.6 and abs(durations[1] - 6) < 0.6
      and abs(durations[2] - 3) < 0.6 and abs(durations[3] - 6) < 0.6, str([round(d, 1) for d in durations]))
text = open(server_log).read()
readies = len(re.findall(r"connected and ready", text))
check("the server was told a new client is ready for every run", readies >= 4, f"{readies} Ready")
check("no server errors", "Error" not in text and "Traceback" not in text)
if "RTC chunk" in text:  # with --rtc=true: the first chunk of every run continues nothing from the previous one
    firsts = [m.group(1) for m in re.finditer(r"connected and ready.*?RTC chunk @ timestep \d+: guided by (\d+)", text, re.S)]
    check("RTC: the first chunk of every run is unguided (the previous episode's chunk is forgotten)",
          len(firsts) >= 4 and all(g == "0" for g in firsts), str(firsts))
    check("RTC: later chunks are guided", len(re.findall(r"guided by [1-9]", text)) > 0)
print(f"ASYNC EPISODES {policy_type}: {len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
