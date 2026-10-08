"""Async inference (policy server + robot client) on the mocked robot.

    python chain_async.py <smolvla|groot> <checkpoint> <actions_per_chunk> <this directory> [seconds]

Extra server / client flags come from the CHAIN_SERVER_ARGS / CHAIN_CLIENT_ARGS environment variables (e.g.
CHAIN_SERVER_ARGS="--rtc=true --rtc_execution_horizon=10" CHAIN_CLIENT_ARGS="--aggregate_fn_name=latest_only").

Starts `python -m lerobot_robot_openarm_umeow.policy_server` as a separate process, then runs lerobot's own
robot client (lerobot.async_inference.robot_client) in this process against the mocked follower, and stops
it after `seconds`. No CAN traffic.
"""
import os
import signal
import subprocess
import sys
import time

policy_type, policy, chunk, here = sys.argv[1:5]
seconds = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
sys.path.insert(0, here)
from mock_hw import CAMERAS_ARG, FAKE, KEYS  # noqa: E402

PORT = 8099
log = open(os.path.join(os.environ.get("CHAIN_OUT", "/tmp"), f"policy_server_{policy_type}.log"), "w")
server = subprocess.Popen([sys.executable, "-u", "-m", "lerobot_robot_openarm_umeow.policy_server",
                           "--host=127.0.0.1", f"--port={PORT}", *os.environ.get("CHAIN_SERVER_ARGS", "").split()],
                          stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True)
time.sleep(8)

import lerobot.async_inference.robot_client as rc  # noqa: E402


def _stop(*_):
    raise KeyboardInterrupt


signal.signal(signal.SIGALRM, _stop)
sys.argv = ["robot_client", f"--server_address=127.0.0.1:{PORT}", "--robot.type=openarm_umeow",
            "--robot.assume_yes=true", f"--robot.cameras={CAMERAS_ARG}", "--task=reach out and back",
            f"--policy_type={policy_type}", f"--pretrained_name_or_path={policy}", "--policy_device=cuda",
            f"--actions_per_chunk={chunk}", "--chunk_size_threshold=0.5", "--aggregate_fn_name=weighted_average",
            "--fps=30", *os.environ.get("CHAIN_CLIENT_ARGS", "").split()]
t0 = time.perf_counter()
n0 = len(FAKE["sends"])
try:
    signal.alarm(int(seconds) + 20)  # connect + approach + run
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()  # what `python -m lerobot.async_inference.robot_client` does first
    rc.async_client()
except KeyboardInterrupt:
    pass
finally:
    os.killpg(server.pid, signal.SIGTERM)

sends = FAKE["sends"][n0:]
arm = [k for k in KEYS if not k.endswith("8.pos")]
moved = max((max(s[1][k] for s in sends) - min(s[1][k] for s in sends)) for k in arm) if sends else 0.0
span = (sends[-1][0] - sends[0][0]) if len(sends) > 1 else 0.0
steps = [max(abs(b[1][k] - a[1][k]) for k in arm) for a, b in zip(sends, sends[1:])]
print(f"ASYNC {policy_type}: {len(sends)} commands reached the robot over {span:.1f} s"
      f" ({len(sends) / span if span else 0:.0f} Hz), policy moved the arm up to {moved:.3f} rad,"
      f" largest step between consecutive commands {max(steps) if steps else 0:.4f} rad")
