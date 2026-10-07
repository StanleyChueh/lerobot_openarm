"""Chain step 3: the official lerobot-rollout with the trained SmolVLA checkpoint on the mocked robot.

    python chain_rollout.py base|episodic <checkpoint> <eval dataset root or -> <this directory>     (run_chain.sh does it)
"""
import json, shutil, sys, time
mode, policy, root, here = sys.argv[1:5]
sys.path.insert(0, here)
from mock_hw import CAMERAS_ARG, FAKE, KEYS
import numpy as np
import lerobot.scripts.lerobot_rollout as ro
import lerobot.rollout.strategies.core as core
import lerobot.rollout.strategies.episodic as epi
from lerobot_robot_openarm_umeow.openarm_umeow import _slow_down_rollout_returns

EVENTS = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
epi.init_keyboard_listener = lambda: (None, EVENTS)  # do not listen to the real desktop keyboard
_slow_down_rollout_returns(0.3)
RETURNS = []
_ret = core.RolloutStrategy.return_to_initial_position
def timed(hw, duration_s=3.0, fps=50):
    t0 = time.perf_counter(); n0 = len(FAKE["sends"]); ok = _ret(hw, duration_s=duration_s, fps=fps)
    RETURNS.append((t0, time.perf_counter(), n0, len(FAKE["sends"]))); return ok
core.RolloutStrategy.return_to_initial_position = staticmethod(timed)

argv = ["lerobot-rollout", f"--policy.path={policy}", "--robot.type=openarm_umeow", "--robot.assume_yes=true",
        f"--robot.cameras={CAMERAS_ARG}", "--task=reach out and back",
        '--rename_map={"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}', "--play_sounds=false", "--display_data=false"]
if mode == "base":
    argv += ["--strategy.type=base", "--duration=8"]
else:
    shutil.rmtree(root, ignore_errors=True)
    argv += ["--strategy.type=episodic", "--dataset.repo_id=local/rollout_chain", f"--dataset.root={root}",
             "--dataset.push_to_hub=false", "--dataset.num_episodes=2", "--dataset.episode_time_s=4",
             "--dataset.reset_time_s=2", "--dataset.single_task=reach out and back"]
sys.argv = argv
t_start = time.perf_counter()
ro.main()

sends = FAKE["sends"]
arm = [k for k in KEYS if not k.endswith("8.pos")]
def peak(i0, i1):
    seg = sends[i0:i1]
    v = [max(abs(b[1][k] - a[1][k]) for k in arm) / (b[0] - a[0]) for a, b in zip(seg, seg[1:]) if b[0] > a[0]]
    return max(v) if v else 0.0
ret_idx = set()
for _, _, a, b in RETURNS: ret_idx.update(range(a, b))
policy_sends = [s for i, s in enumerate(sends) if i not in ret_idx]
moved = max(np.ptp([s[1][k] for s in policy_sends]) for k in arm) if policy_sends else 0.0
print(f"ROLLOUT {mode}: {len(sends)} commands reached the robot ({len(policy_sends)} from the policy), policy moved the arm up to {moved:.3f} rad")
for t0, t1, a, b in RETURNS:
    print(f"ROLLOUT {mode}: return-to-start took {t1 - t0:.1f} s, peak joint speed {peak(a, b):.2f} rad/s")
if mode != "base":
    info = json.load(open(f"{root}/meta/info.json"))
    print(f"ROLLOUT {mode}: eval dataset {info['total_episodes']} episodes, {info['total_frames']} frames")
