"""Matched evaluation in simulation: every policy on the SAME can poses, for the SAME number of simulated steps.

    python -m lerobot_robot_openarm_umeow.robot_client <client flags> --robot.type=openarm_isaac \
        --num_episodes=100 --episode_steps=330 --poses=<poses.json> --results_dir=<dir> [--resume=true]

robot_client.py's time-based episodes (--episode_time_s) end on the WALL clock, so the simulated length of an
episode depends on how fast the machine renders (270-321 steps for 15 s), and the scene's random can pose is
neither chosen nor logged. With --episode_steps and --poses (robot_client.py, openarm_isaac only):

  pose     trial i places the can at poses[i] (env-local x, y; upright, at the task's resting height) with ONE
           scene reset right before the episode, and checks where the simulator actually put it. A pose that
           cannot be applied stops the run; it is never replaced. The return to the start pose between
           episodes does not reset the scene in this mode (openarm_isaac.auto_scene_reset), so nothing can
           re-randomize the can after it was placed. The robot keeps its start pose through the reset.
  steps    the policy drives for exactly --episode_steps simulation steps (one per executed action, lockstep),
           however long that takes on the wall clock. A trial that stops early -- a key, a stalled policy or
           simulator (no step for --stall_timeout_s), an error -- is INCOMPLETE: recorded as such, never as a
           success or a failure. An error stops the run; --resume=true continues it.
  metrics  computed from the simulator's own ground truth (lerobot_sim_server.py's "handover" reply), over
           the policy's steps only -- taken BEFORE the arm returns home, so the return can never make a trial
           successful. See trial_metrics().
  results  <results_dir>/run_meta.json (versions, flags, pose file hash), trials.jsonl (one line per trial),
           steps/trial_XXX.json (every step's ground truth). An existing results_dir needs --resume=true,
           which runs only the trials not yet complete.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_STEPS = (240, 270, 300, 330)
POSE_TOL_XY_M = 0.002  # where the simulator put the can vs the request, after the reset's settle steps
POSE_TOL_Z_M = 0.004  # vs the task's resting height
POSE_TOL_TILT = 0.02  # |qx|, |qy| of the can's quaternion (~2.3 deg): standing upright


class PoseNotApplied(RuntimeError):
    """The simulator did not put the can where it was asked to."""


# -- poses ---------------------------------------------------------------------------------------------------
def load_poses(path: str | os.PathLike) -> tuple[list[dict], str]:
    """(poses in file order, sha256 of the file). Each pose: {"trial", "x", "y", ...}; trial i is poses[i]."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    poses = data["poses"] if isinstance(data, dict) else data
    for i, p in enumerate(poses):
        if int(p.get("trial", i)) != i:
            raise SystemExit(f"[openarm] {path}: pose {i} is labelled trial {p.get('trial')}; poses must be in trial order.")
        for k in ("x", "y"):
            if not isinstance(p.get(k), (int, float)):
                raise SystemExit(f"[openarm] {path}: pose {i} has no numeric {k!r}.")
    return poses, hashlib.sha256(raw).hexdigest()


def check_reset(requested: tuple[float, float], truth: dict | None) -> dict:
    """Where the can is after a reset that asked for `requested`. Raises PoseNotApplied if it is not there."""
    if truth is None:
        raise PoseNotApplied("the simulator does not report the can (lerobot_sim_server.py without the 'handover'"
                             " reply, or not --task_mode handover): cannot verify the pose. Update the server.")
    x, y, z, qw, qx, qy, qz = truth["can_pose"]
    err_xy = ((x - requested[0]) ** 2 + (y - requested[1]) ** 2) ** 0.5
    err_z = z - truth["can_rest_z"]
    out = {"requested_xy": [float(requested[0]), float(requested[1])], "actual_pose": truth["can_pose"],
           "error_xy_m": err_xy, "error_z_m": err_z, "tilt": max(abs(qx), abs(qy)), "stage": truth["stage"]}
    problems = []
    if err_xy > POSE_TOL_XY_M:
        problems.append(f"{1000 * err_xy:.1f} mm from the requested x, y")
    if abs(err_z) > POSE_TOL_Z_M:
        problems.append(f"{1000 * err_z:+.1f} mm off its resting height")
    if out["tilt"] > POSE_TOL_TILT:
        problems.append(f"tilted (|qx|, |qy| up to {out['tilt']:.3f})")
    if truth["stage"] != 0:
        problems.append(f"hand-over stage {truth['stage']} right after the reset")
    if problems:
        raise PoseNotApplied(f"can requested at ({requested[0]:.4f}, {requested[1]:.4f}) is at"
                             f" ({x:.4f}, {y:.4f}, {z:.4f}): " + ", ".join(problems))
    return out


# -- metrics ---------------------------------------------------------------------------------------------------
def _first(flags, start: int = 0) -> int | None:
    """1-based step of the first True at or after 0-based index `start`."""
    for i in range(start, len(flags)):
        if flags[i]:
            return i + 1
    return None


def holding(step: dict) -> bool:
    """The left hand holds the can and the right hand does not, and the can is off the pad: the task's own
    tests (receiver_grip_confirmed, object_grasped_by) and its put-down height margin."""
    h = step["handover"]
    return bool(h["left_grip_confirmed"] and not h["right_holds"]
                and h["can_pose"][2] > h["can_rest_z"] + h["put_down_margin"])


def trial_metrics(steps: list[dict], budget: int, checkpoints=CHECKPOINT_STEPS) -> dict:
    """The three success metrics and the stage timings of one trial, from its per-step ground truth.

    `steps`: one dict per policy-controlled simulation step, in order: {"success", "handover": {...}}.
    Step numbers are 1-based (step k = the state after the policy's k-th action).

      success_any     METRIC 1, the original: the task's handover_success was true at any policy step.
      stable          METRIC 2: a pass happened (stage >= 2) and then, for >= hold_steps_required consecutive
                      steps, the left hand held the can (aperture band + on the can), the right hand did not,
                      and the can stayed above its resting height + the put-down margin. Recomputed here from
                      the per-step signals -- independently of the task's own counter, which it must match
                      (`stable_matches_stage`).
      holding_end     METRIC 3: holding() at the last step (the budget); also holding_at_<k> for each checkpoint.
    """
    n = len(steps)
    if not n:
        return {"steps": 0}
    hs = [s["handover"] for s in steps]
    need = hs[0]["hold_steps_required"]
    stage = [h["stage"] for h in hs]
    success = [bool(s["success"]) for s in steps]
    right = [h["right_holds"] for h in hs]
    left = [h["left_holds"] for h in hs]

    run, stable_step, longest = 0, None, 0
    for i, h in enumerate(hs):
        carrying = stage[i] >= 2 and h["left_grip_confirmed"] and not h["right_holds"] \
            and h["can_pose"][2] > h["can_rest_z"] + h["put_down_margin"]
        run = run + 1 if carrying else 0
        longest = max(longest, run)
        if stable_step is None and run >= need:
            stable_step = i + 1
    pass_step = _first([s >= 2 for s in stage])
    stage3_step = _first([s >= 3 for s in stage])
    hold = [holding(s) for s in steps]
    z = [h["can_pose"][2] for h in hs]
    rest = hs[0]["can_rest_z"]

    out = {
        "steps": n,
        "complete": n == budget,
        "success_any": any(success),
        "first_success_step": _first(success),
        "success_at_end": success[-1],
        "stable": stable_step is not None,
        "stable_completion_step": stable_step,
        "stage3_step": stage3_step,
        "stable_matches_stage": stable_step == stage3_step,
        "longest_carry_steps": longest,
        "holding_end": hold[-1] if n == budget else None,
        "holding_steps_total": sum(hold),
        "max_stage": max(stage),
        "right_grasp_step": _first(right),
        "left_grasp_step": pass_step,  # the pass: the left hand took the can while the right had it (stage 2)
        "left_touch_step": _first(left),  # the left grasp test alone, pass or not
        "right_release_step": _first([not r for r in right], pass_step - 1) if pass_step else None,
        "was_lifted": hs[-1]["was_lifted"] or any(h["was_lifted"] for h in hs),
        "can_max_height_above_rest": max(z) - rest,
        "can_final_height_above_rest": z[-1] - rest,
        "initial_can_pose": hs[0]["can_pose"],
        "final_can_pose": hs[-1]["can_pose"],
        "can_rest_z": rest,
    }
    for k in checkpoints:
        out[f"holding_at_{k}"] = hold[k - 1] if k <= n else None
    out["outcome"] = outcome(out)
    return out


def outcome(m: dict) -> str:
    """The furthest stage the trial reached, and whether the can was still held at the end."""
    if m["stable"]:
        return "stable_held_to_end" if m["holding_end"] else "stable_then_lost"
    if m["max_stage"] >= 2:
        return "pass_not_secured"  # the left hand took it during the pass but never held it on its own
    if m["max_stage"] >= 1:
        return "right_grasp_no_pass"
    return "no_right_grasp"


def step_record(reply: dict, sent_action: list[float], state: list[float], t: float) -> dict:
    """What a trial keeps of one simulation step (the server's reply to it, minus the camera images). The action
    sent and the joints measured after the step are motor values in common.MOTOR_KEYS order."""
    return {"sim_step": reply.get("step"), "t": t, "success": bool(reply.get("success")),
            "handover": reply.get("handover"), "action": sent_action, "state": state}


# -- results folder --------------------------------------------------------------------------------------------
def _git(path: Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    if not path.exists():
        return {"path": str(path), "missing": True}
    dirty = [line for line in run("status", "--porcelain").splitlines() if line]
    untracked = sorted(line[3:] for line in dirty if line.startswith("??") and not line[3:].startswith(("research/", "outputs/")))
    h = hashlib.sha256(run("diff", "HEAD").encode())  # uncommitted edits: equal hashes = the same code
    for rel in untracked:
        f = path / rel
        for p in sorted(f.rglob("*")) if f.is_dir() else [f]:
            if p.is_file() and p.suffix == ".py":
                h.update(p.read_bytes())
    return {"path": str(path), "commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "dirty": dirty, "worktree_sha256": h.hexdigest()}


def _checkpoint_identity(path: str | None) -> dict:
    """What can be said about the checkpoint without loading it: a Hub cache snapshot's commit, its config hash."""
    if not path:
        return {}
    p = Path(path).expanduser()
    out = {"pretrained_name_or_path": path}
    if p.exists():
        resolved = p.resolve()
        out["resolved"] = str(resolved)
        if resolved.parent.name == "snapshots":  # ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>
            out["hub_repo"] = resolved.parent.parent.name.removeprefix("models--").replace("--", "/")
            out["hub_revision"] = resolved.name
        for name in ("config.json", "model.safetensors", "train_config.json"):
            f = p / name
            if f.exists() and f.stat().st_size < 4 << 30:
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for block in iter(lambda: fh.read(1 << 22), b""):
                        h.update(block)
                out[f"sha256_{name}"] = h.hexdigest()
    return out


class Results:
    """<results_dir>: run_meta.json, trials.jsonl, steps/trial_XXX.json."""

    def __init__(self, root: str | os.PathLike, resume: bool):
        self.root = Path(root).expanduser().resolve()
        self.trials_file = self.root / "trials.jsonl"
        done = self.load(self.root) if self.trials_file.exists() else []
        if self.root.exists() and any(self.root.iterdir()) and not resume:
            raise SystemExit(f"[openarm] --results_dir {self.root} already holds results; --resume=true continues"
                             " them (only the trials not yet complete are run). Never overwritten.")
        self.complete = {t["trial"] for t in done if t["status"] == "complete"}
        (self.root / "steps").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load(root) -> list[dict]:
        path = Path(root) / "trials.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []

    def write_meta(self, meta: dict) -> None:
        """One file per start (a resumed run adds run_meta_<n>.json): what produced these trials."""
        n = len(list(self.root.glob("run_meta*.json")))
        (self.root / ("run_meta.json" if n == 0 else f"run_meta_{n}.json")).write_text(json.dumps(meta, indent=1) + "\n")

    def add(self, trial: dict, steps: list[dict]) -> None:
        if steps:
            (self.root / "steps" / f"trial_{trial['trial']:03d}_{trial['attempt']}.json").write_text(json.dumps(steps))
        with open(self.trials_file, "a") as f:
            f.write(json.dumps(trial) + "\n")
        if trial["status"] == "complete":
            self.complete.add(trial["trial"])


def run_meta(argv: list[str], poses_path: str, poses_sha: str, n_poses: int, budget: int, policy_path: str | None,
             hello: dict | None) -> dict:
    ws = Path(__file__).resolve().parents[3]  # lerobot_openarm
    repos = {"lerobot_openarm": ws, "IsaacLab": ws.parent / "IsaacLab",
             "lerobot_fork": ws.parent / "lerobot_experiment" / "lerobot"}
    try:
        import lerobot

        lerobot_version = getattr(lerobot, "__version__", None)
        lerobot_file = lerobot.__file__
    except Exception:
        lerobot_version = lerobot_file = None
    return {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": argv,
        "episode_steps": budget,
        "poses_file": str(Path(poses_path).resolve()),
        "poses_sha256": poses_sha,
        "poses_total": n_poses,
        "checkpoint": _checkpoint_identity(policy_path),
        "simulator": {k: v for k, v in (hello or {}).items() if k != "joint_names"},
        "git": {name: _git(path) for name, path in repos.items()},
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "lerobot": {"version": lerobot_version, "file": lerobot_file},
        "env": {k: v for k, v in os.environ.items() if k.startswith(("OPENARM_", "CUDA_VISIBLE", "HF_HUB_OFFLINE"))},
    }


# -- the step budget ----------------------------------------------------------------------------------------------
def step_budget_client_class(base):
    """`base` that stops its control loop after exactly `step_budget` simulation steps (robot.sim_steps)."""

    class StepBudgetClient(base):
        step_budget: int | None = None
        _steps_at_start = 0

        def begin_budget(self, steps: int | None) -> None:
            self.step_budget = steps
            self._steps_at_start = self.robot.sim_steps

        def policy_steps(self) -> int:
            return self.robot.sim_steps - self._steps_at_start

        def budget_spent(self) -> bool:
            return self.step_budget is not None and self.policy_steps() >= self.step_budget

        def actions_available(self):
            if self.budget_spent():
                return False
            return super().actions_available()

        def control_loop_action(self, verbose: bool = False):
            performed = super().control_loop_action(verbose)
            if self.budget_spent():
                self.shutdown_event.set()  # the loop ends before it could execute another action
            return performed

        def _ready_to_send_observation(self):
            if self.budget_spent():
                return False
            return super()._ready_to_send_observation()

    return StepBudgetClient
