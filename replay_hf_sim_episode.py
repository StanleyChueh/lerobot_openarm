#!/usr/bin/env python
"""Replay a single episode from a HuggingFace-hosted Isaac Sim OpenArm dataset
(LeRobotDataset format, e.g. ethanCSL/openarm_visuomotor_augmented_dataset_1000_v2)
onto the real dual-arm OpenArm follower.

Unlike replay_sim_dataset.py (which reads IsaacLab's raw HDF5 export directly via
--dump_joint_order), this reads a dataset already pushed to the HF Hub, via the
same `lerobot` LeRobotDataset class replay.py uses.

Three dataset schemas are supported, auto-detected from observation.state's names (see
LEFT_STATE_NAMES_EEF_ACTION / LEFT_STATE_NAMES_JOINT_ACTION / DUAL_STATE_NAMES below) -- always
replays "observation.state" (the actual recorded joint-space trajectory), never "action" directly,
since only observation.state is guaranteed to be real joint positions in every schema:

  1. EEF-action datasets (e.g. openarm_visuomotor_no_domain_randomization_1000): the recorded
     "action" column is the EE-space IK delta command fed into Isaac Sim's controller that step
     -- NOT a joint target. observation.state is the left arm's 7 joints only (left_joint_1..7,
     radians); the gripper is a +-1 *command* in action.left_gripper, not a joint angle, mapped
     via gripper_cmd_to_raw().
  2. Joint-action datasets (e.g. openarm_visuomotor_no_domain_randomization_1000_joints, see
     IsaacLab's convert_hdf5_to_lerobot.py): observation.state is LJ1.pos..LJ8.pos -- the left
     arm's 7 joints AND the gripper as an actual 0-0.044 rad joint angle, both already in the
     representation replayed here. The gripper is mapped via gripper_sim_to_raw() instead.

  3. Dual-arm joint-action datasets (convert_hdf5_to_lerobot.py --arms both/auto): the same
     representation for BOTH arms -- LJ1..LJ8 then RJ1..RJ8, 16 values. Required for a bimanual
     demo such as a right-to-left hand-over; schemas 1 and 2 carry no right-arm data at all and
     cannot describe one.

Schemas 1 and 2 do not record the right arm, so under those it is held at whatever pose it is
actually in when this script connects, rather than being driven from invented values. Schema 3
drives both arms from the recording. Each arm is mapped through its OWN calibration section --
the arms are mirrored, so reusing the left section for the right drives several joints backwards.

REQUIRED BEFORE RUNNING: the same Phase 0 calibration as mirror_bridge.py /
replay_sim_dataset.py -- a real calibration.json (see calibration.example.json).

Like those scripts: this performs a startup handshake against the very first
frame (abort + typed confirmation required if the real arm's current pose
doesn't already closely match it), ramps smoothly into position, clamps every
joint's per-step motion to a conservative speed cap, and supports a 'q' + Enter
kill switch that ramps down and disables rather than just stopping mid-motion.

Usage:
  python replay_hf_sim_episode.py \\
      --repo-id ethanCSL/openarm_visuomotor_augmented_dataset_1000_v2 --episode 0 \\
      --calibration calibration.json --model-path model/openarm_description.urdf \\
      --max-steps 30
"""

import argparse
import csv
import time

import matplotlib
matplotlib.use("Agg")  # headless -- this process only ever saves a PNG, never shows a window
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    approach_pose,
    ARM_JOINT_KEYS,
    StdinKillSwitch,
    clamp_step,
    compute_target_velocity,
    get_current_pos_action,
    gripper_cmd_to_raw,
    gripper_sim_to_raw,
    load_calibration,
    ramp_to,
)

# Two dataset schemas this script knows how to replay (detected from observation.state's names):
#   "eef_action"   -- older EE-delta-action datasets (e.g. openarm_visuomotor_no_domain_
#                      randomization_1000). observation.state is the left arm's 7 joints only
#                      (left_joint_1..7); the gripper is a +-1 *command* living in the action
#                      column (action.left_gripper), not a joint angle -- mapped via
#                      gripper_cmd_to_raw().
#   "joint_action" -- newer joint-space-action datasets (e.g. openarm_visuomotor_no_domain_
#                      randomization_1000_joints, see IsaacLab's convert_hdf5_to_lerobot.py).
#                      observation.state is LJ1.pos..LJ8.pos -- 7 arm joints AND the gripper as
#                      an actual 0-0.044 rad joint angle, both already in the exact target
#                      representation replayed here. Mapped via gripper_sim_to_raw(), the
#                      function this file's own comments already anticipated needing once a
#                      dataset like this existed.
#   "dual_joint_action" -- the same joint-space representation for BOTH arms: LJ1..LJ8 followed
#                      by RJ1..RJ8 (16 values). Produced by convert_hdf5_to_lerobot.py --arms
#                      both/auto, which is what a bimanual recording (e.g. a right-to-left
#                      hand-over) needs -- the 8-wide schemas above physically cannot describe
#                      one, since they carry no right-arm data at all.
LEFT_STATE_NAMES_EEF_ACTION = [f"left_joint_{i}" for i in range(1, 8)]
LEFT_STATE_NAMES_JOINT_ACTION = [f"LJ{i}.pos" for i in range(1, 8)] + ["LJ8.pos"]
DUAL_STATE_NAMES = LEFT_STATE_NAMES_JOINT_ACTION + [f"RJ{i}.pos" for i in range(1, 9)]
POS_KEYS = [f"LJ{i}.pos" for i in range(1, 9)] + [f"RJ{i}.pos" for i in range(1, 9)]


def load_trajectory(repo_id: str, episode: int) -> list[dict]:
    """Returns a list of {"joints": [7 floats, rad], "gripper_mode": "cmd"|"pos", "gripper_value":
    float} per frame -- gripper_mode tells frame_to_motor_action() which raw-mapping function to
    use (see the schema comment above)."""
    # revision="main": this dataset has no version tag on the Hub (unlike e.g.
    # 0422_stanley_red_cube), and LeRobotDataset's default version-tag resolution
    # (get_safe_version) throws on untagged repos due to a huggingface_hub version
    # mismatch in this environment. Pinning to "main" skips that resolution entirely.
    dataset = LeRobotDataset(repo_id, episodes=[episode], revision="main", force_cache_sync=True)
    states = dataset.select_columns(OBS_STATE)
    actions = dataset.select_columns(ACTION)

    state_names = dataset.features[OBS_STATE]["names"]
    action_names = dataset.features[ACTION]["names"]

    if state_names == LEFT_STATE_NAMES_EEF_ACTION:
        schema = "eef_action"
    elif state_names == LEFT_STATE_NAMES_JOINT_ACTION:
        schema = "joint_action"
    elif state_names == DUAL_STATE_NAMES:
        schema = "dual_joint_action"
    else:
        raise ValueError(
            f"observation.state names {state_names} match no known schema -- expected "
            f"{LEFT_STATE_NAMES_EEF_ACTION} (EE-delta-action dataset), "
            f"{LEFT_STATE_NAMES_JOINT_ACTION} (single-arm joint-space dataset), or "
            f"{DUAL_STATE_NAMES} (dual-arm joint-space dataset)."
        )
    print(f"Dataset schema: {schema} ({len(state_names)}D observation.state)")

    if schema == "eef_action":
        if "left_gripper" not in action_names:
            raise ValueError(f"Expected 'left_gripper' in action names, got {action_names}")
        gripper_idx = action_names.index("left_gripper")

    trajectory = []
    for idx in range(dataset.num_frames):
        state_vals = states[idx][OBS_STATE].tolist()
        if schema == "eef_action":
            gripper_cmd = actions[idx][ACTION][gripper_idx].item()
            trajectory.append({"joints": state_vals, "gripper_mode": "cmd", "gripper_value": gripper_cmd})
        else:
            frame = {
                "joints": state_vals[:7],
                "gripper_mode": "pos",
                "gripper_value": state_vals[7],
            }
            if schema == "dual_joint_action":
                # Same representation as the left arm, just the second half of the vector.
                frame["right_joints"] = state_vals[8:15]
                frame["right_gripper_value"] = state_vals[15]
            trajectory.append(frame)
    return trajectory


def _map_arm(action: dict, prefix: str, sec: dict, joints, gripper_value: float, gripper_mode: str):
    """Write one arm's 8 motor keys ({prefix}1..8.pos) into *action*, through its calibration.

    Each side has its OWN sign/offset/gripper-raw calibration -- the two arms are mirrored, so
    applying the left section to the right would drive several joints backwards.
    """
    for n, jkey in enumerate(ARM_JOINT_KEYS, start=1):
        action[f"{prefix}{n}.pos"] = sec["sign"][jkey] * joints[n - 1] + sec["offset_rad"][jkey]
    grip = sec["gripper"]
    to_raw = gripper_cmd_to_raw if gripper_mode == "cmd" else gripper_sim_to_raw
    action[f"{prefix}8.pos"] = to_raw(gripper_value, grip["open_raw"], grip["closed_raw"])


def frame_to_motor_action(frame: dict, calib: dict, hold_right: dict) -> dict:
    """Map one recorded frame to a full 16-key {"LJ1.pos": rad, ...} motor action.

    The left arm always comes from the data. The right arm comes from the data too when the
    dataset carries it (the dual-arm schema -- see DUAL_STATE_NAMES); otherwise it is pinned to
    `hold_right`, the pose it was already in, because the single-arm schemas have nothing to say
    about it and inventing values would move a real arm on no evidence.
    """
    _map_arm(action := {}, "LJ", calib["left"], frame["joints"], frame["gripper_value"], frame["gripper_mode"])
    if "right_joints" in frame:
        _map_arm(action, "RJ", calib["right"], frame["right_joints"],
                 frame["right_gripper_value"], frame["gripper_mode"])
    else:
        action.update(hold_right)
    return action


def save_tracking_plot(plot_history: dict, out_path: str) -> None:
    """Save a per-joint target-vs-actual time series plot. Uses a minimum y-axis
    half-range per joint category (same rationale as record_demos_openarm.py's
    mirroring plot) so small noise or a small steady-state gap can't visually read
    as a large sim-vs-real difference -- only a genuinely large gap will look large."""
    ARM_AXIS_TOLERANCE = 0.1
    GRIPPER_AXIS_TOLERANCE = 0.005

    ncols = 4
    nrows = (len(POS_KEYS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, name in enumerate(POS_KEYS):
        ax = axes[idx // ncols][idx % ncols]
        h = plot_history[name]
        ax.plot(h["t"], h["target"], label="target", linewidth=1)
        ax.plot(h["t"], h["actual"], label="actual", linewidth=1, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)

        all_v = h["target"] + h["actual"]
        if all_v:
            tolerance = GRIPPER_AXIS_TOLERANCE if name.endswith("8.pos") else ARM_AXIS_TOLERANCE
            data_min, data_max = min(all_v), max(all_v)
            center = (data_min + data_max) / 2
            half_range = max((data_max - data_min) / 2 * 1.1, tolerance)
            ax.set_ylim(center - half_range, center + half_range)

    for idx in range(len(POS_KEYS), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Dataset target vs. real robot actual joint positions (replay session)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved target-vs-actual comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id")
    parser.add_argument("--episode", type=int, default=0, help="Episode index inside the dataset")
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration.json")
    parser.add_argument("--right-port", type=str, default="can0")
    parser.add_argument("--left-port", type=str, default="can1")
    parser.add_argument("--model-path", type=str, required=True, help="Path to openarm_description.urdf for gravity comp")
    parser.add_argument("--max-joint-speed", type=float, default=1.2, help="rad/s cap applied to every arm joint's per-tick motion.")
    parser.add_argument("--gripper-max-speed", type=float, default=2.0, help="rad/s cap for the gripper channel specifically -- much higher than the arm cap, since gripper commands are a near-instant open/closed toggle, not a smooth trajectory")
    parser.add_argument("--handshake-tolerance", type=float, default=0.05, help="rad; if every joint is already within this of the first frame the startup approach is skipped as a no-op. NOT an abort threshold any more -- exceeding it just means the arm ramps there (see --max-approach-delta for the gate that does refuse).")
    parser.add_argument("--max-approach-delta", type=float, default=1.8, help="rad; REFUSE to start if any arm joint would have to travel further than this to reach the start pose. This is the real safety gate: a gap that large means the calibration or zeroing is wrong, not that the arm drifted.")
    parser.add_argument("--approach-speed", type=float, default=0.3, help="rad/s ceiling for the startup approach. The ramp duration is derived from this and the furthest-travelling joint, so no joint exceeds it.")
    parser.add_argument("--yes", action="store_true", help="Skip the typed YES confirmation before the startup approach moves the arm. For unattended runs only -- --max-approach-delta still applies.")
    parser.add_argument("--ramp-duration", type=float, default=2.0, help="seconds to smoothly move from real current pose to the first frame")
    parser.add_argument("--playback-hz", type=float, default=20.0, help="rate to step through the recorded trajectory (this dataset's fps is 30)")
    parser.add_argument("--max-steps", type=int, default=None, help="only replay the first N steps of the episode -- use for a cautious first test")
    parser.add_argument(
        "--log-csv", type=str, default=None,
        help="If set, read back the arm's ACTUAL position every step (in addition to sending the"
        " commanded one) and log target/commanded/actual for all 16 channels to this CSV path."
        " Diagnostic only -- the extra read adds latency and will slow playback below --playback-hz.",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="If set, save a per-joint target-vs-actual comparison plot (PNG) to this path at the end"
        " of playback. Implies the same extra per-step read as --log-csv (can be used with or without it).",
    )
    args = parser.parse_args()

    calib = load_calibration(args.calibration)
    print(f"Loading {args.repo_id}, episode {args.episode}...")
    trajectory = load_trajectory(args.repo_id, args.episode)
    if args.max_steps is not None:
        trajectory = trajectory[: args.max_steps]
    print(f"Loaded {len(trajectory)} steps.")

    robot_cfg = OpenArmFollowerConfig(
        right_port=args.right_port,
        left_port=args.left_port,
        enable_fd=True,
        model_path=args.model_path,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    log_file = None
    log_writer = None
    if args.log_csv:
        log_file = open(args.log_csv, "w", newline="")
        fieldnames = ["step", "t"]
        for k in POS_KEYS:
            fieldnames += [f"{k}_target", f"{k}_commanded", f"{k}_actual"]
        log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        log_writer.writeheader()
        print(f"Logging target/commanded/actual positions to {args.log_csv}")

    collect_diagnostics = bool(args.log_csv or args.plot)
    plot_history = {k: {"t": [], "target": [], "actual": []} for k in POS_KEYS} if args.plot else None

    try:
        current_action = get_current_pos_action(robot)
        # Only used for the single-arm schemas, which carry no right-arm data: hold it at whatever
        # pose it is actually in right now for the whole episode, rather than inventing values.
        # A dual-arm dataset drives it from the recording instead (see frame_to_motor_action).
        hold_right = {k: v for k, v in current_action.items() if k.startswith("R")}
        dual_arm = "right_joints" in trajectory[0]
        print(
            "Right arm: DRIVEN from the dataset (dual-arm replay)" if dual_arm
            else "Right arm: HELD at its current pose (dataset has no right-arm data)"
        )

        target_action = frame_to_motor_action(trajectory[0], calib, hold_right)

        # Go to the episode's first frame along a speed-limited ramp instead of demanding the arm
        # already be there. See approach_pose() in sim_bridge_common.py for why the old
        # abort-and-reposition-by-hand gate could not be satisfied: the residual it measured is
        # mostly steady-state droop the arm reproduces every time it holds a pose. The part of it
        # that was load-bearing -- refusing a move so large it implies bad calibration -- survives
        # as --max-approach-delta.
        approached = approach_pose(
            robot, target_action,
            label="the episode's first frame",
            arm_speed=args.approach_speed,
            max_delta=args.max_approach_delta,
            settled_tolerance=args.handshake_tolerance,
            min_duration=args.ramp_duration,
            assume_yes=args.yes,
        )
        if approached is None:
            return
        current_action = approached

        # Started only after the confirmation prompt inside approach_pose(), not before -- this
        # thread continuously reads stdin in the background, and starting it earlier races with
        # input() for whoever typed "YES", occasionally swallowing it and hanging the main thread
        # forever with no error.
        kill_switch = StdinKillSwitch()
        print("Replaying. Type 'q' + Enter at any time to stop playback and safely ramp down.")

        dt = 1.0 / args.playback_hz
        max_delta = args.max_joint_speed * dt
        gripper_max_delta = args.gripper_max_speed * dt
        halted = False
        t0 = time.time()
        tracking_errors = {k: [] for k in POS_KEYS}

        for step_idx, frame in enumerate(trajectory):
            loop_start = time.time()

            if kill_switch.triggered:
                print(f"\nKill switch pressed at step {step_idx}/{len(trajectory)}. Ramping down and disabling.")
                halted = True
                break

            desired = frame_to_motor_action(frame, calib, hold_right)
            target_action = clamp_step(current_action, desired, max_delta, gripper_max_delta)
            # send_action() takes the feedforward velocity as a REQUIRED second argument -- it
            # feeds each joint's MIT-control dq term. Derived from the step the clamp actually
            # allowed (current -> target over dt), not from the raw dataset delta, so a step the
            # speed cap shortened is not accompanied by a velocity asking for the full jump.
            # Same construction mirror_bridge.py uses.
            target_vel = compute_target_velocity(current_action, target_action, dt, args.max_joint_speed)
            robot.send_action(target_action, target_vel)
            current_action = target_action

            if collect_diagnostics:
                try:
                    actual = get_current_pos_action(robot)
                except RuntimeError as e:
                    actual = {}
                    print(f"  [diagnostics] state read failed at step {step_idx}, logging blanks: {e}")
                now_t = time.time() - t0
                if log_writer is not None:
                    row = {"step": step_idx, "t": now_t}
                    for k in POS_KEYS:
                        row[f"{k}_target"] = desired.get(k)
                        row[f"{k}_commanded"] = current_action.get(k)
                        row[f"{k}_actual"] = actual.get(k)
                    log_writer.writerow(row)
                for k in POS_KEYS:
                    if k in actual and k in desired:
                        tracking_errors[k].append(abs(actual[k] - desired[k]))
                        if plot_history is not None:
                            plot_history[k]["t"].append(now_t)
                            plot_history[k]["target"].append(desired[k])
                            plot_history[k]["actual"].append(actual[k])

            if step_idx % 1 == 0:
                print(f"  step {step_idx}/{len(trajectory)}")

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        if not halted:
            print(f"\nReplay of all {len(trajectory)} steps complete.")

        if log_writer is not None:
            print("\nTracking summary (|actual - target| per channel, i.e. how far the real arm"
                  " ended up from where the recorded episode wanted it, each step):")
            for k in POS_KEYS:
                errs = tracking_errors[k]
                if errs:
                    print(f"  {k:10s} mean={sum(errs) / len(errs):.4f} rad   max={max(errs):.4f} rad")

        print("Ramping down to a safe hold before disabling...")
        safe_hold = get_current_pos_action(robot)
        ramp_to(robot, current_action, safe_hold, duration_s=1.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors.")
    finally:
        if log_file is not None:
            log_file.close()
            print(f"Wrote log to {args.log_csv}")
        if plot_history is not None and any(plot_history[k]["t"] for k in POS_KEYS):
            save_tracking_plot(plot_history, args.plot)
        try:
            robot.disconnect()
        except Exception:
            print("WARNING: error during disconnect -- verify motors are physically de-energized.")
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
