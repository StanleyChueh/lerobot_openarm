#!/usr/bin/env python
"""Replay a single episode from a HuggingFace-hosted Isaac Sim OpenArm dataset
(LeRobotDataset format, e.g. ethanCSL/openarm_visuomotor_augmented_dataset_1000_v2)
onto the real dual-arm OpenArm follower.

Unlike replay_sim_dataset.py (which reads IsaacLab's raw HDF5 export directly via
--dump_joint_order), this reads a dataset already pushed to the HF Hub, via the
same `lerobot` LeRobotDataset class replay.py uses. Two format differences from
the raw-HDF5 path this script accounts for:

  1. The recorded "action" column is the EE-space IK delta command fed into Isaac
     Sim's controller that step -- NOT a joint target. The actual joint-space
     trajectory is "observation.state" (left_joint_1..7, radians), which is what
     gets replayed here.
  2. This dataset only records the left arm (no right_joint_* fields at all, and
     no gripper joint angle -- only a +-1 gripper *command* in action.left_gripper).
     The right arm is held at whatever pose it's actually in when this script
     connects (matching the real behavior already confirmed: the right arm is
     idle for this task in both the sim and real datasets inspected). The
     gripper command is mapped via gripper_cmd_to_raw(), not gripper_sim_to_raw()
     (which assumes a 0-0.044 rad joint angle that doesn't exist in this dataset).

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

from reset_to_rest_pose import reset_to_rest_pose
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    ARM_JOINT_KEYS,
    StdinKillSwitch,
    clamp_step,
    get_current_pos_action,
    gripper_cmd_to_raw,
    load_calibration,
    ramp_to,
    run_startup_handshake,
)

LEFT_STATE_NAMES = [f"left_joint_{i}" for i in range(1, 8)]
POS_KEYS = [f"LJ{i}.pos" for i in range(1, 9)] + [f"RJ{i}.pos" for i in range(1, 9)]


def load_trajectory(repo_id: str, episode: int) -> list[dict]:
    """Returns a list of {"joints": [7 floats, rad], "gripper_cmd": float} per frame."""
    # revision="main": this dataset has no version tag on the Hub (unlike e.g.
    # 0422_stanley_red_cube), and LeRobotDataset's default version-tag resolution
    # (get_safe_version) throws on untagged repos due to a huggingface_hub version
    # mismatch in this environment. Pinning to "main" skips that resolution entirely.
    dataset = LeRobotDataset(repo_id, episodes=[episode], revision="main", force_cache_sync=True)
    states = dataset.select_columns(OBS_STATE)
    actions = dataset.select_columns(ACTION)

    state_names = dataset.features[OBS_STATE]["names"]
    action_names = dataset.features[ACTION]["names"]
    if state_names != LEFT_STATE_NAMES:
        raise ValueError(
            f"Expected observation.state names {LEFT_STATE_NAMES}, got {state_names} --"
            " this dataset's schema doesn't match what this script assumes."
        )
    if "left_gripper" not in action_names:
        raise ValueError(f"Expected 'left_gripper' in action names, got {action_names}")
    gripper_idx = action_names.index("left_gripper")

    trajectory = []
    for idx in range(dataset.num_frames):
        joints = states[idx][OBS_STATE].tolist()
        gripper_cmd = actions[idx][ACTION][gripper_idx].item()
        trajectory.append({"joints": joints, "gripper_cmd": gripper_cmd})
    return trajectory


def frame_to_motor_action(frame: dict, calib: dict, hold_right: dict) -> dict:
    """Map one frame (left_joint_1..7 rad + left_gripper +-1 cmd) to a full 16-key
    {"LJ1.pos": rad, ...} motor action. Right arm held constant at `hold_right`."""
    sec = calib["left"]
    action = {}
    for n, jkey in enumerate(ARM_JOINT_KEYS, start=1):
        sign = sec["sign"][jkey]
        offset = sec["offset_rad"][jkey]
        action[f"LJ{n}.pos"] = sign * frame["joints"][n - 1] + offset
    grip = sec["gripper"]
    action["LJ8.pos"] = gripper_cmd_to_raw(frame["gripper_cmd"], grip["open_raw"], grip["closed_raw"])
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
    parser.add_argument("--right-port", type=str, default="can2")
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument("--model-path", type=str, required=True, help="Path to openarm_description.urdf for gravity comp")
    parser.add_argument("--max-joint-speed", type=float, default=0.3, help="rad/s cap applied to every arm joint's per-tick motion.")
    parser.add_argument("--gripper-max-speed", type=float, default=8.0, help="rad/s cap for the gripper channel specifically -- much higher than the arm cap, since gripper commands are a near-instant open/closed toggle, not a smooth trajectory")
    parser.add_argument("--handshake-tolerance", type=float, default=0.1, help="rad; abort startup if any arm joint differs from the first frame by more than this")
    parser.add_argument("--gripper-handshake-tolerance", type=float, default=1.3, help="rad; separate, more generous tolerance for the gripper channel -- open/closed state legitimately varies between episodes")
    parser.add_argument("--ramp-duration", type=float, default=2.0, help="seconds to smoothly move from real current pose to the first frame")
    parser.add_argument("--playback-hz", type=float, default=30.0, help="rate to step through the recorded trajectory (this dataset's fps is 30)")
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
        # This dataset has no right-arm data at all -- hold it at whatever pose it's
        # actually in right now for the whole episode, rather than inventing values.
        hold_right = {k: v for k, v in current_action.items() if k.startswith("R")}

        target_action = frame_to_motor_action(trajectory[0], calib, hold_right)

        if not run_startup_handshake(robot, target_action, args.handshake_tolerance, args.gripper_handshake_tolerance):
            retry = input(
                "\nThis is often just the real arm having drifted since a previous session."
                " Try moving it to match the episode's first frame, then re-check? [y/N]: "
            )
            if retry.strip().lower() != "y":
                return
            if not reset_to_rest_pose(robot, calib, target_action=target_action):
                return
            if not run_startup_handshake(robot, target_action, args.handshake_tolerance, args.gripper_handshake_tolerance):
                print("Still mismatched against the first frame after resetting. Aborting.")
                return

        confirm = input(f"Type YES to ramp to the first frame and replay {len(trajectory)} steps: ")
        if confirm.strip() != "YES":
            print("Not confirmed. Aborting without moving the arm.")
            return

        # Started only now, not before the confirmation prompt above -- this thread
        # continuously reads stdin in the background, and starting it earlier races
        # with input() for whoever typed "YES", occasionally swallowing it and hanging
        # the main thread forever with no error (see conversation history 2026-07-02).
        kill_switch = StdinKillSwitch()
        print("Type 'q' + Enter at any time to stop playback and safely ramp down.")

        print(f"Ramping to first frame over {args.ramp_duration}s...", flush=True)
        current_action = ramp_to(robot, current_action, target_action, args.ramp_duration)
        print("Ramp complete. Replaying. Type 'q' + Enter to stop.")

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
            current_action = clamp_step(current_action, desired, max_delta, gripper_max_delta)
            robot.send_action(current_action)

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
