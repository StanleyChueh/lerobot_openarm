#!/usr/bin/env python
"""Replay a single recorded Isaac Sim OpenArm episode on the real dual-arm OpenArm follower.

This reads a recorded demo directly out of the HDF5 file produced by IsaacLab's
scripts/tools/record_demos_openarm.py (ActionStateRecorderManagerCfg), using plain
h5py -- it does NOT require IsaacLab to be installed in this hardware-control
environment. It reuses the exact same calibration/mapping/clamping/handshake/ramp
logic as mirror_bridge.py (see sim_bridge_common.py), so anything verified for live
mirroring applies here too.

REQUIRED BEFORE RUNNING THIS SCRIPT:
  1. Complete the same Phase 0 bench verification as mirror_bridge.py (zero-calibration
     check, per-joint sign check, gripper scale check) and fill in calibration.json.
  2. Record (or re-use) a demo with `--dump_joint_order <path.json>` passed to
     record_demos_openarm.py, so this script knows which HDF5 column is which joint.
     This does NOT require --mirror_udp_port -- it's independent and always safe/passive.

Like mirror_bridge.py, this performs a startup handshake against the very first frame
of the replay (abort + typed confirmation required if the real arm's current pose
doesn't already closely match it), ramps smoothly into position, clamps every joint's
per-step motion to a conservative speed cap, and supports a 'q' + Enter kill switch
that ramps down and disables rather than just stopping mid-motion.

Usage:
  python replay_sim_dataset.py \\
      --dataset logs/demos/pickup.hdf5 --episode demo_0 \\
      --joint-order joint_order.json --calibration calibration.json \\
      --model-path model/openarm_description.urdf
"""

import argparse
import json
import time

import h5py
import numpy as np

from reset_to_rest_pose import reset_to_rest_pose
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    StdinKillSwitch,
    clamp_step,
    get_current_pos_action,
    load_calibration,
    ramp_to,
    run_startup_handshake,
    sim_joints_to_motor_action,
)


def load_trajectory(dataset_path: str, episode: str, joint_order_path: str) -> list[dict]:
    with open(joint_order_path) as f:
        joint_order = json.load(f)["joint_order"]

    with h5py.File(dataset_path, "r") as f:
        key = f"data/{episode}/states/articulation/robot/joint_position"
        if key not in f:
            available = list(f["data"].keys()) if "data" in f else []
            raise KeyError(f"'{key}' not found in {dataset_path}. Available episodes: {available}")
        raw = np.asarray(f[key])  # shape (num_steps, 1, num_joints) or (num_steps, num_joints)

    if raw.ndim == 3:
        raw = raw[:, 0, :]
    if raw.shape[1] != len(joint_order):
        raise ValueError(
            f"joint_position has {raw.shape[1]} columns but joint_order has {len(joint_order)} entries"
            f" -- did you pass --dump_joint_order for THIS task/robot config?"
        )

    return [dict(zip(joint_order, row.tolist())) for row in raw]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="Path to the recorded .hdf5 file")
    parser.add_argument("--episode", type=str, default="demo_0", help="Episode name inside the HDF5 file")
    parser.add_argument("--joint-order", type=str, required=True, help="Path to JSON written by --dump_joint_order")
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration.json")
    parser.add_argument("--right-port", type=str, default="can2")
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument("--model-path", type=str, required=True, help="Path to openarm_description.urdf for gravity comp")
    parser.add_argument("--max-joint-speed", type=float, default=0.3, help="rad/s cap applied to every arm joint's per-tick motion.")
    parser.add_argument("--gripper-max-speed", type=float, default=8.0, help="rad/s cap for the gripper channel specifically -- much higher than the arm cap, since gripper commands are a near-instant open/closed toggle, not a smooth trajectory")
    parser.add_argument("--handshake-tolerance", type=float, default=0.1, help="rad; abort startup if any arm joint differs from the first frame by more than this")
    parser.add_argument("--gripper-handshake-tolerance", type=float, default=1.3, help="rad; separate, more generous tolerance for the gripper channel -- open/closed state legitimately varies between episodes")
    parser.add_argument("--ramp-duration", type=float, default=2.0, help="seconds to smoothly move from real current pose to the first frame")
    parser.add_argument("--playback-hz", type=float, default=20.0, help="rate to step through the recorded trajectory (match the task's control rate, default 20Hz per decimation=5 @ 100Hz sim)")
    parser.add_argument("--max-steps", type=int, default=None, help="only replay the first N steps of the episode -- use for a cautious first test")
    args = parser.parse_args()

    calib = load_calibration(args.calibration)
    trajectory = load_trajectory(args.dataset, args.episode, args.joint_order)
    if args.max_steps is not None:
        trajectory = trajectory[: args.max_steps]
    print(f"Loaded {len(trajectory)} steps from {args.dataset}::{args.episode}")

    robot_cfg = OpenArmFollowerConfig(
        right_port=args.right_port,
        left_port=args.left_port,
        enable_fd=True,
        model_path=args.model_path,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    try:
        current_action = get_current_pos_action(robot)
        target_action = sim_joints_to_motor_action(trajectory[0], calib)

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
        # the main thread forever with no error.
        kill_switch = StdinKillSwitch()
        print("Type 'q' + Enter at any time to stop playback and safely ramp down.")

        print(f"Ramping to first frame over {args.ramp_duration}s...")
        current_action = ramp_to(robot, current_action, target_action, args.ramp_duration)
        print("Ramp complete. Replaying. Type 'q' + Enter to stop.")

        dt = 1.0 / args.playback_hz
        max_delta = args.max_joint_speed * dt
        gripper_max_delta = args.gripper_max_speed * dt
        halted = False

        for step_idx, sim_joints in enumerate(trajectory):
            loop_start = time.time()

            if kill_switch.triggered:
                print(f"\nKill switch pressed at step {step_idx}/{len(trajectory)}. Ramping down and disabling.")
                halted = True
                break

            desired = sim_joints_to_motor_action(sim_joints, calib)
            current_action = clamp_step(current_action, desired, max_delta, gripper_max_delta)
            robot.send_action(current_action)

            if step_idx % 20 == 0:
                print(f"  step {step_idx}/{len(trajectory)}")

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        if not halted:
            print(f"\nReplay of all {len(trajectory)} steps complete.")

        print("Ramping down to a safe hold before disabling...")
        safe_hold = get_current_pos_action(robot)
        ramp_to(robot, current_action, safe_hold, duration_s=1.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors.")
    finally:
        try:
            robot.disconnect()
        except Exception:
            print("WARNING: error during disconnect -- verify motors are physically de-energized.")
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
