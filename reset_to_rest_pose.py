#!/usr/bin/env python
"""Move the real dual-arm OpenArm follower to the sim's default rest pose --
all arm joints at 0 rad (mapped through calibration), gripper fully open.

Run this BEFORE mirror_bridge.py / replay_sim_dataset.py / replay_hf_sim_episode.py
whenever the real arm has drifted from that pose (e.g. left wherever a previous
session's teleop ended), so those scripts' startup handshake passes cleanly
instead of aborting on a large real-vs-sim delta.

This does NOT touch or loosen the handshake safety check in those scripts --
it's a separate, explicit step you choose to run that proactively fixes the
thing that check would otherwise complain about.

Uses the same startup handshake / ramp / retry-safe-read machinery as those
scripts (see sim_bridge_common.py), just with a looser handshake tolerance --
appropriate here since "the arm drifted from rest" is the expected, normal
reason to run this script, not a fault condition. A truly extreme mismatch
(something actually wrong, not just drift) still pauses for confirmation.

Usage:
  python reset_to_rest_pose.py --calibration calibration.json \\
      --model-path model/openarm_description_leader.urdf
"""

import argparse

from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    ARM_JOINT_KEYS,
    get_current_pos_action,
    load_calibration,
    ramp_to,
    run_startup_handshake,
)


def rest_pose_action(calib: dict) -> dict:
    """All arm joints at sim-joint-angle 0 (mapped through calibration), gripper
    fully open -- matches this task's default env.reset() pose."""
    action = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        sec = calib[side]
        for n, jkey in enumerate(ARM_JOINT_KEYS, start=1):
            sign = sec["sign"][jkey]
            offset = sec["offset_rad"][jkey]
            action[f"{prefix}J{n}.pos"] = sign * 0.0 + offset
        action[f"{prefix}J8.pos"] = sec["gripper"]["open_raw"]
    return action


def reset_to_rest_pose(
    robot, calib: dict, ramp_duration: float = 4.0,
    handshake_tolerance: float = 0.5, gripper_handshake_tolerance: float = 1.3,
    target_action: dict | None = None,
) -> bool:
    """Ramp `robot` toward `target_action` (or, if not given, calibration's assumed
    rest pose -- all joints at 0, gripper open). Callable from other scripts (e.g.
    mirror_bridge.py) as an explicit recovery step when their own handshake fails --
    this does NOT bypass that caller's check: it still requires its own independent
    handshake pass and its own typed "YES" confirmation before moving anything, and
    the caller is expected to re-run ITS OWN handshake afterward rather than assume
    success. Returns True if the ramp completed, False if aborted at either gate.

    Pass the caller's already-known live target_action when available (e.g. sim's
    actual current pose in mirror_bridge.py) rather than relying on the default --
    if sim itself has already moved away from its own rest pose (e.g. teleop started
    before the bridge finished connecting), resetting the real arm toward an assumed
    rest is pointless: it'll just create a new, different mismatch against sim's
    actual current pose instead of fixing the original one.
    """
    if target_action is None:
        target_action = rest_pose_action(calib)
    current_action = get_current_pos_action(robot)

    if not run_startup_handshake(robot, target_action, handshake_tolerance, gripper_handshake_tolerance):
        print("Aborting reset -- this mismatch is larger than normal drift. Check the arm isn't obstructed before retrying.")
        return False

    confirm = input("Type YES to ramp the real arm to the rest pose: ")
    if confirm.strip() != "YES":
        print("Not confirmed. Aborting without moving the arm.")
        return False

    print(f"Ramping to rest pose over {ramp_duration}s...")
    ramp_to(robot, current_action, target_action, ramp_duration)
    print("Done -- arm should now be at rest pose.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration.json")
    parser.add_argument("--right-port", type=str, default="can2")
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument("--model-path", type=str, required=True, help="Path to openarm_description.urdf for gravity comp")
    parser.add_argument(
        "--handshake-tolerance", type=float, default=0.5,
        help="rad; looser than mirror_bridge.py's default (0.1) since normal drift after repeated"
        " sessions is exactly what this script exists to fix. Still pauses on a truly extreme"
        " mismatch that could mean something is actually wrong (e.g. an obstruction).",
    )
    parser.add_argument("--gripper-handshake-tolerance", type=float, default=1.3)
    parser.add_argument(
        "--ramp-duration", type=float, default=4.0,
        help="seconds; longer than the other scripts' default since this may cover more distance than normal frame-to-frame motion",
    )
    args = parser.parse_args()

    calib = load_calibration(args.calibration)

    robot_cfg = OpenArmFollowerConfig(
        right_port=args.right_port,
        left_port=args.left_port,
        enable_fd=True,
        model_path=args.model_path,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    try:
        reset_to_rest_pose(robot, calib, args.ramp_duration, args.handshake_tolerance, args.gripper_handshake_tolerance)
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
