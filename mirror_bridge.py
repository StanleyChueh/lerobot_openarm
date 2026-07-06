#!/usr/bin/env python
"""Mirror a live Isaac Sim OpenArm teleop session onto the real dual-arm OpenArm follower.

This is the real-hardware half of a two-process bridge. The other half is an opt-in UDP
broadcaster added to IsaacLab's scripts/tools/record_demos_openarm.py (--mirror_udp_port).
That process never touches hardware; this process never touches Isaac Sim. They only share
a UDP socket carrying {"seq": int, "t": float, "joints": {joint_name: radians, ...}}.

REQUIRED BEFORE RUNNING THIS SCRIPT (Phase 0 bench verification -- do this by hand first):

  1. Zero-calibration check: with the real arm safe to move by hand, compare its raw joint
     readings (via test_left_joints.py or safe_probe.py) against the sim's default pose (all
     arm joints at 0.0 rad, see stack_joint_pos_env_cfg.py). If they don't match at the same
     physical pose, run `openarm-can-set-zero` yourself first -- this script will not do
     that for you.

  2. Per-joint sign check: for each joint, nudge it a few degrees (see safe_probe.py) and
     confirm it moves the same direction as a positive delta in the sim viewer. Any joint
     that moves opposite gets sign=-1 in calibration.json.

  3. Gripper scale check: ramp LJ8.pos/RJ8.pos through its full range and record the raw
     motor value at fully-open and fully-closed. Those become gripper.open_raw /
     gripper.closed_raw in calibration.json.

Fill in calibration.json (see calibration.example.json for the schema) with what you found.
This script refuses to start without a real calibration file -- there is no safe default.

Even with calibration done, the very first run still performs a startup handshake: it will
refuse to move the arm if the real robot's current pose doesn't already closely match the
sim's current pose, and it asks for a typed confirmation before any motion begins.

Note: relies on OpenArmFollower.get_observation() using a generous recv_all() timeout
(patched in robots/umeow_openarm_follower/openarm_follower.py on 2026-07-01) -- the
500-microsecond default was found to return stale/never-updated positions.
"""

import argparse
import json
import logging
import socket
import threading
import time

from reset_to_rest_pose import reset_to_rest_pose
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    StdinKillSwitch,
    clamp_step,
    get_current_pos_action,
    load_calibration,
    motor_action_to_sim_joints,
    ramp_to,
    run_startup_handshake,
    sim_joints_to_motor_action,
)

logger = logging.getLogger("mirror_bridge")


class LatestPacketReceiver:
    """Background UDP listener that only ever keeps the newest packet."""

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._latest = None  # (seq, recv_time, joints_dict)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                packet = json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            with self._lock:
                self._latest = (packet["seq"], time.time(), packet["joints"])

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()
        self._sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration.json (see calibration.example.json)")
    parser.add_argument("--udp-host", type=str, default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, required=True, help="Must match --mirror_udp_port used in Isaac Sim")
    parser.add_argument("--right-port", type=str, default="can2")
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument("--model-path", type=str, required=True, help="Path to openarm_description.urdf for gravity comp")
    parser.add_argument("--max-joint-speed", type=float, default=0.3, help="rad/s cap applied to every arm joint's per-tick motion. Conservative default; raise only after validating on your setup.")
    parser.add_argument("--gripper-max-speed", type=float, default=8.0, help="rad/s cap for the gripper channel specifically -- much higher than the arm cap, since gripper commands are a near-instant open/closed toggle, not a smooth trajectory")
    parser.add_argument("--handshake-tolerance", type=float, default=0.1, help="rad; abort startup if any arm joint differs from sim by more than this")
    parser.add_argument("--gripper-handshake-tolerance", type=float, default=1.3, help="rad; separate, more generous tolerance for the gripper channel -- open/closed state legitimately varies between episodes")
    parser.add_argument("--ramp-duration", type=float, default=2.0, help="seconds to smoothly move from real current pose to sim's current pose at startup")
    parser.add_argument("--stale-ms", type=float, default=150.0, help="hold last command if no new packet within this long")
    parser.add_argument("--timeout-ms", type=float, default=1000.0, help="ramp down and disable if no new packet within this long")
    parser.add_argument("--loop-hz", type=float, default=50.0)
    parser.add_argument(
        "--feedback-port", type=int, default=0,
        help="If nonzero, read back the arm's ACTUAL position every tick (extra CAN read -- may slow"
        " the loop below --loop-hz) and send it back to 127.0.0.1:<port>, inverse-mapped to sim joint"
        " names, for record_demos_openarm.py's --mirror_feedback_port to plot against sim. Off by"
        " default since normal mirroring doesn't need the extra read.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    calib = load_calibration(args.calibration)

    receiver = LatestPacketReceiver(args.udp_host, args.udp_port)
    print("Listening for sim packets...")

    feedback_sock = None
    feedback_addr = None
    if args.feedback_port:
        feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        feedback_addr = (args.udp_host, args.feedback_port)
        print(f"[FEEDBACK] Will send real joint feedback to {args.udp_host}:{args.feedback_port}")

    robot_cfg = OpenArmFollowerConfig(
        right_port=args.right_port,
        left_port=args.left_port,
        enable_fd=True,  # matches deploy_ACT.py / record.py / teleop.py -- CAN-FD is always used in this codebase
        model_path=args.model_path,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    try:
        current_action = get_current_pos_action(robot)

        print("Waiting for first packet from Isaac Sim...")
        deadline = time.time() + 10.0
        packet = None
        while time.time() < deadline:
            packet = receiver.latest()
            if packet is not None:
                break
            time.sleep(0.1)
        if packet is None:
            print("No packet received from Isaac Sim within 10s. Is --mirror_udp_port set and matching? Aborting.")
            return

        _, _, sim_joints = packet
        target_action = sim_joints_to_motor_action(sim_joints, calib)

        if not run_startup_handshake(robot, target_action, args.handshake_tolerance, args.gripper_handshake_tolerance):
            retry = input(
                "\nThis is often just the real arm having drifted since a previous session."
                " Try moving it to match sim's current pose, then re-check? [y/N]: "
            )
            if retry.strip().lower() != "y":
                return
            # reset_to_rest_pose() has its own independent handshake gate and its own typed "YES"
            # confirmation before it moves anything -- this does not bypass either check, it's a
            # separate, explicitly-offered corrective action. We re-verify below rather than
            # assuming it fixed things.
            if not reset_to_rest_pose(robot, calib, target_action=target_action):
                return
            if not run_startup_handshake(robot, target_action, args.handshake_tolerance, args.gripper_handshake_tolerance):
                print("Still mismatched against the sim pose after resetting. Aborting.")
                return

        confirm = input("Type YES to ramp the real arm to the sim's pose and begin mirroring: ")
        if confirm.strip() != "YES":
            print("Not confirmed. Aborting without moving the arm.")
            return

        # Started only now, not before the confirmation prompt above -- this thread
        # continuously reads stdin in the background, and starting it earlier races
        # with input() for whoever typed "YES", occasionally swallowing it and hanging
        # the main thread forever with no error.
        kill_switch = StdinKillSwitch()
        print("Type 'q' + Enter at any time to stop the arm.")

        print(f"Ramping to sim pose over {args.ramp_duration}s...")
        current_action = ramp_to(robot, current_action, target_action, args.ramp_duration)
        print("Ramp complete. Mirroring live. Type 'q' + Enter to stop.")

        last_seq = packet[0]
        last_command_time = time.time()
        last_packet_time = packet[1]
        dt = 1.0 / args.loop_hz
        halted = False

        while not halted:
            loop_start = time.time()

            if kill_switch.triggered:
                print("\nKill switch pressed. Ramping down and disabling motors.")
                halted = True
                break

            packet = receiver.latest()
            now = time.time()

            if packet is not None and packet[0] != last_seq:
                last_seq, last_packet_time, sim_joints = packet
                desired = sim_joints_to_motor_action(sim_joints, calib)
                tick_dt = now - last_command_time
                max_delta = args.max_joint_speed * max(tick_dt, dt)
                gripper_max_delta = args.gripper_max_speed * max(tick_dt, dt)
                current_action = clamp_step(current_action, desired, max_delta, gripper_max_delta)
                robot.send_action(current_action)
                last_command_time = now
            else:
                staleness_ms = (now - last_packet_time) * 1000.0
                if staleness_ms > args.timeout_ms:
                    print(f"\nNo packet for {staleness_ms:.0f}ms (> --timeout-ms). Ramping down and disabling.")
                    halted = True
                    break
                elif staleness_ms > args.stale_ms:
                    logger.warning(f"Stale packet ({staleness_ms:.0f}ms) -- holding last position.")
                    robot.send_action(current_action)
                    last_command_time = now

            if feedback_sock is not None:
                try:
                    actual = get_current_pos_action(robot)
                    sim_joints_fb = motor_action_to_sim_joints(actual, calib)
                    feedback_sock.sendto(
                        json.dumps({"t": time.time(), "joints": sim_joints_fb}).encode("utf-8"), feedback_addr
                    )
                except (RuntimeError, OSError):
                    pass  # best-effort only -- never let a read glitch or networking hiccup break mirroring

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        print("Ramping down to a safe hold before disabling...")
        safe_hold = get_current_pos_action(robot)
        ramp_to(robot, current_action, safe_hold, duration_s=1.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors.")
    finally:
        receiver.stop()
        if feedback_sock is not None:
            feedback_sock.close()
        try:
            robot.disconnect()
        except Exception:
            logger.exception("Error during disconnect -- verify motors are physically de-energized.")
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
