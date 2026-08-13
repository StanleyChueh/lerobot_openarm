#!/usr/bin/env python
"""Mirror a live Isaac Sim OpenArm teleop session onto the real LEFT OpenArm arm only.

This is the real-hardware half of a two-process bridge. The Isaac Lab recorder broadcasts
only left-arm simulation joints through UDP. This process converts those simulation joint
values to motor commands and updates only the real robot's left-arm channels (LJ1-LJ8).

The OpenArmFollower and sim_bridge_common APIs use full dual-arm action dictionaries. To
remain compatible with those APIs, this script reads the full robot state once at startup
and freezes every non-left motor target at that initial value. Incoming right-arm simulation
joints, if any, are ignored. Right-arm targets are never updated by teleoperation.

UDP packet format:
    {"seq": int, "t": float, "joints": {joint_name: radians, ...}}

REQUIRED BEFORE RUNNING THIS SCRIPT:

  1. Zero-calibration check: with the real LEFT arm safe to move by hand, compare its raw
     joint readings against the simulation's default pose. If they do not match at the same
     physical pose, run `openarm-can-set-zero` yourself first. This script does not modify
     motor zero calibration.

  2. Per-joint sign check: verify LJ1-LJ7 move in the same direction as positive simulation
     joint deltas. Any reversed joint must use sign=-1 in calibration.json.

  3. Left-gripper scale check: determine LJ8.pos at fully open and fully closed and store the
     values as calibration["left"]["gripper"]["open_raw"] and ["closed_raw"].

The calibration file must remain compatible with sim_bridge_common. Because the shared
conversion helpers operate on a full OpenArm state, use the same calibration schema as the
original dual-arm bridge even though only the left side is commanded.

Before motion, the script performs the existing startup handshake and requires the operator
to type YES. Type `q` followed by Enter during mirroring to stop.
"""

import argparse
import json
import logging
import socket
import threading
import time
from typing import Any

from reset_to_rest_pose import reset_to_rest_pose
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    StdinKillSwitch,
    clamp_step,
    get_current_pos_action,
    load_calibration,
    motor_action_to_sim_joints,
    ramp_to,
    raw_to_gripper_sim,
    run_startup_handshake,
    sim_joints_to_motor_action,
)

logger = logging.getLogger("mirror_bridge_left_only")

LEFT_SIM_PREFIX = "openarm_left_"
LEFT_MOTOR_PREFIX = "LJ"
LEFT_GRIPPER_KEY = "LJ8.pos"model/openarm_description_leader.urdf  
GRIPPER_WIDTH_PRINT_PERIOD_S = 0.5


def _is_left_sim_joint(name: str) -> bool:
    """Return True when a simulation joint belongs to the left OpenArm side."""
    return name.startswith(LEFT_SIM_PREFIX)


def _is_left_motor_key(name: str) -> bool:
    """Return True for left OpenArm motor action keys such as LJ1.pos ... LJ8.pos."""
    return name.startswith(LEFT_MOTOR_PREFIX)


def _filter_left_sim_joints(sim_joints: dict[str, float]) -> dict[str, float]:
    """Drop all non-left simulation joints from an incoming UDP packet."""
    return {name: value for name, value in sim_joints.items() if _is_left_sim_joint(name)}


def _filter_left_motor_action(action: dict[str, float]) -> dict[str, float]:
    """Keep only left motor keys from a full OpenArm action/observation dictionary."""
    return {name: value for name, value in action.items() if _is_left_motor_key(name)}


def _freeze_non_left_targets(
    desired_action: dict[str, float],
    non_left_hold_action: dict[str, float],
) -> dict[str, float]:
    """Overwrite every non-left target with its fixed startup hold value."""
    frozen = dict(desired_action)
    for key, value in non_left_hold_action.items():
        frozen[key] = value
    return frozen


def _build_left_only_target(
    incoming_sim_joints: dict[str, float],
    reference_action: dict[str, float],
    non_left_hold_action: dict[str, float],
    calib: dict[str, Any],
) -> dict[str, float]:
    """Convert a partial left-only simulation packet into a full compatible motor action.

    The common converter expects the complete simulation joint dictionary. We reconstruct
    missing joints from `reference_action`, overwrite only left simulation joints using the
    incoming packet, run the standard calibrated conversion, and finally restore all
    non-left motor targets to their fixed startup values.
    """
    left_sim_joints = _filter_left_sim_joints(incoming_sim_joints)
    if not left_sim_joints:
        raise ValueError(
            "UDP packet contains no left OpenArm joints. Expected names beginning with "
            f"'{LEFT_SIM_PREFIX}'."
        )

    reference_sim_joints = motor_action_to_sim_joints(reference_action, calib)
    complete_sim_joints = dict(reference_sim_joints)
    complete_sim_joints.update(left_sim_joints)

    desired_action = sim_joints_to_motor_action(complete_sim_joints, calib)
    return _freeze_non_left_targets(desired_action, non_left_hold_action)


def _left_sim_finger_val(sim_joints: dict[str, float]) -> float:
    """Return one left finger-joint value from a simulation joint dictionary."""
    return next(
        (
            value
            for name, value in sim_joints.items()
            if name.startswith("openarm_left_finger_joint")
        ),
        0.0,
    )


def _print_left_gripper_width(
    sim_joints: dict[str, float],
    actual_action: dict[str, float],
    calib: dict[str, Any],
) -> None:
    """Print left simulated-commanded and real-measured gripper widths in millimetres."""
    if LEFT_GRIPPER_KEY not in actual_action:
        logger.warning("Left gripper feedback key %s is missing.", LEFT_GRIPPER_KEY)
        return

    grip = calib["left"]["gripper"]
    sim_mm = _left_sim_finger_val(sim_joints) * 2000.0
    real_mm = (
        raw_to_gripper_sim(
            actual_action[LEFT_GRIPPER_KEY],
            grip["open_raw"],
            grip["closed_raw"],
        )
        * 2000.0
    )
    print(f"[GRIPPER LEFT] sim={sim_mm:5.1f}mm  real={real_mm:5.1f}mm")


class LeftGripperStallWatchdog:
    """Best-effort detector for a non-responsive left gripper motor.

    A normal grasp may hold away from the fully closed target. Therefore this watchdog does
    not compare target and actual values directly. It checks whether the actual gripper moves
    after the commanded target itself changes substantially.
    """

    RESPONSE_TARGET_RAD = 0.15
    RESPONSE_ACTUAL_RAD = 0.02
    STUCK_POLLS = 3

    def __init__(self) -> None:
        self._prev_target: float | None = None
        self._prev_actual: float | None = None
        self._stuck_count = 0

    def check(self, target: float, actual: float) -> bool:
        """Return True when the left gripper appears stalled."""
        prev_target = self._prev_target
        prev_actual = self._prev_actual
        self._prev_target = target
        self._prev_actual = actual

        if prev_target is None or prev_actual is None:
            return False

        target_moved = abs(target - prev_target)
        actual_moved = abs(actual - prev_actual)

        if target_moved >= self.RESPONSE_TARGET_RAD and actual_moved < self.RESPONSE_ACTUAL_RAD:
            self._stuck_count += 1
        else:
            self._stuck_count = 0

        if self._stuck_count >= self.STUCK_POLLS:
            self._stuck_count = 0
            return True
        return False


def _recover_left_gripper(robot: OpenArmFollower) -> None:
    """Power-cycle only the left gripper motor as a heuristic stall recovery."""
    print(
        "\n[GRIPPER LEFT] not responding to commands -- attempting recovery "
        "by disabling and re-enabling the left gripper motor only."
    )
    try:
        robot.left_arm.get_gripper().disable_all()
        time.sleep(0.1)
        robot.left_arm.get_gripper().enable_all()
    except Exception:
        logger.exception("[GRIPPER LEFT] recovery attempt raised; the gripper may remain stuck.")


class LatestPacketReceiver:
    """Background UDP listener that keeps only the newest valid packet."""

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._latest: tuple[int, float, dict[str, float]] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                packet = json.loads(data.decode("utf-8"))
                seq = int(packet["seq"])
                joints = packet["joints"]
                if not isinstance(joints, dict):
                    raise TypeError("packet['joints'] must be a dictionary")
                joints = {str(name): float(value) for name, value in joints.items()}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Ignoring malformed UDP packet: %s", exc)
                continue

            left_joints = _filter_left_sim_joints(joints)
            if not left_joints:
                logger.warning(
                    "Ignoring packet %d because it contains no joints beginning with '%s'.",
                    seq,
                    LEFT_SIM_PREFIX,
                )
                continue

            with self._lock:
                self._latest = (seq, time.time(), left_joints)

    def latest(self) -> tuple[int, float, dict[str, float]] | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        self._sock.close()


def _wait_for_first_packet(
    receiver: LatestPacketReceiver,
    timeout_s: float = 10.0,
) -> tuple[int, float, dict[str, float]] | None:
    """Wait up to `timeout_s` for the first valid left-arm simulation packet."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        packet = receiver.latest()
        if packet is not None:
            return packet
        time.sleep(0.1)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Path to calibration.json using the sim_bridge_common schema.",
    )
    parser.add_argument("--udp-host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--udp-port",
        type=int,
        required=True,
        help="Must match --mirror_udp_port used by the Isaac Lab recorder.",
    )
    parser.add_argument(
        "--right-port",
        type=str,
        default="can1",
        help=(
            "Right-arm CAN interface required by OpenArmFollowerConfig. The right-arm motor "
            "targets are frozen and are never updated from teleoperation."
        ),
    )
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to openarm_description.urdf for gravity compensation.",
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=0.3,
        help="Maximum left-arm joint speed in rad/s.",
    )
    parser.add_argument(
        "--gripper-max-speed",
        type=float,
        default=8.0,
        help="Maximum left-gripper command speed in rad/s.",
    )
    parser.add_argument(
        "--handshake-tolerance",
        type=float,
        default=0.1,
        help="Arm-joint startup tolerance in radians.",
    )
    parser.add_argument(
        "--gripper-handshake-tolerance",
        type=float,
        default=1.3,
        help="Left-gripper startup tolerance in radians.",
    )
    parser.add_argument(
        "--ramp-duration",
        type=float,
        default=2.0,
        help="Seconds used to ramp the real left arm to the initial simulation pose.",
    )
    parser.add_argument(
        "--stale-ms",
        type=float,
        default=150.0,
        help="Hold the last command after this packet age.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=float,
        default=1000.0,
        help="Stop and disable after this packet age.",
    )
    parser.add_argument("--loop-hz", type=float, default=50.0)
    parser.add_argument(
        "--feedback-port",
        type=int,
        default=0,
        help=(
            "If nonzero, send only actual left-arm simulation-mapped joint feedback to "
            "<udp-host>:<feedback-port>."
        ),
    )
    args = parser.parse_args()

    if args.max_joint_speed <= 0.0:
        parser.error("--max-joint-speed must be greater than zero")
    if args.gripper_max_speed <= 0.0:
        parser.error("--gripper-max-speed must be greater than zero")
    if args.loop_hz <= 0.0:
        parser.error("--loop-hz must be greater than zero")
    if args.stale_ms < 0.0 or args.timeout_ms <= args.stale_ms:
        parser.error("--timeout-ms must be greater than --stale-ms, and --stale-ms must be nonnegative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    calib = load_calibration(args.calibration)
    if "left" not in calib or "gripper" not in calib["left"]:
        raise ValueError("Calibration file is missing calibration['left']['gripper'].")

    receiver = LatestPacketReceiver(args.udp_host, args.udp_port)
    print(f"Listening for LEFT-arm simulation packets on {args.udp_host}:{args.udp_port}...")

    feedback_sock: socket.socket | None = None
    feedback_addr: tuple[str, int] | None = None
    if args.feedback_port:
        feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        feedback_addr = (args.udp_host, args.feedback_port)
        print(
            f"[FEEDBACK] Will send LEFT-arm real joint feedback to "
            f"{args.udp_host}:{args.feedback_port}"
        )

    robot_cfg = OpenArmFollowerConfig(
        right_port=args.right_port,
        left_port=args.left_port,
        enable_fd=True,
        model_path=args.model_path,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    try:
        # The follower API returns a full dual-arm action dictionary. Save all non-left
        # values once and keep them fixed for the complete session.
        current_action = get_current_pos_action(robot)
        non_left_hold_action = {
            key: value for key, value in current_action.items() if not _is_left_motor_key(key)
        }

        left_keys = sorted(_filter_left_motor_action(current_action))
        if not left_keys:
            raise RuntimeError(
                "Robot observation contains no left motor keys beginning with "
                f"'{LEFT_MOTOR_PREFIX}'."
            )

        print(f"[LEFT ONLY] Controllable motor keys: {left_keys}")
        print(
            f"[LEFT ONLY] Freezing {len(non_left_hold_action)} non-left motor targets at "
            "their startup values."
        )

        print("Waiting for the first LEFT-arm packet from Isaac Sim...")
        packet = _wait_for_first_packet(receiver, timeout_s=10.0)
        if packet is None:
            print(
                "No valid left-arm packet received within 10 seconds. Verify that "
                "--mirror_udp_port matches --udp-port and that the recorder broadcasts "
                "openarm_left_* joints. Aborting."
            )
            return

        _, _, sim_joints = packet
        target_action = _build_left_only_target(
            incoming_sim_joints=sim_joints,
            reference_action=current_action,
            non_left_hold_action=non_left_hold_action,
            calib=calib,
        )

        if not run_startup_handshake(
            robot,
            target_action,
            args.handshake_tolerance,
            args.gripper_handshake_tolerance,
        ):
            retry = input(
                "\nThe real LEFT arm does not match the simulation pose. Move it to match "
                "the simulation and re-check, or use the guarded reset helper. Retry with "
                "reset_to_rest_pose? [y/N]: "
            )
            if retry.strip().lower() != "y":
                return

            # The target keeps non-left channels at their measured startup values, so the
            # corrective target does not request a new right-arm pose.
            if not reset_to_rest_pose(robot, calib, target_action=target_action):
                return

            current_action = get_current_pos_action(robot)
            non_left_hold_action = {
                key: value for key, value in current_action.items() if not _is_left_motor_key(key)
            }
            target_action = _build_left_only_target(
                incoming_sim_joints=sim_joints,
                reference_action=current_action,
                non_left_hold_action=non_left_hold_action,
                calib=calib,
            )

            if not run_startup_handshake(
                robot,
                target_action,
                args.handshake_tolerance,
                args.gripper_handshake_tolerance,
            ):
                print("The LEFT arm still does not match the simulation pose. Aborting.")
                return

        confirm = input(
            "Type YES to ramp the real LEFT arm to the simulation pose and begin mirroring: "
        )
        if confirm.strip() != "YES":
            print("Not confirmed. Aborting without commanding motion.")
            return

        kill_switch = StdinKillSwitch()
        print("Type 'q' + Enter at any time to stop the left arm.")

        print(f"Ramping the LEFT arm to the simulation pose over {args.ramp_duration}s...")
        current_action = ramp_to(robot, current_action, target_action, args.ramp_duration)
        current_action = _freeze_non_left_targets(current_action, non_left_hold_action)
        print("Ramp complete. Mirroring the LEFT arm live. Type 'q' + Enter to stop.")

        last_seq = packet[0]
        last_command_time = time.time()
        last_packet_time = packet[1]
        last_width_print_time = 0.0
        stall_watchdog = LeftGripperStallWatchdog()
        dt = 1.0 / args.loop_hz
        halted = False

        while not halted:
            loop_start = time.time()

            if kill_switch.triggered:
                print("\nKill switch pressed. Stopping and disabling motors.")
                halted = True
                break

            packet = receiver.latest()
            now = time.time()

            if packet is not None and packet[0] != last_seq:
                last_seq, last_packet_time, sim_joints = packet

                desired_action = _build_left_only_target(
                    incoming_sim_joints=sim_joints,
                    reference_action=current_action,
                    non_left_hold_action=non_left_hold_action,
                    calib=calib,
                )

                tick_dt = now - last_command_time
                max_delta = args.max_joint_speed * max(tick_dt, dt)
                gripper_max_delta = args.gripper_max_speed * max(tick_dt, dt)

                current_action = clamp_step(
                    current_action,
                    desired_action,
                    max_delta,
                    gripper_max_delta,
                )
                current_action = _freeze_non_left_targets(
                    current_action,
                    non_left_hold_action,
                )
                robot.send_action(current_action)
                last_command_time = now
            else:
                staleness_ms = (now - last_packet_time) * 1000.0
                if staleness_ms > args.timeout_ms:
                    print(
                        f"\nNo packet for {staleness_ms:.0f} ms (> --timeout-ms). "
                        "Stopping and disabling motors."
                    )
                    halted = True
                    break
                if staleness_ms > args.stale_ms:
                    logger.warning(
                        "Stale packet (%.0f ms); holding the last left-arm position.",
                        staleness_ms,
                    )
                    current_action = _freeze_non_left_targets(
                        current_action,
                        non_left_hold_action,
                    )
                    robot.send_action(current_action)
                    last_command_time = now

            if now - last_width_print_time >= GRIPPER_WIDTH_PRINT_PERIOD_S:
                last_width_print_time = now
                try:
                    actual_action = get_current_pos_action(robot)
                except RuntimeError as exc:
                    logger.warning("[GRIPPER LEFT] could not read real position: %s", exc)
                    actual_action = None

                if actual_action is not None:
                    _print_left_gripper_width(sim_joints, actual_action, calib)
                    if (
                        LEFT_GRIPPER_KEY in current_action
                        and LEFT_GRIPPER_KEY in actual_action
                        and stall_watchdog.check(
                            current_action[LEFT_GRIPPER_KEY],
                            actual_action[LEFT_GRIPPER_KEY],
                        )
                    ):
                        _recover_left_gripper(robot)

            if feedback_sock is not None and feedback_addr is not None:
                try:
                    actual_action = get_current_pos_action(robot)
                    sim_feedback_all = motor_action_to_sim_joints(actual_action, calib)
                    left_sim_feedback = _filter_left_sim_joints(sim_feedback_all)
                    feedback_sock.sendto(
                        json.dumps(
                            {
                                "t": time.time(),
                                "joints": left_sim_feedback,
                            }
                        ).encode("utf-8"),
                        feedback_addr,
                    )
                except (RuntimeError, OSError, ValueError, TypeError):
                    # Feedback is best-effort and must not interrupt hardware control.
                    pass

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        print("Ramping the LEFT arm to its measured safe hold before disabling...")
        measured_action = get_current_pos_action(robot)
        safe_hold_action = dict(current_action)
        for key, value in measured_action.items():
            if _is_left_motor_key(key):
                safe_hold_action[key] = value
        safe_hold_action = _freeze_non_left_targets(safe_hold_action, non_left_hold_action)
        ramp_to(robot, current_action, safe_hold_action, duration_s=1.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors.")
    finally:
        receiver.stop()
        if feedback_sock is not None:
            feedback_sock.close()
        try:
            robot.disconnect()
        except Exception:
            logger.exception(
                "Error during disconnect; verify that the motors are physically de-energized."
            )
        print("Robot disconnected.")


if __name__ == "__main__":
    main()