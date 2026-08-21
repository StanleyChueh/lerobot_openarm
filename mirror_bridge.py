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

Startup: rather than requiring the real arm to already be at sim's pose, this drives it there
along a speed-limited ramp (--approach-speed, --max-approach-delta) and asks for a typed
confirmation first -- see approach_pose() in sim_bridge_common.py. It still refuses outright if a
joint would have to travel further than --max-approach-delta, which is the case the old
abort-on-mismatch check was really guarding: a gap that large means the calibration or zeroing is
wrong, not that the arm drifted. Pass --yes to skip the confirmation for an unattended run.

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

from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    StdinKillSwitch,
    approach_pose,
    compute_target_velocity,
    clamp_step,
    get_current_pos_action,
    load_calibration,
    motor_action_to_sim_joints,
    ramp_to,
    raw_to_gripper_sim,
    sim_joints_to_motor_action,
)

logger = logging.getLogger("mirror_bridge")

GRIPPER_WIDTH_PRINT_PERIOD_S = 0.5  # throttle -- the mirror loop runs at --loop-hz (default 50Hz)


def _sim_finger_val(sim_joints: dict, side: str) -> float:
    return next((v for k, v in sim_joints.items() if k.startswith(f"openarm_{side}_finger_joint")), 0.0)


def _print_gripper_widths(sim_joints: dict, actual: dict, calib: dict) -> None:
    """Print sim-commanded vs. real-measured gripper opening width (mm) side by side.

    `actual` is the real gripper position read fresh (not the clamped commanded target) so
    a grasped object stalling the real gripper short of its commanded width is visible here.
    """
    for side, prefix in (("left", "L"), ("right", "R")):
        grip = calib[side]["gripper"]
        sim_mm = _sim_finger_val(sim_joints, side) * 2000.0
        real_mm = raw_to_gripper_sim(actual[f"{prefix}J8.pos"], grip["open_raw"], grip["closed_raw"]) * 2000.0
        print(f"[GRIPPER {side.upper():5s}] sim={sim_mm:5.1f}mm  real={real_mm:5.1f}mm")


class GripperStallWatchdog:
    """Best-effort detector for a gripper motor that has stopped responding to commands.

    The Damiao DM4310 gripper motor reports a fault/error code (overcurrent, overload,
    overtemp, etc.) in every CAN feedback frame, but the openarm_can binding this codebase
    uses never decodes that byte and exposes no clear-error call -- so a real fault (e.g.
    the motor's own overcurrent/overload protection tripping after holding grasp torque
    against a stalled object for a while) is invisible to us except behaviorally: the motor
    stops tracking commanded position entirely, even open/close commands that used to work.

    This can NOT simply check "actual != target": a normal grasp hold against an object
    legitimately sits away from its fully-closed target for as long as the grasp lasts --
    that is the gripper working correctly, not a fault. Instead this checks whether the
    actual position responds at all when the COMMANDED target moves substantially -- that
    only fails to happen when the motor has genuinely stopped listening.
    """

    RESPONSE_TARGET_RAD = 0.15  # commanded target must move at least this much to test response
    RESPONSE_ACTUAL_RAD = 0.02  # actual position moving less than this counts as "didn't respond"
    STUCK_POLLS = 3  # consecutive non-responses (at the caller's poll cadence) before recovering

    def __init__(self):
        self._prev_target = {"left": None, "right": None}
        self._prev_actual = {"left": None, "right": None}
        self._stuck_count = {"left": 0, "right": 0}

    def check(self, side: str, target: float, actual: float) -> bool:
        """Return True if `side`'s gripper looks stalled and recovery should be attempted."""
        prev_target, prev_actual = self._prev_target[side], self._prev_actual[side]
        self._prev_target[side], self._prev_actual[side] = target, actual
        if prev_target is None:
            return False
        target_moved = abs(target - prev_target)
        actual_moved = abs(actual - prev_actual)
        if target_moved >= self.RESPONSE_TARGET_RAD and actual_moved < self.RESPONSE_ACTUAL_RAD:
            self._stuck_count[side] += 1
        else:
            self._stuck_count[side] = 0
        if self._stuck_count[side] >= self.STUCK_POLLS:
            self._stuck_count[side] = 0  # avoid re-triggering every poll while recovery is retried
            return True
        return False


def _recover_gripper(robot, side: str) -> None:
    """Attempt to clear a suspected motor-side fault-latch by power-cycling (disable then
    re-enable) JUST this gripper's motor -- not the 7 arm joints on the same CAN bus, which
    keep holding their last commanded position throughout. This is a heuristic recovery
    (no real fault-clear API exists -- see GripperStallWatchdog docstring), not a guaranteed
    fix: if the motor is latched in a way disable/enable doesn't reset, it will stay stuck
    and this will just repeat every time the watchdog re-triggers."""
    arm = robot.left_arm if side == "left" else robot.right_arm
    print(f"\n[GRIPPER {side.upper()}] not responding to commands -- attempting recovery"
          " (disable/enable this gripper motor only; likely a motor-side overcurrent/overload"
          " fault-latch after a sustained grasp).")
    try:
        arm.get_gripper().disable_all()
        time.sleep(0.1)
        arm.get_gripper().enable_all()
    except Exception:
        logger.exception(f"[GRIPPER {side.upper()}] recovery attempt raised -- may still be stuck")


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
    parser.add_argument("--handshake-tolerance", type=float, default=0.05, help="rad; if every joint is already within this of sim's pose the startup approach is skipped as a no-op. NOT an abort threshold any more -- exceeding it just means the arm ramps there (see --max-approach-delta for the gate that does refuse).")
    parser.add_argument("--max-approach-delta", type=float, default=1.8, help="rad; REFUSE to start if any arm joint would have to travel further than this to reach sim's pose. This is the real safety gate: a gap this large means the calibration or zeroing is wrong, not that the arm drifted, and auto-moving on that assumption is what must not happen.")
    parser.add_argument("--approach-speed", type=float, default=0.3, help="rad/s ceiling for the startup approach to sim's pose. The ramp duration is derived from this and the furthest-travelling joint, so no joint exceeds it.")
    parser.add_argument("--yes", action="store_true", help="Skip the typed YES confirmation before the startup approach moves the arm. For unattended runs only -- --max-approach-delta still applies.")
    parser.add_argument("--ramp-duration", type=float, default=2.0, help="seconds; MINIMUM duration of the startup approach. A longer one is used automatically when --approach-speed requires it for the distance being covered.")
    parser.add_argument("--first-packet-timeout", type=float, default=600.0, help="Seconds to wait for the sim's first packet before giving up; 0 waits indefinitely. Generous by default because the sim-side script takes minutes to reach the point where it starts broadcasting, and nothing has been commanded to the arm while this waits.")
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

        # The old hardcoded 10s was far too tight for the workflow this is actually used in: the
        # sim-side script does not broadcast anything until it has loaded Isaac Sim, connected to
        # the policy server and reset the scene, which is minutes, and if it dies on the way (a
        # refused policy-server connection, say) this would abort long before the operator could
        # see why. Waiting costs nothing -- nothing has been commanded yet.
        wait_desc = "indefinitely" if args.first_packet_timeout <= 0 else f"up to {args.first_packet_timeout:g}s"
        print(f"Waiting for first packet from Isaac Sim ({wait_desc}; Ctrl-C to give up)...")
        deadline = None if args.first_packet_timeout <= 0 else time.time() + args.first_packet_timeout
        packet = None
        waited = 0.0
        while deadline is None or time.time() < deadline:
            packet = receiver.latest()
            if packet is not None:
                break
            time.sleep(0.1)
            waited += 0.1
            if abs(waited % 15.0) < 0.05:
                print(f"  … still no packet after {waited:.0f}s. The sim-side script only starts"
                      " broadcasting once it reaches its first rollout hold.")
        if packet is None:
            print(f"No packet received from Isaac Sim within {args.first_packet_timeout:g}s."
                  " Check --udp-port here matches --mirror_udp_port there, and that the sim-side"
                  " script is still alive. Aborting.")
            return

        _, _, sim_joints = packet
        target_action = sim_joints_to_motor_action(sim_joints, calib)

        # Go to sim's pose along a speed-limited ramp instead of demanding the arm already be
        # there. The old flow compared the two, aborted on any joint past --handshake-tolerance,
        # and offered a rest-pose reset that could not fix it anyway: the residual it measures is
        # mostly steady-state droop the arm reproduces every time it holds a pose, not drift a
        # human can correct by repositioning. approach_pose() keeps the part of that check that
        # was actually load-bearing -- refusing a move so large it implies bad calibration -- as
        # --max-approach-delta. See its docstring.
        approached = approach_pose(
            robot, target_action,
            label="sim's current pose",
            arm_speed=args.approach_speed,
            gripper_speed=args.gripper_max_speed,
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
        print("Mirroring live. Type 'q' + Enter at any time to stop the arm.")

        last_seq = packet[0]
        last_command_time = time.time()
        last_packet_time = packet[1]
        last_width_print_time = 0.0
        target_vel = {}
        stall_watchdog = GripperStallWatchdog()
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
                target_action = clamp_step(current_action, desired, max_delta, gripper_max_delta)
                target_vel = compute_target_velocity(current_action, target_action, tick_dt, args.max_joint_speed)
                robot.send_action(target_action, target_vel)
                current_action = target_action
                last_command_time = now
            else:
                staleness_ms = (now - last_packet_time) * 1000.0
                if staleness_ms > args.timeout_ms:
                    print(f"\nNo packet for {staleness_ms:.0f}ms (> --timeout-ms). Ramping down and disabling.")
                    halted = True
                    break
                elif staleness_ms > args.stale_ms:
                    logger.warning(f"Stale packet ({staleness_ms:.0f}ms) -- holding last position.")
                    target_vel =  {k: 0.0 for k in target_vel.keys()}
                    robot.send_action(current_action, target_vel)
                    last_command_time = now

            if now - last_width_print_time >= GRIPPER_WIDTH_PRINT_PERIOD_S:
                last_width_print_time = now
                try:
                    actual_gripper = get_current_pos_action(robot)
                except RuntimeError as e:
                    logger.warning(f"[GRIPPER] could not read real position: {e}")
                    actual_gripper = None
                if actual_gripper is not None:
                    _print_gripper_widths(sim_joints, actual_gripper, calib)
                    for side, prefix in (("left", "L"), ("right", "R")):
                        key = f"{prefix}J8.pos"
                        if stall_watchdog.check(side, current_action[key], actual_gripper[key]):
                            _recover_gripper(robot, side)

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