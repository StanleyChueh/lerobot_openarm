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
import math
import signal
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
from real_episode_recorder import DEFAULT_CAMERAS, EpisodeEventReceiver, RealEpisodeRecorder, parse_camera_spec

logger = logging.getLogger("mirror_bridge")

GRIPPER_CHECK_PERIOD_S = 0.5  # stall watchdog (and --print-gripper-widths) cadence; the loop runs at --loop-hz


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


class LoopStats:
    """Every REPORT_PERIOD_S, one line saying where the real arm's lag is coming from.

    "The real arm feels laggy" has three unrelated causes that look the same from the headset: the
    sim is sending fewer packets than it should, this loop is overrunning its tick, or the
    --max-joint-speed clamp is holding the arm back from a target it could otherwise reach. The
    last is by far the usual one, and it is the only one a flag fixes, so it gets its own numbers:
    how often the clamp was the thing limiting a command, and how far behind it left the arm.
    """

    REPORT_PERIOD_S = 5.0
    ARM_KEYS = [f"{p}J{i}.pos" for p in "LR" for i in range(1, 8)]

    def __init__(self):
        self._reset(time.perf_counter())

    def _reset(self, now: float):
        self._t0 = now
        self._ticks = 0
        self._busy_max = 0.0
        self._commands = 0
        self._clamped = 0
        self._behind_max = 0.0

    def command(self, desired: dict, sent: dict):
        self._commands += 1
        behind = max(abs(desired[k] - sent[k]) for k in self.ARM_KEYS)
        if behind > 1e-6:
            self._clamped += 1
        self._behind_max = max(self._behind_max, behind)

    def tick(self, busy_s: float):
        self._ticks += 1
        self._busy_max = max(self._busy_max, busy_s)

    def maybe_report(self, max_joint_speed: float):
        now = time.perf_counter()
        span = now - self._t0
        if span < self.REPORT_PERIOD_S:
            return
        if self._commands:
            clamped_pct = 100.0 * self._clamped / self._commands
            line = (f"[BRIDGE] loop {self._ticks / span:.0f} Hz (worst tick {self._busy_max * 1e3:.1f} ms)"
                    f" | sim packets {self._commands / span:.0f} Hz"
                    f" | speed cap {max_joint_speed:g} rad/s limited {clamped_pct:.0f}% of commands,"
                    f" arm up to {self._behind_max:.2f} rad behind sim")
            if clamped_pct > 20:
                line += "  <-- the cap is the lag; raise --real_arm_max_joint_speed"
            print(line)
        self._reset(now)


class LatestPacketReceiver:
    """Background UDP listener that only ever keeps the newest packet."""

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._lock = threading.Lock()
        self._latest = None  # (seq, recv_time, joints_dict)
        # The newest packet's "hold_ms" (0 when absent): the sim sends one right before a blocking
        # env.reset(), to say "no packets for up to this long, and that is expected".
        self.hold_ms = 0.0
        self.shutdown = False  # set once the sim says it is exiting
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
                self.hold_ms = float(packet.get("hold_ms", 0.0))
                if packet.get("shutdown"):
                    self.shutdown = True

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
    rec = parser.add_argument_group(
        "real-robot dataset recording (see real_episode_recorder.py; set by record_demos_openarm.py's"
        " --real_arm_dataset)")
    rec.add_argument("--record-root", type=str, default=None,
                     help="If set, record the real cameras + joints into a LeRobot v3 dataset at this"
                          " directory, one episode per demo the sim saves.")
    rec.add_argument("--record-repo-id", type=str, default=None,
                     help="repo_id stored in the dataset. Default: local/<basename of --record-root>.")
    rec.add_argument("--record-task", type=str, default=None,
                     help="Task string for every frame. Required with --record-root; must match the"
                          " sim dataset's task string verbatim if the two are trained together.")
    rec.add_argument("--record-cameras", type=str, default=DEFAULT_CAMERAS,
                     help="Comma-separated <dataset_key>=<video index or udev alias> pairs.")
    rec.add_argument("--record-fps", type=int, default=30,
                     help="Dataset fps. The mirror loop runs at this rate while recording (overrides"
                          " --loop-hz), one frame per tick.")
    rec.add_argument("--record-event-port", type=int, default=5559,
                     help="UDP port the sim sends start/save/reset episode events to.")
    rec.add_argument("--record-action-source", choices=["command", "next_state"], default="command",
                     help="'command': the sim target the real arm was told to follow. 'next_state':"
                          " the next tick's measured real state (what convert_hdf5_to_lerobot.py uses"
                          " for sim data).")
    rec.add_argument("--record-vcodec", type=str, default="libsvtav1",
                     help="Video codec (libsvtav1, h264, ...). Not 'auto': it picks h264_nvenc here, which fails to open.")
    rec.add_argument("--record-view-hz", type=float, default=10.0,
                     help="Rate of the live rerun window showing the real cameras and the collection"
                          " status (collection_viewer.py). 0 disables it.")
    rec.add_argument("--record-resume", action="store_true", help="Append to an existing dataset.")
    rec.add_argument("--record-overwrite", action="store_true", help="Delete an existing dataset first.")
    parser.add_argument("--print-gripper-widths", action="store_true",
                        help="Print sim-commanded vs. real-measured gripper width every 0.5s (debugging).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    calib = load_calibration(args.calibration)

    # Cameras and the dataset come up BEFORE the CAN motors are energised, so a missing camera or a
    # refused dataset directory fails with the arm still limp.
    recorder = None
    event_receiver = None
    viewer = None
    record_every = 1
    if args.record_root:
        if not args.record_task:
            parser.error("--record-root needs --record-task")
        if args.record_resume and args.record_overwrite:
            parser.error("--record-resume and --record-overwrite are mutually exclusive")
        recorder = RealEpisodeRecorder(
            root=args.record_root, repo_id=args.record_repo_id, task=args.record_task,
            fps=args.record_fps, cameras=parse_camera_spec(args.record_cameras), calib=calib,
            resume=args.record_resume, overwrite=args.record_overwrite,
            action_source=args.record_action_source, vcodec=args.record_vcodec,
        )
        recorder.connect()
        event_receiver = EpisodeEventReceiver(args.udp_host, args.record_event_port)
        # The loop stays at least as fast as --loop-hz asked for, rounded UP to a whole multiple of
        # the dataset fps, and a frame is recorded every record_every-th tick. Running the loop AT
        # the dataset fps instead (as this first did) means a sim packet waits up to a full 33 ms
        # tick to be forwarded, and two unsynchronised 30 Hz clocks beat against each other.
        record_every = max(1, math.ceil(args.loop_hz / args.record_fps - 1e-9))
        args.loop_hz = float(args.record_fps * record_every)
        print(f"[REAL REC] Mirror loop at {args.loop_hz:g} Hz, one dataset frame every"
              f" {record_every} tick(s) = {args.record_fps} fps.")
        print(f"[REAL REC] Listening for episode events on {args.udp_host}:{args.record_event_port}")
        if args.record_view_hz > 0:
            try:
                from collection_viewer import CollectionViewer

                viewer = CollectionViewer(recorder.cameras, hz=args.record_view_hz, recorder=recorder)
            except Exception as e:
                print(f"[VIEWER] could not start the rerun window ({e}) -- recording without it.")

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
        last_gripper_check_time = 0.0
        target_vel = {}
        stall_watchdog = GripperStallWatchdog()
        dt = 1.0 / args.loop_hz
        halted = False
        holding_for_reset = False
        tick = 0
        next_tick = time.perf_counter()
        stats = LoopStats()

        while not halted:
            loop_start = time.time()
            tick += 1

            if kill_switch.triggered:
                print("\nKill switch pressed. Ramping down and disabling motors.")
                halted = True
                break

            if receiver.shutdown:
                print("\nSim finished. Ramping down and disabling motors.")
                halted = True
                break

            if event_receiver is not None:
                for event in event_receiver.drain():
                    if event.get("event") == "status":
                        if viewer is not None:
                            viewer.set_status(event)
                    else:
                        recorder.handle_event(event)

            packet = receiver.latest()
            now = time.time()

            if packet is not None and packet[0] != last_seq:
                holding_for_reset = False
                last_seq, last_packet_time, sim_joints = packet
                desired = sim_joints_to_motor_action(sim_joints, calib)
                tick_dt = now - last_command_time
                max_delta = args.max_joint_speed * max(tick_dt, dt)
                gripper_max_delta = args.gripper_max_speed * max(tick_dt, dt)
                target_action = clamp_step(current_action, desired, max_delta, gripper_max_delta)
                stats.command(desired, target_action)
                target_vel = compute_target_velocity(current_action, target_action, tick_dt, args.max_joint_speed)
                robot.send_action(target_action, target_vel)
                current_action = target_action
                last_command_time = now
            else:
                staleness_ms = (now - last_packet_time) * 1000.0
                # During a sim reset the silence is announced (see LatestPacketReceiver.hold_ms), so
                # the arm holds its last position instead of being disabled and dropping.
                hold_ms = receiver.hold_ms
                if staleness_ms > max(args.timeout_ms, hold_ms):
                    print(f"\nNo packet for {staleness_ms:.0f}ms (> --timeout-ms"
                          f"{f' / announced hold {hold_ms:.0f}ms' if hold_ms else ''})."
                          " Ramping down and disabling.")
                    halted = True
                    break
                elif staleness_ms > args.stale_ms:
                    if not hold_ms:
                        logger.warning(f"Stale packet ({staleness_ms:.0f}ms) -- holding last position.")
                    elif not holding_for_reset:
                        print(f"[BRIDGE] Sim is resetting -- holding position (up to {hold_ms / 1000:.0f}s).")
                    holding_for_reset = bool(hold_ms)
                    target_vel =  {k: 0.0 for k in target_vel.keys()}
                    robot.send_action(current_action, target_vel)
                    last_command_time = now

            # One CAN read per tick serves the recorder, the feedback plot and the gripper stall check;
            # it is only made on ticks where one of them needs it.
            recording = recorder is not None and recorder.recording and tick % record_every == 0
            check_grippers = now - last_gripper_check_time >= GRIPPER_CHECK_PERIOD_S
            actual = None
            if recording or feedback_sock is not None or check_grippers:
                try:
                    actual = get_current_pos_action(robot)
                except RuntimeError as e:
                    logger.warning(f"could not read real position: {e}")

            if recording:
                recorder.record(actual, sim_joints)

            if check_grippers:
                last_gripper_check_time = now
                if actual is not None:
                    if args.print_gripper_widths:
                        _print_gripper_widths(sim_joints, actual, calib)
                    for side, prefix in (("left", "L"), ("right", "R")):
                        key = f"{prefix}J8.pos"
                        if stall_watchdog.check(side, current_action[key], actual[key]):
                            _recover_gripper(robot, side)

            if feedback_sock is not None and actual is not None:
                try:
                    sim_joints_fb = motor_action_to_sim_joints(actual, calib)
                    feedback_sock.sendto(
                        json.dumps({"t": time.time(), "joints": sim_joints_fb}).encode("utf-8"), feedback_addr
                    )
                except OSError:
                    pass  # best-effort only -- never let a networking hiccup break mirroring

            # Absolute schedule rather than "sleep whatever is left of this tick": that one drifts by
            # the sleep overshoot every tick, so a 30 fps recording would really be ~29.x fps while
            # its timestamps claim 30.
            stats.tick(time.time() - loop_start)
            next_tick += dt
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -dt:
                next_tick = time.perf_counter()  # fell a whole tick behind: don't burst to catch up
            stats.maybe_report(args.max_joint_speed)

        print("Ramping down to a safe hold before disabling...")
        safe_hold = get_current_pos_action(robot)
        ramp_to(robot, current_action, safe_hold, duration_s=1.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors.")
    finally:
        receiver.stop()
        if feedback_sock is not None:
            feedback_sock.close()
        if event_receiver is not None:
            event_receiver.stop()
        if viewer is not None:
            viewer.stop()
        if recorder is not None:
            # The sim side sends SIGINT when it exits; a second one landing mid-finalize would
            # leave the parquet footers unwritten and the dataset unreadable.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                # Anything still queued (e.g. the last demo's "save", sent just before the sim
                # exited) is applied before the dataset is finalized.
                for event in event_receiver.drain():
                    if event.get("event") != "status":
                        recorder.handle_event(event)
                recorder.close()
            except Exception:
                logger.exception("[REAL REC] error while finalizing the dataset")
        try:
            robot.disconnect()
        except Exception:
            logger.exception("Error during disconnect -- verify motors are physically de-energized.")
        print("Robot disconnected.")


if __name__ == "__main__":
    main()