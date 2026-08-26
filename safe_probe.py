#!/usr/bin/env python
"""Low-authority, single-joint probe/bring-up tool for one OpenArm side.

Why this exists: OpenArmFollower.connect() -> configure() calls enable_all()
with no safety net -- no neutral command before/after enabling, no
read-before-command, no gain ramp. On 2026-07-01 this produced a fast,
uncontrolled motion right at enable time: two position reads taken a
fraction of a second apart (one at script start, one right before the
deliberate ramp) differed by ~12.5 rad, before any deliberate command had
even been issued. This is consistent with a known category of MIT-mode
motor controller behavior: a motor can resume driving toward whatever
kp/kd/q/tau was last active in its control register the instant it is
re-enabled, even if that target is stale or came from an earlier/corrupted
session. A smaller step size in the *deliberate* command does not fix this,
because the dangerous motion happens before that command is ever sent.

This script narrows that window two ways:
  1. It sends an explicit zero-gain, zero-torque MIT command to every motor
     on this arm BEFORE calling enable_all() -- in case the firmware latches
     whatever command is queued and applies it the instant enable fires.
     (Unconfirmed whether the firmware accepts frames while disabled --
     harmless to attempt either way.)
  2. It sends the same zero-gain command again immediately AFTER
     enable_all() -- the more reliable layer, in case step 1 is ignored
     while disabled.

Only after both of those does it read positions, so what you see is the
real, torque-free position -- not a stale target. A single-joint move then
ramps BOTH the gain (0 up to a modest ceiling, well under the OpenArmFollower
default nominal gains) and the position together, so authority builds in
gradually even if something else is still wrong.

This does NOT fully eliminate risk: the few milliseconds inside the
enable_all() CAN transaction itself are outside any Python script's control.
Keep a hand near the E-stop/power switch every time you run this, and only
ever probe one side (--side) at a time.

Usage:
  python safe_probe.py --side left --joint 1 --step 0.02   # read + one move
  python safe_probe.py --side left                          # read-only, no move
"""

import argparse
import importlib.util
import os
import time

import openarm_can as oa

# Loaded by PATH, not by package import, deliberately. `from robots.umeow_openarm_follower.
# can_monitor import ...` would execute that package's __init__, which pulls in the whole
# follower, lerobot and pinocchio -- and this is the tool you reach for when the rig is already
# broken. It must not stop working because something further up the stack does.
def _load_can_monitor():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "robots", "umeow_openarm_follower", "can_monitor.py")
    try:
        spec = importlib.util.spec_from_file_location("safe_probe_can_monitor", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:  # noqa: BLE001 -- a missing tap must never block a safety probe
        print(f"[note] CAN feedback tap unavailable ({type(e).__name__}: {e});"
              " freshness of the readings below cannot be confirmed.")
        return None


can_monitor = _load_can_monitor()

ARM_JOINT_COUNT = 7
MOTOR_TYPES = [
    oa.MotorType.DM8009, oa.MotorType.DM8009,
    oa.MotorType.DM4340, oa.MotorType.DM4340,
    oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
]
SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
GRIPPER_MOTOR_TYPE = oa.MotorType.DM4310
GRIPPER_SEND_ID = 0x08
GRIPPER_RECV_ID = 0x18

DEFAULT_PORT_FOR_SIDE = {"left": "can1", "right": "can0"}

# A human-scale OpenArm joint should never legitimately approach the motor's full
# +/-12.5 rad encoder range (that range exists to cover multi-turn gripper gearing,
# not a single arm joint's operating envelope). The observed read glitch consistently
# decodes to a value near this extreme (~-12.4676, raw code ~85 of 65535). Two
# consecutive reads landing on the SAME glitched value is possible by chance (the
# glitch always produces roughly the same number), so consistency alone is not
# sufficient -- reject implausible arm-joint readings outright, regardless of
# whether they appear "stable" across reads.
PLAUSIBLE_ARM_JOINT_RANGE = 3.2  # rad, generous vs any real arm pose, well short of +/-12.5


def neutral_params(n: int):
    return [oa.MITParam(kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=0.0) for _ in range(n)]


def hold_neutral(arm) -> None:
    arm.get_arm().mit_control_all(neutral_params(ARM_JOINT_COUNT))
    arm.get_gripper().mit_control_all(neutral_params(1))


def safe_connect(arm, write_ctrl_mode: bool = False) -> None:
    arm.init_arm_motors(MOTOR_TYPES, SEND_IDS, RECV_IDS)
    arm.init_gripper_motor(GRIPPER_MOTOR_TYPE, GRIPPER_SEND_ID, GRIPPER_RECV_ID)
    if write_ctrl_mode:
        # OFF BY DEFAULT, and only opt-in-able so the experiment below stays reproducible --
        # this write BRICKS the motors it touches until they are power-cycled. Confirmed both
        # directions on 2026-08-26 with a single-joint probe at kp=15:
        #
        #   right J5, write skipped -> -0.0341 moved to +0.2447 (target +0.2659) -- healthy
        #   left  J5, write sent    -> +0.7582 moved to +0.7582 -- bit-identical, no motion
        #
        # A bricked motor still answers refresh_all() with fresh, plausible positions, so it
        # looks alive to every reader in this repo; it just never executes a MIT command again.
        # The state SURVIVES process exit, so one run of this script with the write enabled
        # silently poisons every later session on that arm -- that is what left the right
        # wrist (J5-J8, all the DM4310s) unable to reach the reset pose across days of
        # deploy_smolvla_pickup_jointspace.py runs, showing up as RJ7 flagged "stale read" in
        # reset_to_rest_pose.py's tracking plot. Nothing in the normal bring-up path repairs
        # it: OpenArmFollower.configure() passes the modes to init_arm_motors() instead of
        # writing CTRL_MODE, which neither breaks nor fixes a motor. Only power-cycling does.
        #
        # lerobot's DamiaoMotorsBus.configure_motors() (proven to work, used by replay.py)
        # never sends this either -- it only enables torque, relying on the motor already
        # being provisioned for MIT mode from a one-time setup. Re-writing CTRL_MODE at
        # runtime, every invocation, was this script's own addition and was never needed.
        print("[WARNING] --write-ctrl-mode: set_control_mode_all(MIT) is about to be written."
              " This has been measured to leave motors unresponsive to MIT commands until they"
              " are power-cycled. See safe_connect() in this file.", flush=True)
        arm.get_arm().set_control_mode_all(oa.ControlMode.MIT)
        arm.get_gripper().set_control_mode_all(oa.ControlMode.MIT)
    arm.set_callback_mode_all(oa.CallbackMode.STATE)

    hold_neutral(arm)  # best-effort pre-enable neutral -- see module docstring
    arm.enable_all()
    hold_neutral(arm)  # authoritative post-enable neutral


READ_DEADLINE_S = 0.008     # ceiling on how long one read waits for the eight answers


def read_positions(arm, prefix: str, recv_timeout_us: int = 50_000, tap=None) -> dict:
    """Read positions, WAITING for the answers rather than polling for them.

    recv_all()'s timeout argument does not produce a wait on this build: measured 2026-08-26, it
    returns in 0.04-0.16 ms whether it is passed 500 us or 200 000 us. So the old "8 rounds of
    50_000 us" was not 400 ms of patience, it was about a millisecond of spinning -- against
    motors that answer 0.13 ms (J1) to 0.87 ms (J8) after the request, in ascending CAN-id order.
    The joints that answer LAST were the ones that kept falling outside the window, which is the
    mechanism behind this file's own "unstable by 12.4676 rad" retries and behind the stale wrist
    readings in reset_to_rest_pose.py.

    With a tap open, the loop stops the moment every channel has answered, so a healthy read is
    faster than the old spin. Without one it simply waits out the deadline.
    """
    if tap is not None and tap.available:
        tap.mark_cycle()
    arm.refresh_all()
    deadline = time.perf_counter() + READ_DEADLINE_S
    while True:
        arm.recv_all(recv_timeout_us)
        if tap is not None and tap.available:
            tap.poll()
            if not tap.pending():
                break
        if time.perf_counter() >= deadline:
            break
        time.sleep(50e-6)
    pos = {f"{prefix}J{i + 1}.pos": m.get_position() for i, m in enumerate(arm.get_arm().get_motors())}
    pos[f"{prefix}J8.pos"] = arm.get_gripper().get_motor().get_position()
    return pos


def find_implausible_key(pos: dict) -> str | None:
    """Return the first arm-joint (not gripper) key sitting suspiciously near the +/-12.5 rad
    encoder extreme, or None if all arm-joint readings look plausible."""
    for k, v in pos.items():
        if k.endswith("8.pos"):
            continue  # gripper can legitimately sit near an extreme when fully open/closed
        if abs(v) > PLAUSIBLE_ARM_JOINT_RANGE:
            return k
    return None


def read_positions_stable(arm, prefix: str, recv_timeout_us: int = 50_000, agreement_tol: float = 0.01,
                           max_attempts: int = 8, tap=None) -> dict:
    """Retry until two CONSECUTIVE, individually-plausible reads agree.

    An intermittent single-motor glitch (~12.5 rad, consistent with a stale/never-updated
    Motor object for that one CAN response) has been observed to land on a different joint
    almost every call, even with a generous recv_all() timeout. Because the glitch tends to
    produce roughly the same implausible value each time, requiring only that two consecutive
    reads AGREE is not enough -- the same joint could coincidentally glitch identically twice
    in a row. Each candidate read is also checked against find_implausible_key() before it is
    allowed to count. Silently returning a possibly-glitched reading is not acceptable for
    anything safety-critical, so this raises rather than guessing if positions never settle.
    """
    prev = read_positions(arm, prefix, recv_timeout_us, tap)
    for attempt in range(1, max_attempts):
        cur = read_positions(arm, prefix, recv_timeout_us, tap)

        bad_key = find_implausible_key(cur)
        if bad_key is not None:
            print(f"  [retry {attempt}/{max_attempts - 1}] {bad_key}={cur[bad_key]:+.4f} rad is implausible"
                  f" for an arm joint (> {PLAUSIBLE_ARM_JOINT_RANGE} rad), re-reading...")
            prev = cur
            continue

        worst_key, worst_delta = None, 0.0
        for k in cur:
            delta = abs(cur[k] - prev[k])
            if delta > worst_delta:
                worst_key, worst_delta = k, delta
        if worst_delta <= agreement_tol:
            return cur
        print(f"  [retry {attempt}/{max_attempts - 1}] {worst_key} unstable by {worst_delta:.4f} rad, re-reading...")
        prev = cur
    raise RuntimeError(
        f"Position reads did not stabilize after {max_attempts} attempts. Refusing to report"
        " untrustworthy positions -- this points to a real communication reliability issue,"
        " not just a slow read."
    )


def gain_and_position_ramp(arm, motor_index: int, is_gripper: bool, q_start: float, q_target: float,
                            kp_max: float, kd_max: float, steps: int, dt: float) -> None:
    for i in range(1, steps + 1):
        alpha = i / steps
        param = oa.MITParam(kp=kp_max * alpha, kd=kd_max * alpha, q=q_start + alpha * (q_target - q_start), dq=0.0, tau=0.0)
        if is_gripper:
            arm.get_gripper().mit_control_all([param])
        else:
            params = neutral_params(ARM_JOINT_COUNT)
            params[motor_index] = param
            arm.get_arm().mit_control_all(params)
        time.sleep(dt)


def joint_motor(arm, motor_index: int, is_gripper: bool):
    """The Motor object for this probe's target. Re-fetch after every read: get_motors() hands
    back COPIES of the motor state, so a handle kept from before a read is a stale snapshot."""
    if is_gripper:
        return arm.get_gripper().get_motor()
    return arm.get_arm().get_motors()[motor_index]


def report_move(key, start, target, observed, motor, max_kp, step, fresh):
    """Say what happened, and when nothing happened, say which of the three reasons it was.

    "Observed after move: +0.7105" on a joint that started at +0.7105 is the least informative
    thing this script could print, because three different situations produce it and they need
    opposite responses:

      - the feedback never refreshed, so the number is retained state and says nothing at all;
      - the motor pushed as hard as it was allowed to and the load won;
      - the motor is not executing commands at all.

    The reported torque separates the last two, and the CAN tap separates the first. Note the
    ramp commands tau=0 and leaves every OTHER joint at kp=0, so on a joint that carries weight
    the ONLY thing opposing gravity is kp*(q_target - q), i.e. at most max_kp*step -- which on
    this rig is 0.75 N-m at the defaults, against a J4 elbow that needs about 1.9 N-m just to
    stay where it is. That is not a fault, it is the probe being asked for more authority than
    its safety ceiling allows.
    """
    moved = observed - start
    want = target - start
    print(f"\nObserved after move: {key} = {observed:+.4f}"
          f"  (started {start:+.4f}, target {target:+.4f}, moved {moved:+.4f} of {want:+.4f})")

    if fresh is False:
        print("  VERDICT: unknown -- this channel sent no state frames during the read, so the"
              " number above is retained state, not a measurement. Nothing can be concluded"
              " about whether the joint moved. Fix the feedback first.")
        return

    if abs(moved) >= 0.2 * abs(want):
        print("  VERDICT: the joint followed.")
        return

    torque = motor.get_torque()
    ceiling = max_kp * abs(step)
    print(f"  motor reported torque {torque:+.3f} N-m at the end of the ramp"
          f" (this command's ceiling is kp {max_kp:g} x {abs(step):g} rad = {ceiling:.2f} N-m)")
    if abs(torque) >= 0.5 * ceiling:
        print("  VERDICT: the motor pushed and the load won. It was producing most of the torque"
              " this command allows and the joint still did not move, so the command has less"
              " authority than the joint's gravity and friction load -- expected on J1-J4, which"
              " carry the arm's weight, because the ramp sends tau=0 (no gravity feedforward)."
              "\n    This says nothing bad about the motor.")
        # The measured torque is a LOWER bound on what this joint needs, not an estimate of it:
        # the motor hit the command's own ceiling, so all it proves is "more than this much".
        # Scaling the suggestion off that number lands just short every time and costs another
        # run -- the first version of this message suggested kp=30 for a joint measured at 1.9
        # N-m of gravity load, i.e. 1.5 N-m of authority against 1.9 N-m of load.
        print(f"    Authority here is kp x step. {abs(torque):.2f} N-m is only a FLOOR on what"
              f" this joint needs, so step up properly rather than by a little:"
              f" --max-kp {max_kp * 4:.0f} gives {max_kp * 4 * abs(step):.1f} N-m."
              " A larger --step raises the demand the same way and moves the joint further,"
              " so raise one or the other, not both."
              "\n    For a joint that carries weight, the honest fix is gravity feedforward"
              " rather than brute gain: that is what OpenArmFollower.send_action() adds (tau from"
              " pinocchio, kp=60 on J4) and why reset_to_rest_pose.py moves this joint at gains"
              " this probe would call unsafe.")
    else:
        print("  VERDICT: the motor did NOT push. Fresh feedback, near-zero torque, no motion:"
              " it is answering on the bus but not executing MIT commands. That is the signature"
              " of a motor left in a bad control mode -- see safe_connect()'s --write-ctrl-mode"
              " note; only a power cycle clears it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--port", type=str, default=None, help="Defaults to can3 for left, can2 for right.")
    parser.add_argument("--joint", type=int, default=None, choices=range(1, 9), help="1-7 = arm joint, 8 = gripper. Omit for a read-only probe.")
    parser.add_argument("--step", type=float, default=0.02, help="radians, applied directly -- NOT degrees.")
    parser.add_argument("--max-kp", type=float, default=15.0, help="gain ceiling for this probe (OpenArmFollower's nominal goes up to 200).")
    parser.add_argument("--max-kd", type=float, default=1.0)
    parser.add_argument("--ramp-steps", type=int, default=60)
    parser.add_argument("--ramp-dt", type=float, default=0.03)
    parser.add_argument(
        "--write-ctrl-mode", action="store_true",
        help="Send the runtime set_control_mode_all(MIT) write before enabling. OFF by default"
        " because it has been measured to leave the motors it touches unresponsive to MIT"
        " commands until they are power-cycled, and that state survives process exit (see"
        " safe_connect()). Only pass this to reproduce that failure deliberately.",
    )
    args = parser.parse_args()

    port = args.port or DEFAULT_PORT_FOR_SIDE[args.side]
    prefix = "L" if args.side == "left" else "R"
    arm = oa.OpenArm(port, True)
    connected = False

    tap = None
    if can_monitor is not None:
        channels = {rid: f"{prefix}J{i + 1}" for i, rid in enumerate(RECV_IDS)}
        channels[GRIPPER_RECV_ID] = f"{prefix}J8"
        tap = can_monitor.CanFeedbackMonitor(port, channels)

    try:
        safe_connect(arm, write_ctrl_mode=args.write_ctrl_mode)
        connected = True

        pos = read_positions_stable(arm, prefix, tap=tap)
        print(f"Torque-free positions on {args.side} arm ({port}) -- safe to trust, read after neutral hold:")
        for k, v in pos.items():
            print(f"  {k}: {v:+.4f}")

        if args.joint is None:
            print("\nNo --joint given. Read-only probe complete, exiting.")
            return

        key = f"{prefix}J{args.joint}.pos"
        current = pos[key]
        target = current + args.step
        duration = args.ramp_steps * args.ramp_dt
        print(f"\nAbout to move {key}: {current:+.4f} -> {target:+.4f} rad")
        print(f"Gain ramps 0 -> kp={args.max_kp}, kd={args.max_kd} over {duration:.2f}s (position ramps together with gain).")
        if input("Type YES to proceed: ").strip() != "YES":
            print("Not confirmed. Holding neutral and exiting without moving.")
            return

        is_gripper = args.joint == 8
        motor_index = args.joint - 1
        if tap is not None and tap.available:
            tap.take_counts()
        gain_and_position_ramp(arm, motor_index, is_gripper, current, target,
                                args.max_kp, args.max_kd, args.ramp_steps, args.ramp_dt)

        pos_after = read_positions_stable(arm, prefix, tap=tap)
        fresh = None
        if tap is not None and tap.available:
            fresh = tap.take_counts().get(f"{prefix}J{args.joint}", 0) > 0
        report_move(key, current, target, pos_after[key],
                    joint_motor(arm, motor_index, is_gripper),
                    args.max_kp, args.step, fresh)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Holding neutral and disabling...")
        if connected:
            try:
                hold_neutral(arm)
                time.sleep(0.1)
            except Exception:
                print("WARNING: failed to send neutral hold before disabling -- verify motor power manually.")
        try:
            arm.disable_all()
        except Exception:
            print("WARNING: disable_all() failed -- verify motor power manually / cut power at the source.")
        if tap is not None:
            tap.close()
        print("Done.")


if __name__ == "__main__":
    main()
