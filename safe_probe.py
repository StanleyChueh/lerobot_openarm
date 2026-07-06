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
import time

import openarm_can as oa

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

DEFAULT_PORT_FOR_SIDE = {"left": "can3", "right": "can2"}

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


def safe_connect(arm, skip_ctrl_mode: bool = False) -> None:
    arm.init_arm_motors(MOTOR_TYPES, SEND_IDS, RECV_IDS)
    arm.init_gripper_motor(GRIPPER_MOTOR_TYPE, GRIPPER_SEND_ID, GRIPPER_RECV_ID)
    if not skip_ctrl_mode:
        # lerobot's DamiaoMotorsBus.configure_motors() (proven to work, used by replay.py)
        # never sends this -- it only enables torque, relying on the motor already being
        # provisioned for MIT mode from a one-time setup. Re-writing CTRL_MODE at runtime,
        # every invocation, is this script's own addition and is suspected of leaving
        # joints 5-7 in a bad state. --skip-ctrl-mode tests that theory directly.
        arm.get_arm().set_control_mode_all(oa.ControlMode.MIT)
        arm.get_gripper().set_control_mode_all(oa.ControlMode.MIT)
    arm.set_callback_mode_all(oa.CallbackMode.STATE)

    hold_neutral(arm)  # best-effort pre-enable neutral -- see module docstring
    arm.enable_all()
    hold_neutral(arm)  # authoritative post-enable neutral


def read_positions(arm, prefix: str, recv_timeout_us: int = 50_000) -> dict:
    """Read positions with a generous per-call recv timeout.

    OpenArm::recv_all(int timeout_us = 500) takes MICROSECONDS, not
    milliseconds -- the codebase's own "recv_all(2500)" calls elsewhere
    (e.g. just_enable.py) are really only waiting 2.5ms, apparently just
    barely enough for the 7 arm joints but not reliably enough for the
    gripper. 50_000us = 50ms is still effectively instant to a human and
    costs nothing -- this is a read-only operation with zero motion risk
    regardless of how long it waits.
    """
    arm.refresh_all()
    for _ in range(8):
        arm.recv_all(recv_timeout_us)
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
                           max_attempts: int = 8) -> dict:
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
    prev = read_positions(arm, prefix, recv_timeout_us)
    for attempt in range(1, max_attempts):
        cur = read_positions(arm, prefix, recv_timeout_us)

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
        "--skip-ctrl-mode", action="store_true",
        help="Skip the runtime set_control_mode_all(MIT) write -- test whether that write itself"
        " is what's leaving certain joints unresponsive (see safe_connect() comment).",
    )
    args = parser.parse_args()

    port = args.port or DEFAULT_PORT_FOR_SIDE[args.side]
    prefix = "L" if args.side == "left" else "R"
    arm = oa.OpenArm(port, True)
    connected = False

    try:
        safe_connect(arm, skip_ctrl_mode=args.skip_ctrl_mode)
        connected = True

        pos = read_positions_stable(arm, prefix)
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
        gain_and_position_ramp(arm, motor_index, is_gripper, current, target,
                                args.max_kp, args.max_kd, args.ramp_steps, args.ramp_dt)

        pos_after = read_positions_stable(arm, prefix)
        print(f"\nObserved after move: {key} = {pos_after[key]:+.4f}")

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
        print("Done.")


if __name__ == "__main__":
    main()
