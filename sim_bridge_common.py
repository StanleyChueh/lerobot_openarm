#!/usr/bin/env python
"""Shared calibration/mapping/safety helpers for bridging Isaac Sim OpenArm joint
data (live via mirror_bridge.py, or from a recorded file via replay_sim_dataset.py)
onto the real dual-arm OpenArm follower.

Kept in one place so both bridges apply IDENTICAL calibration mapping, clamping,
and ramping logic -- see mirror_bridge.py's module docstring for the required
Phase 0 bench verification before either script is used, and calibration.example.json
for the calibration file schema.
"""

import json
import sys
import threading
import time

ARM_JOINT_KEYS = [f"joint{i}" for i in range(1, 8)]  # openarm_{side}_joint1..7
GRIPPER_SIM_OPEN = 0.044  # radians, matches BinaryJointPositionActionCfg open_command_expr in sim
GRIPPER_SIM_CLOSED = 0.0


def load_calibration(path: str) -> dict:
    with open(path) as f:
        calib = json.load(f)

    for side in ("left", "right"):
        if side not in calib:
            raise ValueError(f"calibration.json missing '{side}' section")
        sec = calib[side]
        sec.setdefault("offset_rad", {j: 0.0 for j in ARM_JOINT_KEYS})
        for j in ARM_JOINT_KEYS:
            if j not in sec.get("sign", {}):
                raise ValueError(f"calibration.json[{side}].sign missing '{j}'")
            if sec["sign"][j] not in (1, -1):
                raise ValueError(f"calibration.json[{side}].sign['{j}'] must be 1 or -1 (got {sec['sign'][j]!r})")
            offset = sec["offset_rad"].get(j)
            if not isinstance(offset, (int, float)):
                raise ValueError(f"calibration.json[{side}].offset_rad['{j}'] must be a number (got {offset!r})")
        grip = sec.get("gripper", {})
        for key in ("open_raw", "closed_raw"):
            if not isinstance(grip.get(key), (int, float)):
                raise ValueError(f"calibration.json[{side}].gripper['{key}'] must be a number (got {grip.get(key)!r})")
    return calib


ARMS_CROSSED_MARGIN = 0.15  # rad of slack past a gripper's calibrated end before it counts as
# "outside its own range". The grippers legitimately sit a little beyond open_raw when pressed
# against the mechanical stop (-1.3136 read against an open_raw of -1.2980, measured 2026-08-26).


def check_arms_not_crossed(robot, calib: dict, margin: float = ARMS_CROSSED_MARGIN) -> None:
    """Raise if the two arms' CAN cables are swapped relative to right_port/left_port.

    The grippers are the ONLY channel that can detect this. Every arm joint has sign=+1 and
    offset=0.0 on both sides of calibration.json, so a crossed pair produces perfectly plausible
    arm-joint readings and nothing anywhere notices. The grippers do not: the two motors turn
    opposite ways to close, so their calibrated travel has OPPOSITE SIGN -- left runs
    0.0 (closed) .. +1.2121 (open), right runs 0.0 (closed) .. -1.2980 (open). An arm reading the
    other side's open position is therefore unambiguous.

    Worth failing hard over, because "crossed" is not a cosmetic mix-up. sim_init_pose_action()
    commands each gripper to its OWN side's open_raw, so under a crossed pair each gripper is sent
    a target ~1.2 rad past its own CLOSED stop and stalls there for the whole run. Measured
    2026-08-26: both grippers held 3.3 N-m against their stops from the first tick, and the four
    DM4310s (J5-J8) on one arm dropped off the CAN bus partway through every such run while the
    same hardware was clean on a correctly-wired run minutes earlier.

    Conservative by construction: it fires only when BOTH grippers read outside their own range
    AND inside the other's. The two ranges meet at 0.0, so two closed grippers are ambiguous and
    are never flagged; one odd reading is never enough on its own.
    """
    def span(side):
        grip = calib[side]["gripper"]
        lo, hi = sorted((grip["closed_raw"], grip["open_raw"]))
        return lo, hi

    def inside(value, bounds):
        lo, hi = bounds
        return lo - margin <= value <= hi + margin

    pos = get_current_pos_action(robot)
    readings = {"left": pos["LJ8.pos"], "right": pos["RJ8.pos"]}
    spans = {"left": span("left"), "right": span("right")}
    other = {"left": "right", "right": "left"}

    crossed = all(
        not inside(readings[side], spans[side]) and inside(readings[side], spans[other[side]])
        for side in ("left", "right")
    )
    if not crossed:
        return

    detail = "\n".join(
        f"  {'L' if side == 'left' else 'R'}J8.pos reads {readings[side]:+.4f}, which is outside"
        f" the {side} gripper's own range [{spans[side][0]:+.4f}, {spans[side][1]:+.4f}]"
        f" and inside the {other[side]} one's"
        f" [{spans[other[side]][0]:+.4f}, {spans[other[side]][1]:+.4f}]"
        for side in ("left", "right")
    )
    raise RuntimeError(
        "The two arms' CAN cables are SWAPPED -- each side is reading the other gripper's"
        f" calibrated open position:\n{detail}\n"
        "  Swap the two CAN cables at the adapter (or swap --right-port/--left-port) and re-run."
        " Refusing to move: every gripper command would be sent to the wrong arm, ~1.2 rad past"
        " its own closed stop, and both grippers would stall against their stops for the whole"
        " run."
    )


def gripper_sim_to_raw(sim_val: float, open_raw: float, closed_raw: float) -> float:
    frac = max(0.0, min(1.0, (sim_val - GRIPPER_SIM_CLOSED) / (GRIPPER_SIM_OPEN - GRIPPER_SIM_CLOSED)))
    return closed_raw + frac * (open_raw - closed_raw)


def gripper_cmd_to_raw(cmd: float, open_raw: float, closed_raw: float) -> float:
    """Map a +-1 gripper command (e.g. record_demos_openarm.py's binary gripper action,
    +1=open/-1=closed) directly to raw motor units -- for datasets that record a gripper
    command rather than a physical joint angle (see replay_hf_sim_episode.py)."""
    frac = max(0.0, min(1.0, (cmd + 1.0) / 2.0))
    return closed_raw + frac * (open_raw - closed_raw)


def sim_joints_to_motor_action(sim_joints: dict, calib: dict) -> dict:
    """Map {"openarm_left_joint1": rad, ...} -> full 16-key {"LJ1.pos": rad, ...} action dict."""
    action = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        sec = calib[side]
        for n, jkey in enumerate(ARM_JOINT_KEYS, start=1):
            sim_name = f"openarm_{side}_joint{n}"
            if sim_name not in sim_joints:
                raise KeyError(f"expected '{sim_name}' in incoming packet, got keys {list(sim_joints)}")
            sign = sec["sign"][jkey]
            offset = sec["offset_rad"][jkey]
            action[f"{prefix}J{n}.pos"] = sign * sim_joints[sim_name] + offset

        finger_vals = [v for k, v in sim_joints.items() if k.startswith(f"openarm_{side}_finger_joint")]
        if not finger_vals:
            raise KeyError(f"expected an 'openarm_{side}_finger_joint*' key in incoming packet")
        grip = sec["gripper"]
        action[f"{prefix}J8.pos"] = gripper_sim_to_raw(finger_vals[0], grip["open_raw"], grip["closed_raw"])
    return action


def raw_to_gripper_sim(raw: float, open_raw: float, closed_raw: float) -> float:
    """Inverse of gripper_sim_to_raw() -- maps a real raw gripper reading back to the
    0 (closed) .. GRIPPER_SIM_OPEN (open) sim finger-joint range."""
    frac = max(0.0, min(1.0, (raw - closed_raw) / (open_raw - closed_raw)))
    return GRIPPER_SIM_CLOSED + frac * (GRIPPER_SIM_OPEN - GRIPPER_SIM_CLOSED)


def motor_action_to_sim_joints(action: dict, calib: dict) -> dict:
    """Inverse of sim_joints_to_motor_action() -- maps a real motor action dict
    {"LJ1.pos": rad, ...} back to sim joint names {"openarm_left_joint1": rad, ...},
    matching JointMirrorBroadcaster's naming on the Isaac Sim side (including both
    finger joints per gripper). Used to send real-robot feedback back to a sim-side
    process for a sim-vs-real comparison plot (see mirror_bridge.py --feedback-port
    and record_demos_openarm.py --mirror_feedback_port)."""
    sim_joints = {}
    for side, prefix in (("left", "L"), ("right", "R")):
        sec = calib[side]
        for n, jkey in enumerate(ARM_JOINT_KEYS, start=1):
            sign = sec["sign"][jkey]
            offset = sec["offset_rad"][jkey]
            sim_joints[f"openarm_{side}_joint{n}"] = (action[f"{prefix}J{n}.pos"] - offset) / sign
        grip = sec["gripper"]
        grip_val = raw_to_gripper_sim(action[f"{prefix}J8.pos"], grip["open_raw"], grip["closed_raw"])
        sim_joints[f"openarm_{side}_finger_joint1"] = grip_val
        sim_joints[f"openarm_{side}_finger_joint2"] = grip_val
    return sim_joints


def clamp_step(current: dict, target: dict, max_delta: float, gripper_max_delta: float | None = None) -> dict:
    """gripper_max_delta, if given, applies instead of max_delta to keys ending in "8.pos".
    Gripper commands in recorded datasets are typically a near-instant open/closed toggle,
    not a smooth trajectory like the arm joints -- clamping it to the same conservative
    per-tick speed as the arm makes it lag up to the gripper's full range behind schedule
    every time the episode calls for an open/close change."""
    out = {}
    for k, tgt in target.items():
        cur = current[k]
        k_max_delta = gripper_max_delta if (gripper_max_delta is not None and k.endswith("8.pos")) else max_delta
        delta = max(-k_max_delta, min(k_max_delta, tgt - cur))
        out[k] = cur + delta
    return out
def compute_target_velocity(
    current: dict[str, float], 
    clamped_target: dict[str, float], 
    dt: float,
    max_velocity: float | dict[str, float] | None = None
) -> dict[str, float]:
    """Compute the target velocity for each joint based on the current position, clamped target position, and time step.
    Args:
        current: Current joint positions.
        clamped_target: Clamped target joint positions.
        dt: Time step in seconds.
        max_velocity: Maximum allowed velocity for each joint. Can be a single float or a dictionary mapping joint names to their respective maximum velocities.    
    """
    target_vel = {}
    for k, tgt_pos in clamped_target.items():
        cur_pos = current[k]
        raw_vel = (tgt_pos - cur_pos) / dt
        
        if max_velocity is not None:
            if isinstance(max_velocity, dict):
                v_limit = max_velocity.get(k, float("inf"))
            else:
                v_limit = max_velocity
            vel = max(-v_limit, min(v_limit, raw_vel))
        else:
            vel = raw_vel

        vel_key = k.replace(".pos", ".vel") if k.endswith(".pos") else f"{k}_vel"
        target_vel[vel_key] = vel

    return target_vel

def ramp_to(robot, start_action: dict, end_action: dict, duration_s: float, rate_hz: float = 50.0,
            on_tick=None):
    """Linearly interpolate from start_action to end_action over duration_s.

    on_tick, if given, is called as on_tick(elapsed_s, commanded_action, step_index, steps)
    after each send_action(), for callers that want to log or plot what the ramp did.
    Anything it does costs wall-clock inside the tick budget -- notably reading the robot
    back, which on this CAN stack is tens of milliseconds (see _read_motor_positions_once).
    That overrun makes the ramp take LONGER than duration_s, i.e. it moves the arm more
    slowly than planned, never faster; the interpolation is indexed by step, not by clock.
    So a hook is safe to add, but its own timestamps -- not `duration_s` -- are what any
    resulting plot's time axis should use. The send_action() slow-call warning below is
    measured around send_action() alone so the hook's cost can't be misattributed to it.
    """
    steps = max(1, int(duration_s * rate_hz))
    dt = 1.0 / rate_hz
    # The MIT command's dq is a velocity SETPOINT, and this used to send 0 for every joint on
    # every tick while simultaneously commanding a position that was moving. The motor-side law is
    # tau = kp*(q_des - q) + kd*(dq_des - dq) + tau_ff, so dq_des = 0 during a ramp turns the kd
    # term into a brake proportional to the joint's actual speed, and the loop can only settle
    # where kp*err balances it: a standing lag of kd*v/kp rad for the whole ramp, on top of
    # whatever friction costs. At the arm's default 0.3 rad/s cap that is ~15 mrad on J5/J7
    # (kd=1.0, kp=20) and ~13 mrad on J4 (kd=2.5, kp=60) -- present on every joint, in the
    # direction of travel, and entirely self-inflicted since the ramp knows its own velocity.
    # A hold is start_action == end_action, which makes every entry exactly 0.0 as before.
    target_vel = {
        f"{k[:-4]}.vel": (end_action[k] - start_action[k]) / duration_s
        for k in start_action if k.endswith(".pos") and k in end_action
    } if duration_s > 0 else {}
    t0 = time.time()
    print(f"  [ramp_to] entering loop, about to call send_action() for step 1/{steps}...", flush=True)
    for i in range(1, steps + 1):
        step_start = time.time()
        alpha = i / steps
        cmd = {k: start_action[k] + alpha * (end_action[k] - start_action[k]) for k in start_action}
        robot.send_action(cmd, target_vel)
        call_dt = time.time() - step_start
        if call_dt > dt * 3:
            # send_action() itself took much longer than one tick -- likely a slow/stalled
            # CAN response, not a frozen script. Surface it instead of silently absorbing it.
            print(f"  [ramp_to step {i}/{steps}] send_action() took {call_dt * 1000:.0f}ms (expected ~{dt * 1000:.0f}ms)", flush=True)
        if on_tick is not None:
            on_tick(time.time() - t0, cmd, i, steps)
        remaining = dt - (time.time() - step_start)
        if remaining > 0:
            time.sleep(remaining)
    return end_action


PLAUSIBLE_JOINT_RANGE = 3.2  # rad; matches safe_probe.py's constant. Values beyond this
# (typically landing near +-12.5, the raw encoder extreme) are a known intermittent
# read glitch, not a real position -- see safe_probe.py's module docstring.


def get_current_pos_action(robot, max_attempts: int = 8) -> dict:
    """Read the robot's current position state, retrying on the known implausible-read
    glitch. Unlike safe_probe.py's read_positions_stable(), this also checks gripper
    channels: this robot's calibrated gripper range (see calibration.json) never
    approaches +-3.2 rad, so treating gripper readings as always-plausible would let
    the exact same glitch slip through here, e.g. into a startup-handshake hold target."""
    for attempt in range(1, max_attempts + 1):
        obs = dict(robot.get_observation())
        pos = {k: v for k, v in obs.items() if k.endswith(".pos")}
        bad = {k: v for k, v in pos.items() if abs(v) > PLAUSIBLE_JOINT_RANGE}
        if not bad:
            return pos
        print(f"  [get_current_pos_action retry {attempt}/{max_attempts}] implausible reading(s): {bad} -- re-reading...")
    raise RuntimeError(
        f"Position reads did not stabilize to plausible values after {max_attempts} attempts."
        " Refusing to trust an implausible reading for a startup handshake or hold target."
    )


def run_startup_handshake(
    robot, target_action: dict, tolerance: float, gripper_tolerance: float | None = None
) -> bool:
    """Print a real-vs-sim comparison and return True iff within tolerance.

    gripper_tolerance, if given, applies instead of `tolerance` to keys ending in
    "8.pos" (the gripper channels). Gripper open/closed state legitimately varies
    between episodes/tasks as part of the task setup -- it isn't a safety-relevant
    mismatch the way an arm joint being way off would be (which usually means bad
    calibration or a real hazard), so gating the whole handshake on it with the same
    strict tolerance is overly strict and blocks perfectly safe replays.
    """
    current_action = get_current_pos_action(robot)
    print("\nStartup handshake -- comparing real pose to sim pose:")
    worst_key, worst_delta, worst_ratio = None, 0.0, 0.0
    for k in target_action:
        delta = abs(target_action[k] - current_action[k])
        k_tolerance = gripper_tolerance if (gripper_tolerance is not None and k.endswith("8.pos")) else tolerance
        ratio = delta / k_tolerance
        flag = "  <-- exceeds its tolerance" if ratio > 1.0 else ""
        print(f"  {k:10s} real={current_action[k]:+.4f}  sim-mapped={target_action[k]:+.4f}  delta={delta:.4f}{flag}")
        if ratio > worst_ratio:
            worst_key, worst_delta, worst_ratio = k, delta, ratio

    if worst_ratio > 1.0:
        print(
            f"\nABORT: {worst_key} differs by {worst_delta:.4f} rad, exceeding its tolerance."
            " Move the real arm to match the sim's current pose, or re-check Phase 0"
            " calibration. Refusing to move the arm."
        )
        return False
    print(f"\nAll channels within tolerance (worst: {worst_key} at {worst_ratio:.0%} of its limit).")
    return True


class StdinKillSwitch:
    """Type 'q' + Enter at any time to trigger an immediate ramp-down and stop."""

    def __init__(self):
        self._triggered = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for line in sys.stdin:
            if line.strip().lower() == "q":
                self._triggered.set()
                return

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()


# Isaac Sim's own reset pose for this robot, from OPENARM_BI_CFG's init_state.joint_pos in
# IsaacLab's source/isaaclab_assets/isaaclab_assets/robots/openarm.py: every arm joint at 0.0
# EXCEPT joint4 at pi/2, both grippers fully open. This is the pose env.reset() puts the sim robot
# in, so it is the pose every recorded demo starts from and therefore the one a policy expects to
# be looking at on step 0 of a rollout. NOT all-zeros -- joint4 is the elbow, and starting a
# rollout with it straight instead of bent is a different task from the one the policy learned.
SIM_INIT_ARM_JOINT_RAD = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 1.570796,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.0,
}


def sim_init_pose_action(calib: dict) -> dict:
    """The full 16-key motor action for Isaac Sim's reset pose (see SIM_INIT_ARM_JOINT_RAD),
    mapped through calibration exactly the way a live sim packet would be."""
    sim_joints = {}
    for side in ("left", "right"):
        for n in range(1, 8):
            sim_joints[f"openarm_{side}_joint{n}"] = SIM_INIT_ARM_JOINT_RAD[f"joint{n}"]
        sim_joints[f"openarm_{side}_finger_joint1"] = GRIPPER_SIM_OPEN
    return sim_joints_to_motor_action(sim_joints, calib)


def approach_pose(
    robot,
    target_action: dict,
    *,
    label: str = "the target pose",
    arm_speed: float = 0.3,
    gripper_speed: float = 1.5,
    max_delta: float = 1.8,
    settled_tolerance: float = 0.05,
    min_duration: float = 1.5,
    assume_yes: bool = False,
    rate_hz: float = 50.0,
    on_tick=None,
) -> dict | None:
    """Drive the arm to `target_action` along a speed-limited ramp, and return the action it ended
    up commanded to (or None if the move was refused or not confirmed).

    This exists to replace "compare, then abort and tell the human to move the arm by hand", which
    run_startup_handshake() does and which is unworkable as a precondition for an autonomous
    rollout. Two reasons it cannot be satisfied by hand:

      - The residual it measures is largely steady-state droop, not drift. Ramping to the rest pose
        and re-reading still leaves joints tens of milliradians off (observed 2026-08-19: RJ2 at
        0.118 rad immediately after a clean 4 s ramp to that exact target), because these joints
        settle wherever gravity and their gains balance. No amount of repositioning by hand fixes a
        number the arm reproduces every time it holds a pose.
      - Every rollout needs the SAME start pose, so the approach has to happen before each episode,
        not once per session with a human in the loop.

    What is actually worth gating is the case the handshake was really protecting against: a target
    so far from the current pose that the calibration is probably wrong and moving would be
    dangerous. That is `max_delta`, and it still refuses outright.

    The velocity bound comes from the ramp duration rather than from a per-tick clamp: ramp_to()
    interpolates linearly from the current pose to the target over `duration`, so choosing
    duration = worst_delta / arm_speed makes every joint move at or below `arm_speed` rad/s by
    construction, with the longest-travelling joint setting the pace and the rest arriving together.

    Grippers get their own `gripper_speed` for the reason clamp_step() gives them their own delta
    cap: an open/close is a near-instant toggle, and pacing it like an arm joint would stretch the
    whole approach to the gripper's full travel time for no benefit.

    `on_tick` is forwarded verbatim to ramp_to() -- see there for what it costs.
    """
    current = get_current_pos_action(robot)
    missing = [k for k in target_action if k not in current]
    if missing:
        raise KeyError(f"target_action has keys the robot does not report: {missing}")

    deltas = {k: target_action[k] - current[k] for k in target_action}
    arm_keys = [k for k in target_action if not k.endswith("8.pos")]
    grip_keys = [k for k in target_action if k.endswith("8.pos")]
    worst_arm_key = max(arm_keys, key=lambda k: abs(deltas[k]))
    worst_arm = abs(deltas[worst_arm_key])
    worst_grip = max((abs(deltas[k]) for k in grip_keys), default=0.0)

    print(f"\nApproach to {label} -- current vs target:")
    for k in target_action:
        flag = "  <-- furthest" if k == worst_arm_key else ""
        print(f"  {k:10s} real={current[k]:+.4f}  target={target_action[k]:+.4f}"
              f"  delta={deltas[k]:+.4f}{flag}")

    if worst_arm > max_delta:
        print(
            f"\nREFUSED: {worst_arm_key} would have to travel {worst_arm:.3f} rad, beyond the"
            f" {max_delta:.2f} rad this approach is willing to move in one go. A gap that large"
            " usually means the calibration or the zeroing is wrong rather than that the arm"
            " drifted -- moving on that assumption is exactly what should not happen"
            " automatically. Check Phase 0 calibration, or move the arm closer by hand first."
        )
        return None

    if worst_arm <= settled_tolerance and worst_grip <= settled_tolerance:
        print(f"\nAlready at {label} (worst joint {worst_arm_key} at {worst_arm:.4f} rad)."
              " No approach motion needed.")
        return current

    duration = max(min_duration, worst_arm / arm_speed, worst_grip / gripper_speed)
    print(
        f"\nPlanned approach: {worst_arm:.3f} rad on {worst_arm_key} over {duration:.1f}s"
        f" (<= {arm_speed:g} rad/s per joint; grippers <= {gripper_speed:g} rad/s)."
    )

    if not assume_yes:
        confirm = input(f"Type YES to move the real arm to {label}: ")
        if confirm.strip() != "YES":
            print("Not confirmed. Aborting without moving the arm.")
            return None

    ramp_to(robot, current, target_action, duration, rate_hz, on_tick=on_tick)

    # Report where it actually landed rather than assuming the command was reached. A residual of
    # a few tens of milliradians here is normal steady-state error, not a failure -- see above --
    # so this prints it instead of gating on it.
    try:
        settled = get_current_pos_action(robot)
        worst_key = max(target_action, key=lambda k: abs(target_action[k] - settled[k]))
        worst = abs(target_action[worst_key] - settled[worst_key])
        print(f"Approach complete. Worst residual: {worst_key} at {worst:.4f} rad"
              " (steady-state error at this hold; expected, not a fault).")
    except RuntimeError as e:
        print(f"Approach complete, but the settle-check read did not stabilize: {e}")

    return dict(target_action)
