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


def ramp_to(robot, start_action: dict, end_action: dict, duration_s: float, rate_hz: float = 50.0):
    steps = max(1, int(duration_s * rate_hz))
    dt = 1.0 / rate_hz
    print(f"  [ramp_to] entering loop, about to call send_action() for step 1/{steps}...", flush=True)
    for i in range(1, steps + 1):
        step_start = time.time()
        alpha = i / steps
        cmd = {k: start_action[k] + alpha * (end_action[k] - start_action[k]) for k in start_action}
        robot.send_action(cmd)
        call_dt = time.time() - step_start
        if call_dt > dt * 3:
            # send_action() itself took much longer than one tick -- likely a slow/stalled
            # CAN response, not a frozen script. Surface it instead of silently absorbing it.
            print(f"  [ramp_to step {i}/{steps}] send_action() took {call_dt * 1000:.0f}ms (expected ~{dt * 1000:.0f}ms)", flush=True)
        remaining = dt - call_dt
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
