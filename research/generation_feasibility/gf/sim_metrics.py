"""Pilot metrics computed from SIMULATOR GROUND TRUTH of a rollout (can pose, grasp-frame poses, joints, jaw state).

Nothing here looks at images. The same code is exercised now on recorded GT episodes (local Mimic-annotated HDF5),
which are successful sim recordings: that validates the code path and the thresholds' behaviour on known successes, it
does not validate generated data.

Timeline convention of the recorded HDF5 (and of the pilot runner, which will write the same layout): can pose and
joints are POST-step of step t, EEF poses are PRE-step (obs) of step t; so the grasp frame matching state t is eef[t+1].
"""

from __future__ import annotations

import numpy as np

from . import config as C
from .sources import Episode


def _quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def can_T(ep: Episode) -> np.ndarray:
    T = np.repeat(np.eye(4)[None], ep.T, 0)
    for t in range(ep.T):
        T[t, :3, :3] = _quat_R(ep.can[t, 3:7])
        T[t, :3, 3] = ep.can[t, :3]
    return T


def aligned_eef(ep: Episode) -> dict:
    """grasp frame aligned with the post-step state of step t (eef[t+1]; the last step repeats)."""
    return {s: np.concatenate([ep.eef[s][1:], ep.eef[s][-1:]]) for s in "RL"}


def hand_on_object(Tg: np.ndarray, Tc: np.ndarray) -> np.ndarray:
    """openarm_task_modes.hand_on_object: cylinder test in the can frame (radial / axial)."""
    d = Tg[:, :3, 3] - Tc[:, :3, 3]
    loc = np.einsum("tji,tj->ti", Tc[:, :3, :3], d)          # R^T d
    return (np.hypot(loc[:, 0], loc[:, 1]) < C.EEF_TO_CAN_RADIAL) & (np.abs(loc[:, 2]) < C.GRASP_AXIAL_TOL)


def handover_stage_replica(ep: Episode) -> dict:
    """Offline replica of openarm_task_modes._handover_tick + handover_success (nominal can size and pad height)."""
    fps = ep.fps
    hold_steps = C.scale_steps(20, fps)
    Tc = can_T(ep)
    E = aligned_eef(ep)
    finger = {s: ep.grip_state[s] * C.FINGER_OPEN for s in "RL"}       # mean of the two finger joints, m
    closed = {s: np.abs(finger[s] - C.FINGER_OPEN) > C.GRIPPER_THRESHOLD for s in "RL"}
    on = {s: hand_on_object(E[s], Tc) for s in "RL"}
    holds = {s: on[s] & closed[s] for s in "RL"}
    aperture_L = 2 * finger["L"]
    firm = on["L"] & (aperture_L >= C.RECEIVER_APERTURE[0]) & (aperture_L <= C.RECEIVER_APERTURE[1])
    z = Tc[:, 2, 3]
    lifted_now = z > C.CAN_REST_Z + C.LIFT_MIN
    stage, was_lifted, rhsp, grip_steps = 0, False, False, 0
    stages, gsteps = [], []
    t_done = None
    for t in range(ep.T):
        was_lifted |= bool(holds["R"][t] and lifted_now[t])
        if stage == 0 and holds["R"][t]:
            stage = 1
        put_down = (not holds["R"][t]) and z[t] <= C.CAN_REST_Z + C.PUT_DOWN_MARGIN
        rhsp = (rhsp or bool(holds["R"][t])) and not put_down
        if stage == 1 and holds["L"][t] and rhsp and was_lifted:
            stage = 2
        carrying = stage >= 2 and firm[t] and not holds["R"][t] and not put_down
        grip_steps = grip_steps + 1 if carrying else 0
        if stage == 2 and grip_steps >= hold_steps:
            stage, t_done = 3, t
        stages.append(stage)
        gsteps.append(grip_steps)
    stages = np.array(stages)
    gsteps = np.array(gsteps)
    return {"stage": stages, "grip_steps": gsteps, "holds": holds, "firm": firm, "on": on, "closed": closed,
            "success_final": bool(stages[-1] >= 3 and gsteps[-1] >= hold_steps), "t_done": t_done, "Tc": Tc, "E": E}


def _first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else None


def rollout_metrics(ep: Episode, kin=None) -> dict:
    """All pilot metrics for one rollout (or recorded GT episode)."""
    fps = ep.fps
    r = handover_stage_replica(ep)
    Tc, E, holds = r["Tc"], r["E"], r["holds"]
    z = Tc[:, 2, 3]
    t_rh = _first(holds["R"])
    t_lift = _first(z > C.CAN_REST_Z + C.LIFT_MIN, t_rh or 0) if t_rh is not None else None
    t_pass = _first(r["stage"] >= 2)
    # giver release: first step after the pass where the right hand no longer holds
    t_rel = _first(~holds["R"], t_pass) if t_pass is not None else None
    out = {"success_final": r["success_final"], "t_right_hold": t_rh, "t_lift": t_lift, "t_pass": t_pass,
           "t_giver_release": t_rel, "t_done": r["t_done"]}
    # unsupported: neither hand holds the can (raw grasp tests; a flicker of <= 2 steps is tolerated)
    held_any = r["firm"] | holds["R"] | holds["L"]
    def first_run(mask, start, k):
        run = 0
        for t in range(start, len(mask)):
            run = run + 1 if mask[t] else 0
            if run >= k:
                return t - k + 1
        return None
    flick = C.scale_steps(2, fps)
    t_drop = first_run(~held_any, t_lift, flick + 1) if t_lift is not None else None
    out["t_drop"] = t_drop
    out["off_table"] = bool((z < C.PAD_TOP_Z - 0.03).any())
    # retention: firm receiver grip after the giver let go, until the end or the first loss longer than a flicker
    if t_rel is not None:
        loss = first_run(~r["firm"], t_rel, flick + 1)
        out["retention_steps"] = int((ep.T if loss is None else loss) - t_rel)
        out["retention_to_end"] = loss is None
    else:
        out["retention_steps"], out["retention_to_end"] = 0, False
    # informational: the receiver lowered the held can to within 5 mm of its rest height (operators do this)
    out["lowered_to_pad_while_held"] = bool(t_rel is not None and (z[t_rel:] <= C.CAN_REST_Z + C.PUT_DOWN_MARGIN).any())
    ret_need = C.scale_steps(C.POST_RELEASE_RETENTION * 20 / 30, fps)   # 30 steps at 30 Hz = 1 s
    out["stable_success"] = bool(out["success_final"] and out["retention_steps"] >= ret_need and out["retention_to_end"]
                                 and t_drop is None and not out["off_table"])
    # slip: drift of the can in the holding hand's frame
    def drift(side, a, b):
        if a is None or b is None or b - a < 3:
            return None
        X = np.einsum("tij,tjk->tik", np.linalg.inv(E[side][a:b]), Tc[a:b])
        return float(np.linalg.norm(X[:, :3, 3] - X[0, :3, 3], axis=1).max())
    settle = C.scale_steps(10, fps)
    out["slip_giver_m"] = drift("R", None if t_rh is None else t_rh + settle, t_pass)
    out["slip_receiver_m"] = drift("L", None if t_rel is None else t_rel + settle, ep.T)
    out["slip_flag"] = bool(max([v for v in (out["slip_giver_m"], out["slip_receiver_m"]) if v is not None] or [0])
                            > C.SLIP_TOL)
    # jolts (as the DATA_GENERATION_AUDIT heuristics)
    v = np.linalg.norm(np.diff(Tc[:, :3, 3], axis=0), axis=1) * fps
    tilt = np.degrees(np.arccos(np.clip(np.abs(Tc[:, 2, 2]), -1, 1)))
    out["can_speed_max"] = float(v.max()) if len(v) else 0.0
    out["tilt_max_deg"] = float(tilt.max())
    # joint-command validity vs the real-robot wrapper contracts
    gap = max(np.abs(ep.a[s] - ep.q[s]).max() for s in "RL")
    jump = max(np.abs(np.diff(ep.a[s], axis=0)).max() for s in "RL")
    out["max_request_state_gap"] = float(gap)
    out["max_request_jump"] = float(jump)
    out["contract_violations"] = int(gap > C.MAX_TARGET_JUMP * 2) + int(jump > C.MAX_TARGET_JUMP)
    # offline arm-arm clearance on the MEASURED joints (sim self-collision is off by default)
    if kin is not None:
        from .kin import arm_arm_clearance
        mins = [arm_arm_clearance(kin.capsules(ep.q["R"][t], ep.q["L"][t]))[0] for t in range(0, ep.T, 2)]
        out["arm_clearance_min_m"] = float(min(mins))
        out["arm_clearance_violation"] = bool(min(mins) < C.ARM_CLEARANCE)
    return out


def executed_descriptor(ep: Episode, m: dict) -> dict | None:
    """Hand-over descriptor from simulator truth (the only basis for a NOVELTY claim)."""
    from .descriptors import handover_descriptor

    if m["t_pass"] is None or m["t_giver_release"] is None:
        return None
    r = handover_stage_replica(ep)
    Tc, E = r["Tc"], r["E"]
    pL = E["L"][:, :3, 3]
    t_lc = int(np.flatnonzero(r["closed"]["L"][: m["t_pass"] + 1])[0]) if r["closed"]["L"][: m["t_pass"] + 1].any() else m["t_pass"]
    t_la = _first(np.linalg.norm(pL - pL[0], axis=1) > 0.02) or 0
    pR = E["R"][:, :3, 3]
    far = np.linalg.norm(pR[:t_lc + 1] - pR[t_lc], axis=1) > 0.01
    t_ga = int(np.flatnonzero(far)[-1] + 1) if far.any() else t_lc
    return handover_descriptor(E["R"], E["L"], Tc, m["t_pass"], t_ga, t_la, m["t_giver_release"], t_close=t_lc)
