"""v3.1 corrections to the v3 generator, frozen before the v3.1 regeneration (v3 code and results are untouched;
these are runtime replacements installed by `install()`).

1. G+ boundary states. v3's G+ used Gaussian tails before its first and after its last waypoint, so the first robot
   state moved by up to 0.10 rad (v3 review). v3.1 G+ uses compact smoothstep ramps from an identity anchor at
   max(0, t_first - 3 sigma) and back to identity at min(T-1, t_last + 3 sigma): q_G+(0) == q_source(0) == q_C(0).
   Waypoints, targets, scales and sources are unchanged.
2. Label validation split into
     HARD (gate):        exact per-joint construction a = q(t+lead) + r_src(tau) at every valid row, no finite
                         labels on unsupported rows, command chronology/latency of both grippers, finite values,
                         candidate hash;
     DIAGNOSTIC (never a gate): per-joint apparent lead (v3's 0.75-frame rule, kept and reported);
     PHYSICAL UNCERTAINTY: a per-joint tracking-residual model fitted on the SOURCE episode,
                         r_j(t) = a_j(t) - q_j(t+lead) ~ alpha + beta qdot_j(t+lead) + gamma qddot_j(t+lead),
                         is evaluated on the candidate's own motion; if, for any arm joint, the RMS discrepancy
                         between the transported residual and the model's prediction exceeds 2.0 x the source's
                         own model RMS (and 0.002 rad), the candidate is marked RESIDUAL_UNCERTAIN.
   Acceptance uses HARD only. RESIDUAL_UNCERTAIN candidates are kept but excluded from the 'evidence-complete'
   export subset, exactly like candidates whose receiver-window image alignment is UNKNOWN.
"""

from __future__ import annotations

import numpy as np

from sfa3 import labels as lb, planner3 as pl
from sfa2 import se3

UNCERTAIN_RATIO, UNCERTAIN_FLOOR = 2.0, 0.002
TAIL = 3.0 * pl.SIGMA


def waypoint_delta_bounded(wps, Tw, t):
    T = len(Tw)
    ds = [(tk, Tk @ se3.inv(Tw[tk])) for tk, Tk, _ in wps]
    a0 = max(0.0, ds[0][0] - TAIL)
    a1 = min(T - 1.0, ds[-1][0] + TAIL)
    if t <= ds[0][0]:
        u = 0.0 if ds[0][0] <= a0 else float(pl.ss((t - a0) / (ds[0][0] - a0)))
        return se3.power(ds[0][1], u) if t > a0 else np.eye(4)
    if t >= ds[-1][0]:
        u = 0.0 if a1 <= ds[-1][0] else float(pl.ss((a1 - t) / (a1 - ds[-1][0])))
        return se3.power(ds[-1][1], u) if t < a1 else np.eye(4)
    for (t0, D0), (t1, D1) in zip(ds[:-1], ds[1:]):
        if t0 <= t <= t1:
            u = float(pl.ss((t - t0) / max(1e-9, t1 - t0)))
            return se3.power(D1 @ se3.inv(D0), u) @ D0


def residual_model(state, action, lead, valid=None):
    """per-joint least-squares model of the command-state residual; returns coefficients and RMS."""
    T = len(state)
    valid = np.ones(T, bool) if valid is None else valid
    out = {}
    for name in lb.ARM:
        j = lb.rb.NAMES.index(name)
        q = state[:, j]
        qd = np.gradient(q)
        qdd = np.gradient(qd)
        t = np.arange(T - lead)
        t = t[valid[t] & np.isfinite(action[t, j])]
        r = action[t, j] - q[t + lead]
        Xm = np.stack([np.ones(len(t)), qd[t + lead], qdd[t + lead]], 1)
        coef, *_ = np.linalg.lstsq(Xm, r, rcond=None)
        out[name] = {"coef": coef, "rms": float(np.sqrt(np.mean((r - Xm @ coef) ** 2)))}
    return out


def residual_check(rec, c, model=None) -> dict:
    model = model or residual_model(rec.state, rec.action, rec.lead)
    T, L = len(c["state"]), rec.lead
    valid = np.asarray(c["valid"], bool) & (np.arange(T) <= T - 1 - L)
    worst, ratios = None, {}
    for name in lb.ARM:
        j = lb.rb.NAMES.index(name)
        q = c["state"][:, j]
        qd, qdd = np.gradient(q), None
        qdd = np.gradient(qd)
        t = np.flatnonzero(valid)
        r = c["action"][t, j] - q[t + L]
        m = model[name]["coef"] @ np.stack([np.ones(len(t)), qd[t + L], qdd[t + L]])
        rms = float(np.sqrt(np.mean((r - m) ** 2)))
        ratios[name] = rms / max(model[name]["rms"], 1e-9)
        if rms > UNCERTAIN_FLOOR and ratios[name] > UNCERTAIN_RATIO and (worst is None or ratios[name] > ratios[worst]):
            worst = name
    return {"residual_uncertain": worst is not None, "worst_joint": worst, "max_ratio": float(max(ratios.values())),
            "ratios": {k: round(v, 3) for k, v in ratios.items()}}


def validate_v31(rec, c, src_leads=None, src_chron=None) -> dict:
    con = lb.construction(rec, c)
    gr = lb.gripper_check(rec, c, src_chron)
    finite_ok = bool(np.isfinite(c["state"]).all() and np.isfinite(c["action"][np.asarray(c["valid"], bool)]).all())
    out = {"construction": con, "gripper": gr, "finite": finite_ok, "hash": lb.candidate_hash(c),
           "per_joint_lead": lb.per_joint_check(rec, c, src_leads),        # diagnostic only
           "residual": residual_check(rec, c)}
    out["ok"] = bool(con["ok"] and gr["ok"] and finite_ok)
    out["lead_diagnostic_flag"] = not out["per_joint_lead"]["ok"]
    out["residual_uncertain"] = out["residual"]["residual_uncertain"]
    return out


def install():
    """swap the two v3 functions at runtime (the v3 source files are not edited)."""
    pl.waypoint_delta = waypoint_delta_bounded
    lb.validate = validate_v31
    return {"waypoint_delta": "bounded", "labels.validate": "v3.1 hard/diagnostic split"}
