"""Action-label validation for v3 (closes the v2 review's false negatives). No dynamics are validated: these tests
certify that the exported labels are the declared construction and keep the source's measured timing relations.

1. construction: at every valid row, arm a(t) - q(t+lead) must equal the source residual transported along the time
   warp, r_src(tau(t)), per joint (|dev| <= 1e-6 rad); gripper commands must equal the zero-order hold of the source
   command stream, a_src[floor(tau(t))]. Catches source-action substitution, single-channel delays, any edit.
2. per-joint lead: for each of the 14 arm joints, the sub-frame lead (parabolic refinement of the k minimising
   mean |a_j(t) - s_j(t+k)| over that joint's informative-motion frames) must match the source joint's lead within
   0.75 frame (independent of the declared construction). Integer argmins were unstable: the true per-joint lead is
   ~1.5 frames, so they flipped between 1 and 2 on valid candidates (v3 init sweep, 105-122 false rejections).
3. gripper chronology: command transitions (openness 0.5) and planned-state transitions (openness 0.7) are paired
   per gripper and direction; the command->state latency must equal the source latency (+-1 frame) and command
   transitions must land at the warped source command times (+-1 frame). Catches command-only delays / premature
   opening that leave the state untouched.
4. integrity: sha256 of the candidate arrays, recomputed at export.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .recon3 import rb

ARM = [f"{s}J{i}.pos" for s in "RL" for i in range(1, 8)]
LEAD_TOL = 0.75     # frames: sub-frame per-joint lead must match the source joint's (integer argmins flip 1<->2 at ~1.5)
GRIP = {"R": "RJ8.pos", "L": "LJ8.pos"}


def candidate_hash(c) -> str:
    h = hashlib.sha256()
    for k in ("state", "action", "can", "tau"):
        h.update(np.ascontiguousarray(np.nan_to_num(np.asarray(c[k], float), nan=-999.0)).tobytes())
    return h.hexdigest()


def construction(rec, c) -> dict:
    T, L = rec.T, rec.lead
    tau = c["tau"]
    valid = c["valid"]
    rows = np.flatnonzero(valid)
    dev = {}
    for name in ARM:
        j = rb.NAMES.index(name)
        r = rec.action[: T - L, j] - rec.state[L:, j]
        exp = c["state"][rows + L, j] + np.interp(tau[rows], np.arange(T - L), r)
        dev[name] = float(np.abs(c["action"][rows, j] - exp).max())
    zi = np.clip(np.floor(tau + 1e-9).astype(int), 0, T - 1)
    gdev = {s: float(np.abs(c["action"][rows, rb.NAMES.index(n)] - rec.action[zi[rows], rb.NAMES.index(n)]).max())
            for s, n in GRIP.items()}
    worst = max(dev, key=dev.get)
    ok = max(dev.values()) <= 1e-6 and max(gdev.values()) <= 1e-9
    return {"ok": bool(ok), "max_arm_dev_rad": dev[worst], "worst_joint": worst, "gripper_dev": gdev}


def joint_leads(state, action, valid=None, kmax=8, thr=0.004, min_n=20) -> dict:
    T = len(state)
    valid = np.isfinite(action).all(1) if valid is None else valid
    out = {}
    for name in ARM:
        j = rb.NAMES.index(name)
        s, a = state[:, j], action[:, j]
        v = np.r_[0, np.abs(np.diff(s))] > thr
        v = np.convolve(v, np.ones(11), "same") > 0
        err = {}
        for k in range(-kmax, kmax + 1):
            t = np.arange(max(0, -k), min(T, T - k))
            t = t[valid[t] & v[t]]
            err[k] = float(np.abs(a[t] - s[t + k]).mean()) if len(t) >= min_n else np.inf
        best = min(err, key=err.get)
        srt = sorted(e for e in err.values() if np.isfinite(e))
        conf = (srt[1] / srt[0]) if len(srt) > 1 and srt[0] > 0 else np.inf
        frac = float(best)
        if -kmax < best < kmax and all(np.isfinite(err[best + d]) for d in (-1, 1)):
            a, b, c = err[best - 1], err[best], err[best + 1]
            den = a - 2 * b + c
            frac = best + (0.5 * (a - c) / den if den > 0 else 0.0)        # parabolic sub-frame minimum
        out[name] = {"lead": int(best), "lead_frac": frac, "err": err[best], "confidence": float(conf), "n": int(v.sum()),
                     "informative": bool(np.isfinite(err[best]))}
    return out


def transitions(x, thr):
    b = x < thr
    return [(int(t), "close" if b[t] else "open") for t in range(1, len(b)) if b[t] != b[t - 1]]


def gripper_chronology(state, action, valid) -> dict:
    g_s, g_a = rb.Robot.split(state), rb.Robot.split(np.where(np.isfinite(action), action, np.nan))
    out = {}
    n = int(np.flatnonzero(valid).max()) + 1 if valid.any() else 0       # valid rows are a prefix (tail trimmed)
    for s in "RL":
        cmd = transitions(g_a["g" + s][:n], 0.5)
        st = transitions(g_s["g" + s], 0.7)
        pairs = []
        for tc, kind in cmd:
            nxt = next((ts for ts, k2 in st if k2 == kind and ts >= tc), None)
            pairs.append({"cmd": tc, "kind": kind, "state": nxt, "latency": None if nxt is None else nxt - tc})
        out[s] = pairs
    return out


def per_joint_check(rec, c, src_leads=None) -> dict:
    src = src_leads or joint_leads(rec.state, rec.action, np.ones(rec.T, bool))
    gen = joint_leads(c["state"], c["action"], c["valid"])
    bad = [n for n in ARM if src[n]["informative"] and gen[n]["informative"]
           and abs(src[n]["lead_frac"] - gen[n]["lead_frac"]) > LEAD_TOL]
    return {"ok": not bad, "mismatched_joints": bad, "leads": {n: round(gen[n]["lead_frac"], 2) for n in ARM},
            "source_leads": {n: round(src[n]["lead_frac"], 2) for n in ARM},
            "max_lead_diff": max(abs(src[n]["lead_frac"] - gen[n]["lead_frac"]) for n in ARM if src[n]["informative"] and gen[n]["informative"]),
            "worst_err": max(gen[n]["err"] for n in ARM)}


def gripper_check(rec, c, src_chron=None) -> dict:
    T = rec.T
    srcc = src_chron or gripper_chronology(rec.state, rec.action, np.ones(T, bool))
    gen = gripper_chronology(c["state"], c["action"], c["valid"])
    kn_new = np.interp(np.arange(T, dtype=float), c["tau"], np.arange(T, dtype=float))  # tau^-1 on the grid
    problems = []
    for s in "RL":
        ps, pg = [p for p in srcc[s] if p["cmd"] < T - rec.lead], [p for p in gen[s]]
        if len(ps) != len(pg):
            problems.append(f"{s}: {len(pg)} command transitions vs source {len(ps)}")
            continue
        for a, b in zip(ps, pg):
            want = float(np.interp(a["cmd"], c["tau"], np.arange(T)))
            if a["kind"] != b["kind"] or abs(b["cmd"] - want) > 1:
                problems.append(f"{s}: {b['kind']} command at {b['cmd']} vs warped source {want:.1f}")
            if a["latency"] is not None and (b["latency"] is None or abs(b["latency"] - a["latency"]) > 1):
                problems.append(f"{s}: command->state latency {b['latency']} vs source {a['latency']}")
    del kn_new
    return {"ok": not problems, "problems": problems, "chronology": gen}


def validate(rec, c, src_leads=None, src_chron=None) -> dict:
    out = {"construction": construction(rec, c), "per_joint_lead": per_joint_check(rec, c, src_leads),
           "gripper": gripper_check(rec, c, src_chron), "hash": candidate_hash(c)}
    out["ok"] = bool(out["construction"]["ok"] and out["per_joint_lead"]["ok"] and out["gripper"]["ok"])
    return out
