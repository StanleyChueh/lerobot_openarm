"""Synthetic counterexample fixtures for Gate 1R (deterministic, no dataset needed).

A commanded bimanual hand-over with a known command-response law, first-order jaw dynamics that stall on the object,
a rigid object attachment to the giver then the receiver, and TCP trajectories with rotation and depth change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "gate1")]
import gate1_core as gc  # noqa: E402
import spec as S  # noqa: E402
import verifier_r as vr  # noqa: E402
from gate1_core import g0  # noqa: E402

NAMES = [f"{p}J{i}.pos" for i in range(1, 9) for p in ("R", "L")]
EV = {"rc": 60, "lc": 150, "ro": 156}
ALPHA, BETA, LEAD = 0.45, 0.30, 2


def rot(axis, ang):
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * ang).as_matrix()


def hands(T, phase=0.0):
    t = np.linspace(0, 1, T)
    out = {}
    for h, (x0, y0) in (("right", (0.25, -0.05)), ("left", (0.22, 0.12))):
        P = np.zeros((T, 4, 4))
        for i, s in enumerate(t):
            P[i, :3, :3] = rot([0.3, 1, 0.2], 2.5 * s + phase) @ rot([1, 0, 0], np.pi * 0.6)
            P[i, :3, 3] = [x0 + 0.12 * np.sin(2 * np.pi * s + phase), y0 + 0.08 * np.sin(np.pi * s), 0.30 + 0.15 * s]
            P[i, 3, 3] = 1
        out[h] = P
    return out


def episode(seed=0, T=240, noise_px=0.5):
    rng = np.random.default_rng(seed)
    a = np.zeros((T, 16))
    q = np.zeros((T, 16))
    t = np.arange(T)
    for k, n in enumerate(NAMES):
        if n.endswith("8.pos"):
            continue
        a[:, k] = 0.3 * np.sin(2 * np.pi * (t / T) * (1 + 0.1 * k) + seed + k) + 0.5
    q[:2] = a[:2]
    for i in range(1, T - 1):
        q[i + 1] = q[i] + ALPHA * (a[max(i - LEAD, 0)] - q[i]) + BETA * (q[i] - q[i - 1])
    q[:, [k for k, n in enumerate(NAMES) if not n.endswith("8.pos")]] += rng.normal(0, 2e-4, (T, 14))
    rc, lc, ro = EV["rc"], EV["lc"], EV["ro"]
    for h, close, open_, stall in (("right", rc, ro, 0.42), ("left", lc, None, 0.35)):
        j = NAMES.index(vr.GRIP[h][0])
        cmd = np.ones(T)
        cmd[close: open_ if open_ else T] = 0.0
        st = np.ones(T) * 0.96
        for i in range(1, T):
            target = stall if cmd[i] == 0 else 0.96
            st[i] = st[i - 1] + 0.4 * (target - st[i - 1])
        _, o, c = vr.GRIP[h]
        a[:, j] = c + cmd * (o - c)
        q[:, j] = c + st * (o - c)
    P = hands(T, phase=0.1 * seed)
    # after releasing, the giver retreats (as in the recorded episodes): 8 cm over 20 frames, away from the object
    ret = np.clip((np.arange(T) - ro) / 20.0, 0, 1)
    P["right"][:, :3, 3] += ret[:, None] * np.array([0.0, -0.08, 0.0])
    pR = np.array([0.12, 0.02, -0.05])
    mid = (lc + ro) // 2
    Xm = P["right"][mid, :3, :3] @ pR + P["right"][mid, :3, 3]
    pL = P["left"][mid, :3, :3].T @ (Xm - P["left"][mid, :3, 3])
    X0 = P["right"][rc, :3, :3] @ pR + P["right"][rc, :3, 3]
    uv = np.zeros((T, 2))
    for i in range(T):
        X = X0 if i < rc else (P["right"][i, :3, :3] @ pR + P["right"][i, :3, 3] if i < ro else P["left"][i, :3, :3] @ pL + P["left"][i, :3, 3])
        uv[i] = g0.project(X[None])[0][0] + rng.normal(0, noise_px, 2)
    s = vr.Sample("original", "valid", "valid", seed, seed, q, a, list(NAMES), uv, np.ones(T, bool))
    return s, P, {"pR": pR, "pL": pL}


def fold_from(dev_seeds=(1, 2)):
    """Freeze a fold from synthetic development episodes with the same rules as run_gate1r.freeze_fold."""
    devs = [episode(sd) for sd in dev_seeds]
    arm = [i for i, n in enumerate(NAMES) if not n.endswith("8.pos")]
    model = vr.ResponseModel(arm).fit([(s.q, s.a) for s, _, _ in devs])
    rms = [model.rms(s.q, s.a) for s, _, _ in devs]
    tau_A = {j: S.fold_threshold([r[j] for r in rms], S.A["response_floor_rad"]) for j in arm}
    st = [vr.dev_statistics(s, {}, P) for s, P, _ in devs]
    tau = {"giver_pred": S.fold_threshold([x.get("giver_pred") for x in st], S.G["B_floor_px"]),
           "recv_pred": S.fold_threshold([x.get("recv_pred") for x in st], S.G["B_floor_px"]),
           "transition": S.fold_threshold([x.get("transition") for x in st], S.G["C_floor_px"]),
           "release": S.fold_threshold([x.get("release") for x in st], kind="low")}
    return {"model": model, "tau_A": tau_A, "tau": tau, "limits": {j: (-3.5, 3.5) for j in arm}}


def gate1_view(s: vr.Sample, P) -> gc.Sample:
    ev = vr.command_events(s)
    events = {"right_close": ev["raw"]["right"][0][0], "left_close": ev["raw"]["left"][0][0], "right_open": ev["raw"]["right"][0][1]}
    return gc.Sample(s.name, s.family, True, s.joints_ep, s.video_ep, P, events, s.uv.copy(), s.ok.copy())
