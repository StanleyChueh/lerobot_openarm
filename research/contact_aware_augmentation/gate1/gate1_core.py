"""Gate 1 verifier core: rigid-attachment geometry, phase structure, scores and failure injection (CPU only).

Object model (replaces Gate-0 M2, which Codex showed is blind to constant offsets and flags valid rotations):

    u_obj(t) = pi( T_world<-hand(q_measured(t + lag)) @ p_hand )

with p_hand a fixed 3-D material point (the lid centre) in the holding hand's TCP frame, fitted per hold by robust
least squares on image-only tracker output. Measured joint state q (not the command) gives the robot pose; command
(action) gripper events give the phases. Nothing here reads force, torque or contact labels.

Scores (larger = more suspicious):
  A  video only           : object-track plausibility (acceleration outliers, pre-grasp stillness); no robot data
  B  robot geometry       : two-hand rigid mixture with free per-frame hand assignment (no phase knowledge)
  Bp generic proximity    : distance from the object to the nearest projected TCP (baseline)
  C  phase-aware geometry : giver/receiver windows from the command phase structure; per-window rigid residual,
                            absolute attachment vs development episodes, take-over continuity, release check,
                            phase logic
  E  event/time only      : object motion onset vs giver close command; phase logic; no geometry
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "research/contact_aware_augmentation/scripts"), str(REPO)]
import gate0_diagnostics as g0  # noqa: E402

HANDS = ("right", "left")
# Pre-specified BEFORE any test episode was scored (Gate 1 plan §5 + Codex corrections). Floors guard against
# thresholds collapsing on two development episodes.
FLOORS = {"res_giver": 3.0, "res_recv": 3.0, "takeover_px": 10.0, "abs_giver_mm": 10.0, "abs_recv_mm": 10.0,
          "B_res": 3.0, "B_abs_mm": 10.0, "Bp": 10.0, "A_acc": 2.0, "A_still": 5.0, "E_onset": 3.0}
MARGIN = 1.5
WINDOW = {"giver_start": 10, "giver_end": -3, "recv_start": 8, "tail_trim": 10, "release": (8, 25)}


# ----------------------------------------------------------------------------------------------- camera
def perturbed_cam(rot_deg=(0.0, 0.0, 0.0), trans_m=(0.0, 0.0, 0.0), f_scale=1.0, base=None) -> dict:
    """Body cam with a small rotation (camera-frame axes, degrees) and translation (camera frame, m)."""
    from scipy.spatial.transform import Rotation as Rot
    cam = copy.deepcopy(base or g0.BODY_CAM)
    R = g0.quat_wxyz_to_R(cam["quat_wxyz"])
    dR = Rot.from_rotvec(np.radians(rot_deg)).as_matrix()
    R2 = R @ dR
    q = Rot.from_matrix(R2).as_quat()  # x, y, z, w
    cam["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
    cam["pos"] = list(np.asarray(cam["pos"]) + R @ np.asarray(trans_m))
    cam["fx"] *= f_scale
    cam["fy"] *= f_scale
    return cam


def project_points(Ts: np.ndarray, p: np.ndarray, cam: dict) -> np.ndarray:
    """Ts (N,4,4) hand poses, p (3,) point in hand frame -> (N,2) pixels."""
    X = Ts[:, :3, :3] @ p + Ts[:, :3, 3]
    uv, _ = g0.project(X, cam)
    return uv


# ----------------------------------------------------------------------------------------------- data
@dataclass
class Episode:
    ep: int
    q: np.ndarray            # (T,16) measured state
    a: np.ndarray            # (T,16) command
    names: list
    T_hand: dict             # hand -> (T,4,4) TCP pose from measured q, ALREADY lag-applied (pose at t + lag)
    events: dict             # action-defined: right_close, left_close, right_open
    uv: np.ndarray           # (T,2) tracker output
    ok: np.ndarray           # (T,) tracker status == ok


@dataclass
class Sample:
    """What the verifier sees: a joint stream (poses + command events) and an object track."""
    name: str
    family: str
    valid: bool
    joints_ep: int
    video_ep: int
    T_hand: dict
    events: dict
    uv: np.ndarray
    ok: np.ndarray
    meta: dict = field(default_factory=dict)


def fk_hands(fk, q: np.ndarray, lag: float = 0.0) -> dict:
    """TCP poses for video frame t from the measured state at t + lag (fractional: linear interpolation of q,
    clamped at the ends). A negative lag means the image shows an earlier joint state than its row."""
    T = len(q)
    tq = np.clip(np.arange(T) + lag, 0, T - 1)
    i0 = np.floor(tq).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (tq - i0)[:, None]
    qi = (1 - w) * q[i0] + w * q[i1]
    out = {h: np.zeros((T, 4, 4)) for h in HANDS}
    for t in range(T):
        F = fk.frames(qi[t])
        for h in HANDS:
            out[h][t] = F[f"openarm_{h}_hand_tcp"]
    return out


def phase_structure(events: dict) -> dict:
    """Template-free giver/receiver from the command events: giver = hand that closes first."""
    rc, lc, ro = events["right_close"], events["left_close"], events["right_open"]
    lo = events.get("left_open")
    if rc is not None and (lc is None or rc <= lc):
        return {"giver": "right", "receiver": "left", "g_close": rc, "r_close": lc, "g_open": ro}
    return {"giver": "left", "receiver": "right", "g_close": lc, "r_close": rc, "g_open": lo}


def sample_from_episode(E: Episode, name="original", family="valid", valid=True) -> Sample:
    return Sample(name, family, valid, E.ep, E.ep, E.T_hand, dict(E.events), E.uv.copy(), E.ok.copy())


# ----------------------------------------------------------------------------------------------- fitting
def fit_point(Ts, uv, cam, lag=0, inits=None):
    """Robust LSQ for p (hand frame) s.t. pi(T p) ~ uv. Returns p, per-frame residual (px), cov-based std (m)."""
    if len(uv) < 5:
        return None
    inits = inits or [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.05, 0.0, 0.05), (-0.05, 0.0, 0.05), (0.0, 0.0, -0.05)]
    best = None
    for p0 in inits:
        r = least_squares(lambda p: (project_points(Ts, p, cam) - uv).ravel(), np.asarray(p0, float),
                          loss="soft_l1", f_scale=3.0, max_nfev=200)
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    res = np.linalg.norm(project_points(Ts, p, cam) - uv, axis=1)
    J = best.jac
    try:
        s2 = float(np.median(res) ** 2) + 1e-6
        cov = s2 * np.linalg.inv(J.T @ J)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        std = np.full(3, np.inf)
    return {"p": p, "res": res, "rms": float(np.sqrt(np.mean(res ** 2))), "std_m": std, "n": len(uv)}


def _idx(lo, hi, T):
    lo, hi = max(int(lo), 0), min(int(hi), T - 1)
    return np.arange(lo, hi + 1) if hi >= lo else np.arange(0)


def windows(s: Sample, tail_end: int) -> dict:
    """Frame index sets (video frames t) per phase, from the sample's own command events."""
    ph = phase_structure(s.events)
    T = len(s.uv)
    w = {"phase": ph}
    if ph["g_close"] is None or ph["r_close"] is None or ph["g_open"] is None:
        return w | {"giver": np.arange(0), "recv": np.arange(0), "takeover": np.arange(0), "release": np.arange(0)}
    w["giver"] = _idx(ph["g_close"] + WINDOW["giver_start"], ph["r_close"] + WINDOW["giver_end"], T)
    w["takeover"] = _idx(min(ph["r_close"], ph["g_open"]), max(ph["r_close"], ph["g_open"]), T)
    w["recv"] = _idx(max(ph["g_open"], ph["r_close"]) + WINDOW["recv_start"], tail_end, T)
    w["release"] = _idx(ph["g_open"] + WINDOW["release"][0], ph["g_open"] + WINDOW["release"][1], T)
    w["pre"] = _idx(15, ph["g_close"] - 25, T)
    return w


def _sel(s: Sample, idx):
    """Frames t in idx with an ok track."""
    idx = idx[idx < len(s.uv)]
    return idx[s.ok[idx]]


# ----------------------------------------------------------------------------------------------- scores
def score_C(s: Sample, cam, tail_end, ref=None) -> dict:
    w = windows(s, tail_end)
    ph = w["phase"]
    out = {"phase_logic_ok": bool(ph["g_open"] is not None and ph["r_close"] is not None and ph["r_close"] < ph["g_open"]),
           "giver": ph["giver"]}
    fits = {}
    for role, hand in (("giver", ph["giver"]), ("recv", ph["receiver"])):
        idx = _sel(s, w[role])
        f = fit_point(s.T_hand[hand][idx], s.uv[idx], cam) if len(idx) >= 8 else None
        fits[role] = f
        out[f"res_{role}"] = f["rms"] if f else np.nan
        out[f"n_{role}"] = int(len(idx))
        out[f"p_{role}"] = f["p"].tolist() if f else None
        out[f"p_{role}_std_mm"] = (f["std_m"] * 1e3).tolist() if f else None
        if ref is not None and f is not None and ref.get(role) is not None:
            out[f"abs_{role}_mm"] = float(np.linalg.norm(f["p"] - ref[role]) * 1e3)
        else:
            out[f"abs_{role}_mm"] = np.nan
    # take-over continuity: both hands' predictions of the same material point must agree (pixels)
    idx = w["takeover"]
    if fits["giver"] and fits["recv"] and len(idx):
        a = project_points(s.T_hand[ph["giver"]][idx], fits["giver"]["p"], cam)
        b = project_points(s.T_hand[ph["receiver"]][idx], fits["recv"]["p"], cam)
        out["takeover_px"] = float(np.median(np.linalg.norm(a - b, axis=1)))
    else:
        out["takeover_px"] = np.nan
    # release: after the giver-open command the object must stop following the giver (median distance, px)
    idx = _sel(s, w["release"])
    if fits["giver"] and len(idx) >= 3:
        pg = project_points(s.T_hand[ph["giver"]][idx], fits["giver"]["p"], cam)
        out["release_px"] = float(np.median(np.linalg.norm(pg - s.uv[idx], axis=1)))
    else:
        out["release_px"] = np.nan
    return out


def score_B(s: Sample, cam, tail_end, ref=None, iters=6) -> dict:
    """Phase-free: after the object first moves, every ok frame is explained by the better of two rigid points
    (one per hand), assignments re-estimated freely (alternating fit)."""
    T = len(s.uv)
    ok_idx = np.where(s.ok)[0]
    if len(ok_idx) < 20:
        return {"B_res": np.nan, "B_abs_mm": np.nan, "Bp": np.nan}
    u0 = s.uv[ok_idx[0]]
    moved = ok_idx[np.linalg.norm(s.uv[ok_idx] - u0, axis=1) > 20]
    if len(moved) < 20:
        return {"B_res": np.nan, "B_abs_mm": np.nan, "Bp": np.nan}
    first = moved[0]
    idx = _sel(s, np.arange(first, tail_end + 1))
    # generic proximity baseline
    d = np.min(np.stack([np.linalg.norm(g0.project(s.T_hand[h][idx][:, :3, 3], cam)[0] - s.uv[idx], axis=1)
                         for h in HANDS]), axis=0)
    out = {"Bp": float(np.percentile(d, 95))}
    # initial assignment: nearest projected TCP
    assign = np.argmin(np.stack([np.linalg.norm(g0.project(s.T_hand[h][idx][:, :3, 3], cam)[0] - s.uv[idx], axis=1)
                                 for h in HANDS]), axis=0)
    ps = {}
    for _ in range(iters):
        res = []
        for k, h in enumerate(HANDS):
            sel = idx[assign == k]
            f = fit_point(s.T_hand[h][sel], s.uv[sel], cam) if len(sel) >= 8 else None
            ps[h] = f["p"] if f else ps.get(h)
        for k, h in enumerate(HANDS):
            res.append(np.linalg.norm(project_points(s.T_hand[h][idx], ps[h], cam) - s.uv[idx], axis=1)
                       if ps.get(h) is not None else np.full(len(idx), np.inf))
        res = np.stack(res)
        new = np.argmin(res, axis=0)
        if np.array_equal(new, assign):
            break
        assign = new
    r = res.min(axis=0)
    out["B_res"] = float(np.sqrt(np.mean(r ** 2)))
    if ref is not None:
        dd = [np.linalg.norm(ps[h] - ref[h]) * 1e3 for h in HANDS if ps.get(h) is not None and ref.get(h) is not None]
        out["B_abs_mm"] = float(max(dd)) if dd else np.nan
    else:
        out["B_abs_mm"] = np.nan
    out["B_p"] = {h: (ps[h].tolist() if ps.get(h) is not None else None) for h in HANDS}
    return out


def score_A(s: Sample, tail_end) -> dict:
    """Video only: acceleration outliers over the scored span, and pre-grasp stillness."""
    idx = np.arange(0, tail_end + 1)
    ok = s.ok[idx]
    u = s.uv[idx]
    acc = []
    for t in range(2, len(idx)):
        if ok[t] and ok[t - 1] and ok[t - 2]:
            acc.append(np.linalg.norm(u[t] - 2 * u[t - 1] + u[t - 2]))
    acc = np.asarray(acc)
    out = {"A_acc": float(np.max(acc) / (np.median(acc) + 1e-3)) if len(acc) else np.nan}
    ph = phase_structure(s.events)
    pre = _idx(15, (ph["g_close"] or 0) - 25, len(s.uv))
    pre = pre[s.ok[pre]]
    out["A_still"] = float(np.max(np.linalg.norm(s.uv[pre] - s.uv[pre[0]], axis=1))) if len(pre) > 2 else np.nan
    return out


def score_E(s: Sample) -> dict:
    """Event/time only: frame at which the object first moves > 15 px from its start vs the giver close command."""
    ph = phase_structure(s.events)
    ok_idx = np.where(s.ok)[0]
    u0 = s.uv[ok_idx[0]]
    mv = ok_idx[np.linalg.norm(s.uv[ok_idx] - u0, axis=1) > 15]
    onset = int(mv[0]) if len(mv) else None
    return {"E_onset_minus_close": (onset - ph["g_close"]) if (onset is not None and ph["g_close"] is not None) else np.nan,
            "phase_logic_ok": bool(ph["g_open"] is not None and ph["r_close"] is not None and ph["r_close"] < ph["g_open"])}


def score_all(s: Sample, cam, tail_end, refC=None, refB=None) -> dict:
    out = {"name": s.name, "family": s.family, "valid": s.valid, "joints_ep": s.joints_ep, "video_ep": s.video_ep}
    out.update({f"C.{k}": v for k, v in score_C(s, cam, tail_end, refC).items()})
    out.update({f"B.{k}": v for k, v in score_B(s, cam, tail_end, refB).items()})
    out.update({f"A.{k}": v for k, v in score_A(s, tail_end).items()})
    out.update({f"E.{k}": v for k, v in score_E(s).items()})
    return out


# ----------------------------------------------------------------------------------------------- thresholds
SCORE_COMPONENTS = {
    "A": [("A.A_acc", "A_acc", "high"), ("A.A_still", "A_still", "high")],
    "Bp": [("B.Bp", "Bp", "high")],
    "B": [("B.B_res", "B_res", "high"), ("B.B_abs_mm", "B_abs_mm", "high")],
    "C": [("C.res_giver", "res_giver", "high"), ("C.res_recv", "res_recv", "high"), ("C.takeover_px", "takeover_px", "high"),
          ("C.abs_giver_mm", "abs_giver_mm", "high"), ("C.abs_recv_mm", "abs_recv_mm", "high"),
          ("C.release_px", "release_px", "low")],
    "E": [("E.E_onset_minus_close", "E_onset", "abs")],
    # POST-HOC variants (defined after the pre-specified B/C false-rejected 2 of 3 real episodes through the absolute-
    # attachment and take-over components). Exploratory only: must be re-validated on the sealed episodes.
    "B_rel": [("B.B_res", "B_res", "high")],
    "C_rel": [("C.res_giver", "res_giver", "high"), ("C.res_recv", "res_recv", "high"), ("C.release_px", "release_px", "low")],
    "C_rel_take": [("C.res_giver", "res_giver", "high"), ("C.res_recv", "res_recv", "high"),
                   ("C.takeover_px", "takeover_px", "high"), ("C.release_px", "release_px", "low")],
}
PREREGISTERED = ("A", "Bp", "B", "C", "E")
POSTHOC = ("B_rel", "C_rel", "C_rel_take")
LOGIC = {"C": "C.phase_logic_ok", "E": "E.phase_logic_ok", "C_rel": "C.phase_logic_ok", "C_rel_take": "C.phase_logic_ok"}


def freeze_thresholds(dev_rows: list[dict]) -> dict:
    """From valid development rows only. high: tau = max(MARGIN*max_dev, floor); low: tau = min_dev / MARGIN;
    abs (E): reference = mean dev value, tau = max(MARGIN*max |dev - ref|, floor)."""
    th = {}
    for method, comps in SCORE_COMPONENTS.items():
        for col, key, kind in comps:
            v = np.array([r[col] for r in dev_rows if r.get(col) is not None and np.isfinite(r[col])], float)
            if not len(v):
                th[col] = {"kind": kind, "tau": np.nan}
                continue
            if kind == "high":
                th[col] = {"kind": kind, "tau": float(max(MARGIN * v.max(), FLOORS.get(key, 0.0)))}
            elif kind == "low":
                th[col] = {"kind": kind, "tau": float(v.min() / MARGIN)}
            else:
                ref = float(v.mean())
                th[col] = {"kind": kind, "ref": ref, "tau": float(max(MARGIN * np.abs(v - ref).max(), FLOORS.get(key, 0.0)))}
    return th


def method_score(row: dict, th: dict, method: str) -> tuple[float, bool, list]:
    """Continuous score = max over components of value/tau (>1 = flagged); NaN component counts as unevaluable."""
    vals, why = [], []
    unevaluable = False
    for col, key, kind in SCORE_COMPONENTS[method]:
        t = th[col]
        v = row.get(col)
        if v is None or not np.isfinite(v) or not np.isfinite(t["tau"]):
            unevaluable = True
            continue
        if kind == "high":
            z = v / t["tau"]
        elif kind == "low":
            z = t["tau"] / max(v, 1e-6)
        else:
            z = abs(v - t["ref"]) / t["tau"]
        vals.append(z)
        if z > 1:
            why.append(col)
    if method in LOGIC and not row.get(LOGIC[method], True):
        vals.append(10.0)
        why.append(LOGIC[method])
    s = float(max(vals)) if vals else np.nan
    return s, bool(vals) and s > 1.0, why + (["unevaluable_component"] if unevaluable else [])


# ----------------------------------------------------------------------------------------------- injections
def lid_3d(E: Episode, fitR, fitL):
    """3-D lid position per frame from the episode's own rigid fits (giver = right before take-over, receiver = left)."""
    T = len(E.uv)
    X = np.full((T, 3), np.nan)
    rc, lc, ro = E.events["right_close"], E.events["left_close"], E.events["right_open"]
    for t in range(T):
        tt = t
        if rc <= t < ro and fitR is not None:
            X[t] = E.T_hand["right"][tt, :3, :3] @ fitR + E.T_hand["right"][tt, :3, 3]
        elif t >= ro and fitL is not None:
            X[t] = E.T_hand["left"][tt, :3, :3] @ fitL + E.T_hand["left"][tt, :3, 3]
    return X


def _ramp(T, t0, t1):
    w = np.zeros(T)
    t = np.arange(T)
    w[t >= t1] = 1.0
    m = (t >= t0) & (t < t1)
    w[m] = 0.5 - 0.5 * np.cos(np.pi * (t[m] - t0) / max(t1 - t0, 1))
    return w


def inject(E: Episode, fits: dict, cam) -> list[Sample]:
    """Corrupted samples for one episode. Displacements are applied in 3-D and re-projected (perspective-correct)
    as a delta on top of the REAL tracker output, so tracker noise and ok-flags are retained."""
    out = []
    T = len(E.uv)
    rc, lc, ro = E.events["right_close"], E.events["left_close"], E.events["right_open"]
    pR, pL = fits["giver"], fits["recv"]
    X = lid_3d(E, pR, pL)
    base = sample_from_episode(E)
    held = ~np.isnan(X[:, 0])

    def with_delta(name, family, dX, meta):
        s = copy.deepcopy(base)
        s.name, s.family, s.valid, s.meta = name, family, False, meta
        m = held & np.any(dX != 0, axis=1)
        uv0, _ = g0.project(X[m], cam)
        uv1, _ = g0.project(X[m] + dX[m], cam)
        s.uv[m] = s.uv[m] + (uv1 - uv0)
        return s

    # 1. object displacement, 1/2/4 cm. (a) world frame, constant after a smooth ramp over [grasp, grasp+10) that
    #    ends before the scored giver window (no 3-D lid estimate exists before the grasp, so the ramp must start at
    #    it to avoid a step); (b) hand frame (rigid but wrong attachment) along the jaw axis (y) and the approach
    #    axis (z).
    ramp = _ramp(T, rc, rc + WINDOW["giver_start"])
    for cm in (1, 2, 4):
        for ax, vec in (("x", (1, 0, 0)), ("y", (0, 1, 0))):
            dX = ramp[:, None] * (np.asarray(vec) * cm / 100.0)[None]
            out.append(with_delta(f"disp_world_{ax}_{cm}cm", "displacement_world", dX, {"cm": cm, "axis": ax}))
        for ax, vec in (("jaw_y", (0, 1, 0)), ("approach_z", (0, 0, 1))):
            # constant in the giver's TCP frame until its release, then constant in the receiver's frame with the
            # value it had at the release instant: continuous across the transfer (no step for a track-only cue)
            dR = np.asarray(vec) * cm / 100.0
            dL = E.T_hand["left"][ro, :3, :3].T @ (E.T_hand["right"][ro, :3, :3] @ dR)
            dX = np.zeros((T, 3))
            for t in range(T):
                dX[t] = ramp[t] * (E.T_hand["right"][t, :3, :3] @ dR if t < ro else E.T_hand["left"][t, :3, :3] @ dL)
            out.append(with_delta(f"disp_hand_{ax}_{cm}cm", "displacement_hand", dX, {"cm": cm, "axis": ax}))
    # Codex counterexample: constant 20-px image offset over the whole hold, including t0 (no boundary in window)
    s = copy.deepcopy(base)
    s.name, s.family, s.valid = "offset_px20_const", "displacement_px", False
    s.uv = s.uv + _ramp(T, rc, rc + WINDOW["giver_start"])[:, None] * np.array([20.0, 0.0])
    out.append(s)
    # 2. temporal offsets: the object track lags (+k) or leads (-k) the joint stream by k frames
    for k in (3, 6, 10):
        for sign in (+1, -1):
            s = copy.deepcopy(base)
            s.name, s.family, s.valid, s.meta = f"time_{'lag' if sign > 0 else 'lead'}_{k}", "temporal", False, {"k": k * sign}
            src = np.clip(np.arange(T) - sign * k, 0, T - 1)
            s.uv, s.ok = E.uv[src].copy(), E.ok[src].copy()
            s.ok[(np.arange(T) - sign * k < 0) | (np.arange(T) - sign * k > T - 1)] = False
            out.append(s)
    # 3. slip during hand-over: after the receiver closes and the giver opens, the object slides down (world -z)
    #    by 2/4 cm over 15 frames inside the receiver hold; and a transfer failure (object stays with the giver)
    for cm in (2, 4):
        dX = _ramp(T, ro + 5, ro + 20)[:, None] * np.array([0, 0, -cm / 100.0])[None]
        dX[:ro] = 0
        out.append(with_delta(f"slip_handover_{cm}cm", "slip", dX, {"cm": cm}))
    s = copy.deepcopy(base)
    s.name, s.family, s.valid = "transfer_failure_stays_with_giver", "slip", False
    if pR is not None and pL is not None:
        idx = np.arange(ro, T)
        tt = idx
        uvR = project_points(E.T_hand["right"][tt], pR, cam)
        uvL = project_points(E.T_hand["left"][tt], pL, cam)
        s.uv[idx] = E.uv[idx] + (uvR - uvL)  # real tracker residual kept on top of the giver's trajectory
        out.append(s)
    # 4. wrong giver/receiver phase (joint-stream corruption; video untouched)
    s = copy.deepcopy(base)
    s.name, s.family, s.valid = "phase_roles_swapped", "phase", False
    s.T_hand = {"right": E.T_hand["left"], "left": E.T_hand["right"]}  # the stream claims the other arm did it
    out.append(s)
    s = copy.deepcopy(base)
    s.name, s.family, s.valid = "phase_giver_releases_10_before_receiver", "phase", False
    s.events = dict(E.events, right_open=lc - 10)
    out.append(s)
    s = copy.deepcopy(base)
    s.name, s.family, s.valid = "phase_receiver_window_shifted_early_15", "phase", False
    s.events = dict(E.events, left_close=lc - 15, right_open=ro - 15)  # plausible order, wrong timing
    out.append(s)
    return out


def cross_pair(Ej: Episode, Ei: Episode) -> Sample:
    """Joints/phases of episode i with the video track of episode j (an actual mismatched pair through the tracker)."""
    T = min(len(Ei.uv), len(Ej.uv))
    return Sample(f"cross_joints{Ei.ep}_video{Ej.ep}", "cross_episode", False, Ei.ep, Ej.ep,
                  {h: v[:T] for h, v in Ei.T_hand.items()}, dict(Ei.events), Ej.uv[:T].copy(), Ej.ok[:T].copy())


def artifact_features(s: Sample, tail_end: int) -> dict:
    """Features of the object track alone that an injection could leave behind (task 9)."""
    idx = np.arange(0, tail_end + 1)
    ok = s.ok[idx]
    u = s.uv[idx]
    d1 = [np.linalg.norm(u[t] - u[t - 1]) for t in range(1, len(idx)) if ok[t] and ok[t - 1]]
    d2 = [np.linalg.norm(u[t] - 2 * u[t - 1] + u[t - 2]) for t in range(2, len(idx)) if ok[t] and ok[t - 1] and ok[t - 2]]
    d1, d2 = np.asarray(d1), np.asarray(d2)
    return {"n_ok": int(ok.sum()), "speed_p95": float(np.percentile(d1, 95)), "speed_max": float(d1.max()),
            "acc_p95": float(np.percentile(d2, 95)), "acc_max": float(d2.max()),
            "hf_ratio": float((d2 ** 2).sum() / max((d1 ** 2).sum(), 1e-9))}


def auc(pos: list[float], neg: list[float]) -> float:
    """P(score_invalid > score_valid), ties 1/2. pos = invalid scores, neg = valid scores."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()
    return float(gt / (len(pos) * len(neg)))
