"""Gate 1R multi-component verifier (CPU only). Every component returns PASS / FAIL / UNKNOWN with its reasons.

  A  action-state alignment and feasibility: wrapper contracts (tracking-error and jump bounds, joint limits),
     a command-response model fitted on development episodes, and gripper command -> measured jaw response
  B  object-to-gripper consistency: per hold, the attachment point is fitted ONLY on a calibration segment, then
     PREDICTED (never refit) on the following frames; observability, depth and attachment-distance priors gate it
  C  bimanual contact-phase consistency: measured jaw order, the OBSERVED object track across the whole transition
     (no unscored gap) against the frozen giver/receiver attachments of the hands that are holding, and departure
     from the giver after release
  D  evidence: every required segment must reach the coverage minimum; otherwise the dependent component is UNKNOWN

Combined: FAIL if any component FAILs; otherwise UNKNOWN if any is UNKNOWN; otherwise PASS. A PASS means
"kinematic-visual consistency was checked and held"; it is NOT a contact certification. Absolute contact (where the
object sits between the pads) needs measured object/pad geometry and validated calibration, and is reported as
UNKNOWN for every sample.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gate1")]
import gate1_core as gc  # noqa: E402
import spec as S  # noqa: E402
from gate1_core import g0  # noqa: E402

GRIP = {"right": ("RJ8.pos", -1.298, 0.0), "left": ("LJ8.pos", 1.2121, 0.0)}  # (name, open, closed) dataset convention
PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


@dataclass
class Sample:
    name: str
    family: str
    label: str             # "valid" | "invalid" | "ambiguous"
    joints_ep: int
    video_ep: int
    q: np.ndarray           # (T,16) measured state, interleaved names
    a: np.ndarray           # (T,16) command
    names: list
    uv: np.ndarray          # (T,2) object track
    ok: np.ndarray          # (T,) bool
    meta: dict = field(default_factory=dict)


def combine(statuses) -> str:
    st = list(statuses)
    return FAIL if FAIL in st else UNKNOWN if UNKNOWN in st else PASS


def openness(x, hand):
    _, o, c = GRIP[hand]
    return (np.asarray(x, float) - c) / (o - c)


def command_events(s: Sample) -> dict:
    """Command-defined gripper events and the template-free giver/receiver structure."""
    ev = {}
    for h in ("right", "left"):
        j = s.names.index(GRIP[h][0])
        seg = gc.g0.segments(openness(s.a[:, j], h) < 0.5)
        ev[h] = [(int(a), int(b) if b < len(s.a) else None) for a, b in seg]
    first = {h: (ev[h][0][0] if ev[h] else None) for h in ev}
    if first["right"] is None or first["left"] is None:
        return {"complete": False, "raw": ev}
    g = "right" if first["right"] <= first["left"] else "left"
    r = "left" if g == "right" else "right"
    return {"complete": ev[g][0][1] is not None, "raw": ev, "giver": g, "receiver": r,
            "g_close": ev[g][0][0], "g_open": ev[g][0][1], "r_close": ev[r][0][0]}


def segments(ev: dict, T: int) -> dict:
    w = S.W
    gc_, go, rc = ev["g_close"], ev["g_open"], ev["r_close"]
    s0 = gc_ + w["settle"]
    seg = {"giver_calib": (s0, s0 + w["calib"]), "giver_pred": (s0 + w["calib"], rc - w["trans_pre"]),
           "transition": (rc - w["trans_pre"], go + w["trans_post"]),
           "recv_calib": (go + w["recv_offset"], go + w["recv_offset"] + w["calib"]),
           "recv_pred": (go + w["recv_offset"] + w["calib"], go + w["recv_offset"] + w["recv_len"]),
           "release": (go + w["release"][0], go + w["release"][1])}
    return {k: np.arange(max(a, 0), min(b, T)) for k, (a, b) in seg.items()}


def coverage(s: Sample, idx) -> float:
    return float(s.ok[idx].mean()) if len(idx) else 0.0


# ----------------------------------------------------------------------------------------------- A
class ResponseModel:
    """Per-joint s[t+1]-s[t] = alpha (a[t-d]-s[t]) + beta (s[t]-s[t-1]); common lead d chosen on development data."""

    def __init__(self, arm_idx):
        self.arm = arm_idx

    @staticmethod
    def _xy(q, a, j, d):
        t = np.arange(max(d, 1), len(q) - 1)
        X = np.stack([a[t - d, j] - q[t, j], q[t, j] - q[t - 1, j]], 1)
        y = q[t + 1, j] - q[t, j]
        return X, y

    def fit(self, episodes):
        best = None
        for d in S.A["response_lead_candidates"]:
            coef, sse = {}, 0.0
            for j in self.arm:
                X = np.concatenate([self._xy(q, a, j, d)[0] for q, a in episodes])
                y = np.concatenate([self._xy(q, a, j, d)[1] for q, a in episodes])
                c, *_ = np.linalg.lstsq(X, y, rcond=None)
                coef[j] = c
                sse += float(((X @ c - y) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, d, coef)
        _, self.d, self.coef = best
        return self

    def rms(self, q, a) -> dict:
        out = {}
        for j in self.arm:
            X, y = self._xy(q, a, j, self.d)
            out[j] = float(np.sqrt(np.mean((X @ self.coef[j] - y) ** 2)))
        return out


def comp_A(s: Sample, model: ResponseModel, tau: dict, limits: dict) -> dict:
    r = {"reasons": []}
    arm = model.arm
    if not (np.isfinite(s.a).all() and np.isfinite(s.q).all()):
        return {"status": UNKNOWN, "reasons": ["non-finite action/state"]}
    gap = np.abs(s.a[:, arm] - s.q[:, arm])
    jump = np.abs(np.diff(s.a[:, arm], axis=0))
    r["max_tracking_error_rad"] = float(gap.max())
    r["max_command_jump_rad"] = float(jump.max()) if len(jump) else 0.0
    if r["max_tracking_error_rad"] > S.A["max_tracking_error_rad"]:
        r["reasons"].append("tracking_error_contract")
    if r["max_command_jump_rad"] > S.A["max_command_jump_rad"]:
        r["reasons"].append("command_jump_contract")
    m = S.A["joint_limit_margin_rad"]
    out = [s.names[j] for j in arm if (s.a[:, j] < limits[j][0] - m).any() or (s.a[:, j] > limits[j][1] + m).any()]
    if out:
        r["reasons"].append("joint_limit:" + ",".join(out))
    rms = model.rms(s.q, s.a)
    z = {s.names[j]: rms[j] / tau[j] for j in arm}
    r["response_z_max"] = float(max(z.values()))
    r["response_worst_joint"] = max(z, key=z.get)
    if r["response_z_max"] > 1:
        r["reasons"].append("command_response")
    # gripper: every command change must be followed by a measured jaw move; no large uncommanded jaw move
    for h in ("right", "left"):
        j = s.names.index(GRIP[h][0])
        cmd = openness(s.a[:, j], h) < 0.5
        st = openness(s.q[:, j], h)
        n = S.A["grip_response_frames"]
        changes = np.where(np.diff(cmd.astype(int)) != 0)[0] + 1
        for t in changes:
            w = st[t: t + n + 1]
            if len(w) < 2:
                continue
            moved = (st[t] - w.min()) if cmd[t] else (w.max() - st[t])
            if moved < S.A["grip_min_move"]:
                r["reasons"].append(f"{h}_jaw_ignores_command@{int(t)}")
        for t in range(0, len(st) - n):
            if abs(st[t + n] - st[t]) > S.A["grip_uncommanded_move"]:
                lo, hi = max(t - n, 0), min(t + 2 * n, len(cmd) - 1)
                if not (np.diff(cmd[lo: hi + 1].astype(int)) != 0).any():
                    r["reasons"].append(f"{h}_uncommanded_jaw_move@{t}")
                    break
    r["status"] = FAIL if r["reasons"] else PASS
    return r


# ----------------------------------------------------------------------------------------------- B
def calibrate(s: Sample, poses: dict, hand: str, idx, cam) -> dict:
    """Fit the attachment point on the calibration segment only; gate by evidence, observability and priors."""
    out = {"n_ok": int(s.ok[idx].sum()) if len(idx) else 0, "coverage": coverage(s, idx)}
    if out["n_ok"] < S.W["min_calib_ok"] or out["coverage"] < S.D["min_coverage"]:
        return out | {"status": UNKNOWN, "reason": "calibration_evidence"}
    sel = idx[s.ok[idx]]
    f = gc.fit_point(poses[hand][sel], s.uv[sel], cam)
    J = jacobian(poses[hand][sel], f["p"], cam)
    s2 = float(np.median(f["res"]) ** 2) + 1e-6
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)
    out.update({"p": f["p"], "cov": cov, "fit_rms_px": f["rms"], "std_mm": (np.sqrt(np.clip(np.diag(cov), 0, None)) * 1e3).tolist()})
    if not np.all(np.isfinite(cov)):
        return out | {"status": UNKNOWN, "reason": "attachment_unobservable"}
    if np.linalg.norm(f["p"]) > S.G["attach_max_norm_m"]:
        return out | {"status": FAIL, "reason": "attachment_implausible_distance"}
    X = poses[hand][sel][:, :3, :3] @ f["p"] + poses[hand][sel][:, :3, 3]
    if (g0.project(X, cam)[1] < S.G["min_depth_m"]).any():
        return out | {"status": FAIL, "reason": "attachment_behind_camera"}
    return out | {"status": PASS}


def jacobian(Ts, p, cam, h=1e-5):
    """d(pixels)/d(p), stacked (2N, 3), central differences."""
    cols = []
    for k in range(3):
        d = np.zeros(3); d[k] = h
        cols.append(((gc.project_points(Ts, p + d, cam) - gc.project_points(Ts, p - d, cam)) / (2 * h)).ravel())
    return np.stack(cols, 1)


def pred_sigma(poses, hand, cal, idx, cam) -> float:
    """p90 over frames of the largest 1-sigma of the predicted pixel, from the calibration covariance."""
    if not len(idx):
        return float("inf")
    J = jacobian(poses[hand][idx], cal["p"], cam).reshape(len(idx), 2, 3)
    sig = [np.sqrt(np.linalg.eigvalsh(Jt @ cal["cov"] @ Jt.T).max()) for Jt in J]
    return float(np.percentile(sig, 90))


def residuals(s, poses, hand, p, idx, cam):
    return np.linalg.norm(gc.project_points(poses[hand][idx], p, cam) - s.uv[idx], axis=1)


def predict_check(s, poses, hand, cal, idx, cam, tau) -> dict:
    out = {"coverage": coverage(s, idx), "n": int(len(idx))}
    if cal["status"] != PASS:
        return out | {"status": UNKNOWN if cal["status"] == UNKNOWN else FAIL, "reason": "calibration_" + cal["status"].lower()}
    if out["coverage"] < S.D["min_coverage"]:
        return out | {"status": UNKNOWN, "reason": "prediction_evidence"}
    sel = idx[s.ok[idx]]
    if tau is None:
        return out | {"status": UNKNOWN, "reason": "no_development_threshold"}
    out["pred_sigma_px"] = pred_sigma(poses, hand, cal, sel, cam)
    if out["pred_sigma_px"] > S.G["pred_sigma_max_frac_tau"] * tau:
        return out | {"status": UNKNOWN, "reason": "prediction_cannot_resolve_threshold"}
    e = residuals(s, poses, hand, cal["p"], sel, cam)
    out["p90_px"] = float(np.percentile(e, 90))
    hi = e > 2 * tau if tau else np.zeros(len(e), bool)
    run = max((len(x) for x in "".join("1" if h else "0" for h in hi).split("0")), default=0)
    out["longest_run_over_2tau"] = int(run)
    out["z"] = out["p90_px"] / tau
    bad = out["p90_px"] > tau or run >= S.G["run_len"]
    return out | {"status": FAIL if bad else PASS}


# ----------------------------------------------------------------------------------------------- C
def jaw_events(s: Sample, hand: str, after: int):
    st = openness(s.q[:, s.names.index(GRIP[hand][0])], hand)
    t = np.arange(len(st))
    c = t[(t >= after) & (st < 0.7)]
    if not len(c):
        return None, None, st
    o = t[(t > c[0]) & (st > 0.7)]
    return int(c[0]), (int(o[0]) if len(o) else None), st


def comp_C(s, poses, ev, seg, calG, calR, cam, tau) -> dict:
    r = {"reasons": []}
    g, rr = ev["giver"], ev["receiver"]
    gc_s, go_s, st_g = jaw_events(s, g, ev["g_close"] - 5)
    rc_s, _, st_r = jaw_events(s, rr, ev["r_close"] - 5)
    r["jaw_state_events"] = {"giver_close": gc_s, "giver_open": go_s, "receiver_close": rc_s}
    # C1: measured jaw order: receiver closes before giver opens; command order agrees
    if not (ev["g_close"] < ev["r_close"] < ev["g_open"]):
        r["reasons"].append("command_phase_order")
    if gc_s is None or rc_s is None or go_s is None or not (rc_s < go_s):
        r["reasons"].append("measured_jaw_order")
    # C2: observed transition against the holding hands' frozen attachments
    idx = seg["transition"]
    r["transition_coverage"] = coverage(s, idx)
    unknown = []
    if calG["status"] != PASS or calR["status"] != PASS:
        unknown.append("transition_needs_both_attachments")
    elif r["transition_coverage"] < S.D["min_coverage"]:
        unknown.append("transition_evidence")
    else:
        sel = idx[s.ok[idx]]
        r["transition_pred_sigma_px"] = max(pred_sigma(poses, g, calG, sel, cam), pred_sigma(poses, rr, calR, sel, cam))
    if not unknown and r["transition_pred_sigma_px"] > S.G["pred_sigma_max_frac_tau"] * tau["transition"]:
        unknown.append("transition_prediction_cannot_resolve_threshold")
    elif not unknown:
        eg = residuals(s, poses, g, calG["p"], sel, cam)
        er = residuals(s, poses, rr, calR["p"], sel, cam)
        hold_g = st_g[sel] < 0.7
        hold_r = st_r[sel] < 0.7
        e = np.where(hold_g & hold_r, np.minimum(eg, er), np.where(hold_g, eg, np.where(hold_r, er, np.minimum(eg, er))))
        k = S.G["smooth_frames"]
        sm = np.array([np.median(e[max(i - k // 2, 0): i + k // 2 + 1]) for i in range(len(e))])
        r["transition_max_px"] = float(sm.max())
        r["transition_z"] = r["transition_max_px"] / tau["transition"]
        if r["transition_z"] > 1:
            r["reasons"].append("transition_track_inconsistent")
    # C3: after the giver opens, the object must stop following it
    idx = seg["release"]
    r["release_coverage"] = coverage(s, idx)
    if calG["status"] != PASS:
        unknown.append("release_needs_giver_attachment")
    elif r["release_coverage"] < S.D["min_coverage"]:
        unknown.append("release_evidence")
    else:
        sel = idx[s.ok[idx]]
        r["release_median_px"] = float(np.median(residuals(s, poses, g, calG["p"], sel, cam)))
        if r["release_median_px"] < tau["release"]:
            r["reasons"].append("object_still_follows_giver")
    r["unknown"] = unknown
    r["status"] = FAIL if r["reasons"] else UNKNOWN if unknown else PASS
    return r


# ----------------------------------------------------------------------------------------------- all
def verify(s: Sample, fold: dict, poses: dict, cam=None) -> dict:
    """fold: frozen development-derived model {'model','tau_A','limits','tau'}; poses: lag-applied TCP poses of s.q."""
    cam = cam or g0.BODY_CAM
    out = {"name": s.name, "family": s.family, "label": s.label, "joints_ep": s.joints_ep, "video_ep": s.video_ep}
    out["A"] = comp_A(s, fold["model"], fold["tau_A"], fold["limits"])
    ev = command_events(s)
    if not ev["complete"]:
        out["B"] = {"status": FAIL, "reason": "phase_incomplete"}
        out["C"] = {"status": FAIL, "reasons": ["phase_incomplete"]}
        out["D"] = {"status": UNKNOWN, "coverage": {}}
    else:
        seg = segments(ev, len(s.uv))
        cov = {k: coverage(s, v) for k, v in seg.items()}
        out["D"] = {"coverage": cov, "status": PASS if all(cov[k] >= S.D["min_coverage"] for k in S.D["required_segments"]) else UNKNOWN,
                    "events": {k: ev[k] for k in ("giver", "g_close", "r_close", "g_open")}}
        calG = calibrate(s, poses, ev["giver"], seg["giver_calib"], cam)
        calR = calibrate(s, poses, ev["receiver"], seg["recv_calib"], cam)
        bg = predict_check(s, poses, ev["giver"], calG, seg["giver_pred"], cam, fold["tau"]["giver_pred"])
        br = predict_check(s, poses, ev["receiver"], calR, seg["recv_pred"], cam, fold["tau"]["recv_pred"])
        out["B"] = {"giver_calib": _strip(calG), "receiver_calib": _strip(calR), "giver": bg, "receiver": br,
                    "status": combine([calG["status"], calR["status"], bg["status"], br["status"]])}
        # C scores frames that belong to neither hold, so its attachments may use each hand's WHOLE hold
        # (calibration + prediction segments): still no fit on any frame C scores, with better observability.
        extG = calibrate(s, poses, ev["giver"], np.concatenate([seg["giver_calib"], seg["giver_pred"]]), cam)
        extR = calibrate(s, poses, ev["receiver"], np.concatenate([seg["recv_calib"], seg["recv_pred"]]), cam)
        out["C"] = comp_C(s, poses, ev, seg, extG, extR, cam, fold["tau"])
        out["C"]["attachments"] = {"giver": _strip(extG), "receiver": _strip(extR)}
    out["absolute_contact"] = {"status": UNKNOWN, "reason": "no measured object/pad geometry; calibration G1 pending"}
    out["status"] = combine([out["A"]["status"], out["B"]["status"], out["C"]["status"], out["D"]["status"]])
    return out


def _strip(c):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.items()}


def dev_statistics(s: Sample, fold_partial: dict, poses, cam=None) -> dict:
    """Raw development statistics used to freeze thresholds (computed on valid development originals only)."""
    cam = cam or g0.BODY_CAM
    ev = command_events(s)
    seg = segments(ev, len(s.uv))
    calG = calibrate(s, poses, ev["giver"], seg["giver_calib"], cam)
    calR = calibrate(s, poses, ev["receiver"], seg["recv_calib"], cam)
    out = {"calG": calG["status"], "calR": calR["status"]}
    for key, hand, cal, idx in (("giver_pred", ev["giver"], calG, seg["giver_pred"]), ("recv_pred", ev["receiver"], calR, seg["recv_pred"])):
        if cal["status"] == PASS:
            sel = idx[s.ok[idx]]
            out[key] = float(np.percentile(residuals(s, poses, hand, cal["p"], sel, cam), 90)) if len(sel) else None
    tmp = {"transition": float("inf"), "release": 0.0}  # statistics only: no decision threshold applies here
    extG = calibrate(s, poses, ev["giver"], np.concatenate([seg["giver_calib"], seg["giver_pred"]]), cam)
    extR = calibrate(s, poses, ev["receiver"], np.concatenate([seg["recv_calib"], seg["recv_pred"]]), cam)
    c = comp_C(s, poses, ev, seg, extG, extR, cam, tmp)
    out["transition"] = c.get("transition_max_px")
    out["release"] = c.get("release_median_px")
    return out
