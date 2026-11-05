"""Source segmentation and the three candidate generators (offline plans; NO physics).

A plan is what a pilot rollout would execute: per arm and step, a target grasp-frame pose (fed to the env's
limited-IK action exactly as Mimic does) and a binary gripper command, plus runtime rules that need simulator state
(they are applied by the runner, not here). The joint trajectory produced by offline IK is used only by the dry-run
checks.

A  geometry-only: an offline re-implementation of the hand-over Mimic config at IsaacLab @4aa5e026
   (`_handover_subtask_configs`): nearest source by can xy (nn_k=1); the right arm's approach-and-grasp segment is
   translated by T = can_new - can_src; everything after the right grasp, and the whole left arm, is replayed
   UNTRANSFORMED; smoothstep bridges of SEAM_INTERP (right: entry + grasp seam) and LEFT_INTERP (left entry).
B  contact-phase-conditioned: A + per-candidate sampled hand-over parameters (pass-point translation, receiver
   approach via-point, receiver grasp height/azimuth on the can, approach speed, receiver dwell, giver release delay).
   Open loop: the receiver grasp is placed relative to the can pose PREDICTED from the source, nothing is checked.
C  contact-constrained: B's sampler + (i) receiver grasp chosen in the can frame within the admissible grasp set and
   ended by a straight pre-grasp line, (ii) runtime re-anchoring of the receiver on the can pose predicted at the pass
   from the MEASURED giver attachment, (iii) joint limits / speed / jump, arm-arm clearance and table clearance enforced
   by shrinking the spatial parameters (<= 3 tries) or rejecting, (iv) runtime receiver-close gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .kin import Kin, arm_arm_clearance
from .sources import Episode


# ====================================================================== small SE(3) helpers
def trans(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


def rot_axis(axis, ang) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    return T


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def interp_pose(T0, T1, s) -> np.ndarray:
    """Position lerp + rotation slerp (via axis-angle of the relative rotation)."""
    import pinocchio as pin

    R = T0[:3, :3] @ pin.exp3(s * pin.log3(T0[:3, :3].T @ T1[:3, :3]))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
    return T


def bridge(T0, T1, n) -> np.ndarray:
    """n interpolated poses strictly after T0 and up to (excluding) T1, smoothstep easing (Mimic interpolation_easing)."""
    return np.stack([interp_pose(T0, T1, smoothstep((i + 1) / (n + 1))) for i in range(n)]) if n > 0 else np.zeros((0, 4, 4))


def smooth_positions(Ts: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average of positions with pinned endpoints (approximates Mimic smooth_eef_pose_segment)."""
    if window <= 1 or len(Ts) < 3:
        return Ts.copy()
    out = Ts.copy()
    P = Ts[:, :3, 3]
    h = window // 2
    for i in range(1, len(Ts) - 1):
        k = min(h, i, len(Ts) - 1 - i)
        out[i, :3, 3] = P[i - k:i + k + 1].mean(0)
    return out


# ====================================================================== source segmentation
@dataclass
class SourceInfo:
    ep: Episode
    TR: np.ndarray          # (T,4,4) right grasp-frame poses (FK of the measured state; = Mimic's 16-D source targets)
    TL: np.ndarray
    t_sig: int              # Mimic grasp_right boundary (right segment 0 ends here)
    t_rc: int               # right close command
    t_lc: int               # left close command
    t_rr: int               # right release command
    t_la: int               # left approach start (left grasp point leaves its rest by > 2 cm)
    t_ga: int               # giver arrival at the pass (stays within 1 cm of its pose at t_lc until t_lc)
    X_attach: np.ndarray    # can pose in the right grasp frame during the giver hold (4,4)
    attach_src: str

    def can_at(self, t: int, TR=None) -> np.ndarray:
        TR = self.TR if TR is None else TR
        return TR[t] @ self.X_attach


def nominal_attachment(pos=C.NOMINAL_ATTACH_POS, axis=C.NOMINAL_ATTACH_AXIS) -> np.ndarray:
    z = np.array(axis, float)
    z /= np.linalg.norm(z)
    x = np.array([0, 0, 1.0]) - z[2] * z          # can x: the grasp frame's approach axis, orthogonalised
    x /= np.linalg.norm(x)
    X = np.eye(4)
    X[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    X[:3, 3] = pos
    return X


def can_cyl(Y: np.ndarray):
    """(azimuth rad, axial m) of a grasp-frame origin expressed in the can frame (z = can axis)."""
    p = Y[:3, 3]
    return float(np.arctan2(p[1], p[0])), float(p[2])


def _first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else None


def segment_source(ep: Episode, kin: Kin) -> SourceInfo:
    TR, TL = kin.grasp_traj("R", ep.q["R"]), kin.grasp_traj("L", ep.q["L"])
    cR, cL = ep.grip_cmd["R"] < 0.5, ep.grip_cmd["L"] < 0.5
    t_rc = _first(cR)
    t_lc = _first(cL, t_rc)
    t_rr = _first(~cR, t_rc)
    if ep.signals is not None:
        t_sig = int(np.argmax(ep.signals["grasp_right"]))
    else:
        t_sig = int(float(ep.meta["events_csv"]["sig_grasp_right"]))
    pL = TL[:, :3, 3]
    t_la = _first(np.linalg.norm(pL - pL[0], axis=1) > 0.02, t_rc)
    pR = TR[:, :3, 3]
    far = np.linalg.norm(pR[:t_lc + 1] - pR[t_lc], axis=1) > 0.01
    t_ga = int(np.flatnonzero(far)[-1] + 1) if far.any() else t_sig
    if ep.can is not None:   # GT source: measured attachment after a 15-frame settle (post-step can t vs pre-step eef t+1)
        import pinocchio as pin
        t = t_sig + 15
        q = ep.can[t, 3:7]
        Tc = np.eye(4)
        Tc[:3, :3] = pin.Quaternion(*[float(v) for v in q]).toRotationMatrix()
        Tc[:3, 3] = ep.can[t, :3]
        X = np.linalg.inv(ep.eef["R"][min(t + 1, ep.T - 1)]) @ Tc
        src = "simulator GT can pose at grasp_right+15"
    else:                    # sim_10 carries no can pose: nominal attachment measured on the 40 local GT demos
        X = nominal_attachment()
        src = "nominal attachment (mean of 40 local GT demos, config.NOMINAL_ATTACH_*); std 1.35/0.29/0.68 cm"
    return SourceInfo(ep, TR, TL, t_sig, t_rc, t_lc, t_rr, t_la, t_ga, X, src)


# ====================================================================== plans
@dataclass
class Plan:
    cond: str
    idx: int
    source: str
    can0: np.ndarray
    params: dict
    target: dict                    # {"R": (T,4,4), "L": (T,4,4)}
    grip: dict                      # {"R": (T,), "L": (T,)} 1 = open, 0 = closed
    marks: dict                     # planned step indices (plan time)
    runtime: dict                   # rules the runner applies with simulator state
    can_pred: np.ndarray            # (T,4,4) planned can pose (prediction; on the pad before the grasp)
    status: str = "planned"         # planned | rejected
    notes: list = field(default_factory=list)
    joint: dict | None = None       # offline IK result, filled by checks

    @property
    def T(self) -> int:
        return len(self.target["R"])


def nearest_source(srcs: list[SourceInfo], can0) -> int:
    d = [np.linalg.norm(s.ep.can0[:2] - np.asarray(can0)[:2]) for s in srcs]
    return int(np.argmin(d))


def _assemble(segments):
    """segments: list of (poses (n,4,4), grip (n,)) concatenated."""
    P = np.concatenate([s[0] for s in segments if len(s[0])])
    G = np.concatenate([s[1] for s in segments if len(s[1])])
    return P, G


def _with_bridge(prev_pose, prev_grip, poses, grips, n):
    b = bridge(prev_pose, poses[0], n)
    return [(b, np.full(len(b), prev_grip)), (poses, grips)]


def plan_A(src: SourceInfo, can0, idx=0) -> Plan:
    s = src
    T = s.ep.T
    gR = (s.ep.grip_cmd["R"] >= 0.5).astype(float)
    gL = (s.ep.grip_cmd["L"] >= 0.5).astype(float)
    delta = np.r_[np.asarray(can0)[:2] - s.ep.can0[:2], 0.0]
    seg0 = smooth_positions(np.stack([trans(delta) @ P for P in s.TR[:s.t_sig + 1]]), C.SMOOTH_WINDOW)
    seg1 = smooth_positions(s.TR[s.t_sig + 1:], C.SMOOTH_WINDOW)
    rest_R, rest_L = s.TR[0], s.TL[0]
    right = _with_bridge(rest_R, gR[0], seg0, gR[:s.t_sig + 1], C.SEAM_INTERP)
    right += _with_bridge(seg0[-1], gR[s.t_sig], seg1, gR[s.t_sig + 1:], C.SEAM_INTERP)
    left = _with_bridge(rest_L, gL[0], smooth_positions(s.TL, C.SMOOTH_WINDOW), gL, C.LEFT_INTERP)
    PR, GR = _assemble(right)
    PL, GL = _assemble(left)
    n = max(len(PR), len(PL))
    PR, GR = _pad(PR, GR, n)
    PL, GL = _pad(PL, GL, n)
    offR1 = C.SEAM_INTERP             # right source index i -> plan step: i + 22 (seg0), i + 44 (seg1)
    offR2 = 2 * C.SEAM_INTERP
    offL = C.LEFT_INTERP
    marks = {"R_close": s.t_rc + offR1, "R_sig": s.t_sig + offR1, "giver_arrive": s.t_ga + offR2,
             "L_approach": s.t_la + offL, "L_close": s.t_lc + offL, "R_release_cmd": s.t_rr + offR2, "end": n - 1,
             "anchor": s.t_sig + 15 + offR2}
    marks["R_release_eff"] = max(marks["R_release_cmd"], marks["L_close"] + C.RELEASE_OVERLAP)
    can_pred = _can_prediction(PR, can0, marks, s, PL)
    runtime = {"giver_hold_until_taken": True, "release_overlap": C.RELEASE_OVERLAP,
               "receiver_close_gate": False, "receiver_reanchor": None}
    return Plan("A", idx, s.ep.name, np.asarray(can0, float), {"delta_xy": delta[:2].tolist()},
                {"R": PR, "L": PL}, {"R": GR, "L": GL}, marks, runtime, can_pred)


def _pad(P, G, n):
    if len(P) >= n:
        return P[:n], G[:n]
    k = n - len(P)
    return np.concatenate([P, np.repeat(P[-1:], k, 0)]), np.concatenate([G, np.repeat(G[-1:], k)])


def _can_prediction(PR, can0, marks, s: SourceInfo, PL=None):
    """Planned can pose (a prediction, not physics): on the pad until the right close; rigidly attached to the right
    target with the source attachment until the giver releases; then attached to the left target with the relative
    pose it had at the planned left close."""
    n = len(PR)
    out = np.repeat(trans(np.r_[np.asarray(can0)[:2], C.CAN_REST_Z])[None], n, 0)
    for t in range(marks["R_close"], n):
        out[t] = PR[t] @ s.X_attach
    if PL is not None:
        tl = min(marks["L_close"], n - 1)
        XL = np.linalg.inv(PL[tl]) @ out[tl]
        t_rel = min(n, marks.get("R_release_eff", marks["R_release_cmd"]))
        for t in range(t_rel, n):
            out[t] = PL[t] @ XL
    return out


def sample_params(rng: np.random.Generator) -> dict:
    r = C.B_RANGES
    u = lambda k: float(rng.uniform(*r[k]))   # noqa: E731
    return {"pass_dx": u("pass_dx"), "pass_dy": u("pass_dy"), "pass_dz": u("pass_dz"),
            "approach_bump": u("approach_bump"), "bump_angle": float(rng.uniform(0, 2 * np.pi)),
            "grasp_axial": u("grasp_axial"), "grasp_azimuth_deg": u("grasp_azimuth_deg"),
            "approach_speed": u("approach_speed"),
            "receiver_dwell": int(rng.integers(r["receiver_dwell"][0], r["receiver_dwell"][1] + 1)),
            "release_delay": int(rng.integers(r["release_delay"][0], r["release_delay"][1] + 1))}


def _ramp(n, a, b, up=True):
    """weights over n steps: 0 before a, smoothstep a..b, 1 after b (or the reverse)."""
    t = np.arange(n)
    w = smoothstep((t - a) / max(1, b - a))
    return w if up else 1 - w


def plan_phase(src: SourceInfo, can0, params: dict, cond: str, idx=0, scale: float = 1.0) -> Plan:
    """Conditions B and C share this builder (C adds the admissible-grasp construction and constraint enforcement)."""
    s = src
    p = dict(params)
    for k in ("pass_dx", "pass_dy", "pass_dz", "approach_bump", "grasp_axial", "grasp_azimuth_deg"):
        p[k] = p[k] * scale
    T = s.ep.T
    gR = (s.ep.grip_cmd["R"] >= 0.5).astype(float)
    gL = (s.ep.grip_cmd["L"] >= 0.5).astype(float)
    delta = np.r_[np.asarray(can0)[:2] - s.ep.can0[:2], 0.0]
    dpass = np.array([p["pass_dx"], p["pass_dy"], p["pass_dz"]])

    # ---------------- receiver timing first (C needs it to synchronise the giver)
    L = s.TL.copy()
    gl = gL.copy()
    t_la, t_lc = s.t_la, s.t_lc
    can_src_pass = s.can_at(t_lc)                       # source can pose at the take (prediction)
    n_ap = t_lc - t_la + 1
    n_new = max(5, int(round(n_ap / p["approach_speed"])))
    dw = int(p["receiver_dwell"])
    d = int(p["release_delay"])
    giver_arrive_plan = s.t_ga + 2 * C.SEAM_INTERP
    extra = {}
    if cond == "C":
        # synchronisation 1: the receiver closes only after the giver has reached the pass (plan time)
        close_plan = t_la + n_new - 1 + dw + C.LEFT_INTERP
        if close_plan < giver_arrive_plan + 2:
            extra["sync_dwell_added"] = int(giver_arrive_plan + 2 - close_plan)
            dw += extra["sync_dwell_added"]
        close_plan = t_la + n_new - 1 + dw + C.LEFT_INTERP
        # synchronisation 2: the giver holds still at the pass (closed) until the receiver has closed and the release
        # overlap has elapsed - otherwise the replayed withdraw drags the clamped can away from a late receiver
        need = close_plan + C.RELEASE_OVERLAP + d - (s.t_rr + 2 * C.SEAM_INTERP)
        if need > d:
            extra["giver_wait_added"] = int(need - d)
            d = int(need)
        # C references the can where the giver will actually be at the planned take (giver source index g)
        g = int(np.clip(close_plan - 2 * C.SEAM_INTERP, s.t_ga, s.t_rr))
        extra["giver_source_index_at_take"] = g
        can_ref = s.can_at(g)
    else:
        can_ref = can_src_pass

    # ---------------- right (giver), in source time
    R0 = np.stack([trans(delta) @ P for P in s.TR[:s.t_sig + 1]])
    R1 = s.TR[s.t_sig + 1:].copy()
    idx1 = np.arange(s.t_sig + 1, T)
    w_up = smoothstep((idx1 - (s.t_sig + 1)) / max(1, s.t_ga - s.t_sig - 1))
    w_dn = 1 - smoothstep((idx1 - s.t_rr) / 30.0)
    w = np.minimum(w_up, w_dn)
    R1 = np.stack([trans(wi * dpass) @ P for wi, P in zip(w, R1)])
    gR1 = gR[s.t_sig + 1:].copy()
    # giver release delay (B: sampled; C: sampled + synchronisation): hold position, closed, at the source release
    k = s.t_rr - (s.t_sig + 1)
    R1 = np.concatenate([R1[:k], np.repeat(R1[k:k + 1], d, 0), R1[k:]])
    gR1 = np.concatenate([gR1[:k], np.zeros(d), gR1[k:]])

    axis = can_src_pass[:3, 2]
    Rcorr = rot_axis(axis, np.radians(p["grasp_azimuth_deg"]))
    # receiver grasp re-pose: rotate about the can axis through the can centre, slide along it, then the pass shift
    G = trans(dpass) @ trans(can_src_pass[:3, 3]) @ Rcorr @ trans(-can_src_pass[:3, 3]) @ trans(p["grasp_axial"] * axis)
    L_take = G @ L[t_lc]                                # B: the source receiver pose moved by G (open loop)
    if cond == "C":
        # C: admissible receiver grasp on the predicted can. Keep the source receiver's orientation (rotated about the
        # can axis by the sampled azimuth) but place the hand so the can centre sits at the GT-measured capture point
        # of the jaws (config.NOMINAL_RECEIVER_POS) - between the open pads by construction - and keep the grasp on
        # the can body and clear of the giver's grasp along the axis.
        Y = rot_axis([0, 0, 1], np.radians(p["grasp_azimuth_deg"])) @ (np.linalg.inv(can_ref) @ L[t_lc])
        v = -Y[:3, :3].T @ Y[:3, 3]                     # can centre in the receiver grasp frame
        v[1], v[2] = C.NOMINAL_RECEIVER_POS[1], C.NOMINAL_RECEIVER_POS[2]
        Y[:3, 3] = -Y[:3, :3] @ v
        h_lim = C.CAN_HALF_HEIGHT - 0.025
        h_R = (np.linalg.inv(can_ref) @ s.TR[g])[2, 3]
        h = float(np.clip(Y[2, 3] + p["grasp_axial"], -h_lim, h_lim))
        if abs(h - h_R) < C.MIN_GRASP_SEPARATION:      # keep the two grasps apart along the can
            h = float(np.clip(h_R + np.sign(h - h_R + 1e-9) * C.MIN_GRASP_SEPARATION, -h_lim, h_lim))
        Y[2, 3] = h
        L_take = trans(dpass) @ can_ref @ Y
        G = L_take @ np.linalg.inv(L[t_lc])             # the world correction the approach blends toward
        extra["receiver_axial_on_can"] = h
    ap_dir = L[t_lc][:3, 3] - L[t_la][:3, 3]
    ap_dir /= max(1e-9, np.linalg.norm(ap_dir))
    perp = np.cross(ap_dir, [0, 0, 1.0])
    perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else np.array([1.0, 0, 0])
    bdir = np.cos(p["bump_angle"]) * perp + np.sin(p["bump_angle"]) * np.cross(perp, ap_dir)
    m = max(4, int(round(0.35 * n_ap))) if cond == "C" else 0   # C: straight pre-grasp tail
    n_head = n_ap - m
    app = []
    for i in range(n_head):
        sg = i / max(1, n_ap - 1)
        sb = i / max(1, n_head - 1)
        Pn = interp_pose(L[t_la + i], G @ L[t_la + i], smoothstep(sg))   # source path -> moved by G
        Pn[:3, 3] += p["approach_bump"] * np.sin(np.pi * sb) * bdir
        app.append(Pn)
    if cond == "C":
        pre = L_take @ trans([0, 0, -C.C_PREGRASP])     # back off along the grasp frame's approach axis (z)
        start = app[-1] if app else L[t_la]
        n_lead = m // 2
        app += list(bridge(start, pre, n_lead - 1)) + [pre]
        n_line = m - n_lead
        app += [interp_pose(pre, L_take, (i + 1) / n_line) for i in range(n_line)]
    app = np.stack(app)
    app[-1] = L_take
    # time-warp the approach
    xs = np.linspace(0, n_ap - 1, n_new)
    app = np.stack([interp_pose(app[int(x)], app[min(n_ap - 1, int(x) + 1)], x - int(x)) for x in xs])
    app[-1] = L_take
    # carry: keep the hand-relative change constant (the can moves rigidly with the receiving hand)
    Hrel = np.linalg.inv(L[t_lc]) @ L_take
    carry = np.stack([P @ Hrel for P in L[t_lc + 1:]])
    Lseq = np.concatenate([L[:t_la], app, np.repeat(L_take[None], dw, 0), carry])
    gseq = np.concatenate([gl[:t_la], np.ones(len(app) - 1 + dw), [0.0], gl[t_lc + 1:]])
    close_idx = t_la + len(app) - 1 + dw               # first closed receiver step (shifted source time)
    assert len(Lseq) == len(gseq)

    # ---------------- assemble with the same bridges as A
    R0s, R1s = smooth_positions(R0, C.SMOOTH_WINDOW), smooth_positions(R1, C.SMOOTH_WINDOW)
    right = _with_bridge(s.TR[0], gR[0], R0s, gR[:s.t_sig + 1], C.SEAM_INTERP)
    right += _with_bridge(R0s[-1], gR[s.t_sig], R1s, gR1, C.SEAM_INTERP)
    left = _with_bridge(s.TL[0], gL[0], smooth_positions(Lseq, C.SMOOTH_WINDOW), gseq, C.LEFT_INTERP)
    PR, GR = _assemble(right)
    PL, GL = _assemble(left)
    n = max(len(PR), len(PL))
    PR, GR = _pad(PR, GR, n)
    PL, GL = _pad(PL, GL, n)
    offR2 = 2 * C.SEAM_INTERP
    offL = C.LEFT_INTERP
    marks = {"R_close": s.t_rc + C.SEAM_INTERP, "R_sig": s.t_sig + C.SEAM_INTERP, "giver_arrive": s.t_ga + offR2,
             "L_approach": s.t_la + offL, "L_close": close_idx + offL, "R_release_cmd": s.t_rr + offR2 + d,
             "end": n - 1, "anchor": s.t_sig + 15 + offR2}
    marks["R_release_eff"] = max(marks["R_release_cmd"], marks["L_close"] + C.RELEASE_OVERLAP + d)
    can_pred = _can_prediction(PR, can0, marks, s, PL)
    runtime = {"giver_hold_until_taken": True, "release_overlap": C.RELEASE_OVERLAP + int(p["release_delay"]),
               "receiver_close_gate": cond == "C", "receiver_reanchor": None}
    if cond == "C":
        runtime["receiver_reanchor"] = {
            "anchor_step": marks["anchor"], "pass_step": marks["L_close"],
            "rule": "at anchor_step measure X = inv(T_R_grasp) @ T_can (sim GT); predicted can at the pass = "
                    "target_R[pass_step] @ X; apply ONE rigid correction inv-plan-prediction to every left target from "
                    "L_approach on, blended in over the approach (smoothstep)",
            "plan_can_at_pass": can_pred[marks["L_close"]].tolist()}
        runtime["close_gate_wait"] = C.C_CLOSE_GATE_WAIT
    out = Plan(cond, idx, s.ep.name, np.asarray(can0, float),
               {**p, "scale": scale, "delta_xy": delta[:2].tolist(), **extra},
               {"R": PR, "L": PL}, {"R": GR, "L": GL}, marks, runtime, can_pred)
    out.notes.append(f"receiver grasp in can frame: axial {p['grasp_axial']:+.3f} m, azimuth {p['grasp_azimuth_deg']:+.1f} deg "
                     f"relative to source; pass shift {np.round(dpass, 3).tolist()}")
    return out
