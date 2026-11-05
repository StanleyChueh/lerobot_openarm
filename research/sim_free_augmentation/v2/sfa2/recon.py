"""Per-episode reconstruction for v2 (body_cam, real Pringles episode). No simulator, no physics.

Reuses the v1 prototype modules (`research/sim_free_augmentation/sfa`, commit e8abc26) for data access, FK/IK, SAM2
segmentation, RAFT and rendering, and changes three things:

1. **Joint, closure-constrained can attachment.** v1 fitted the lid point in the giver TCP frame and, separately, in
   the receiver TCP frame. Evaluated in the window where both grippers are closed, those two independent estimates
   put the can 4.5 cm / 38 deg (axis) apart on episode 0, which is a noise floor larger than any contact tolerance.
   v2 fits both lid points jointly: image reprojection on the giver-hold and receiver-hold frames plus a closure
   residual T_R(t) p_R = T_L(t) p_L on the dual-closed frames; the receiver's can frame is then defined by closure
   (one rigid can across the transfer). Both fits and both closure errors are reported.
2. **Joint axis selection.** The can axis is chosen once (giver frame) by silhouette IoU on giver AND receiver frames,
   the receiver axis following from closure, instead of two independent searches.
3. **Input-keyed caches.** SAM2 masks are cached under a hash of every input that determines them.

3D is only claimed where a hand holds the can (structure from known hand motion); the standing can is placed on its
observed lid ray at the lid height recovered from the grasp.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

V1 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V1))
from sfa import data, recover as rc, render as rd, robot as rb, segment as sg  # noqa: E402

from . import se3  # noqa: E402

REPO = rb.REPO
LID_TRACK = REPO / "research/contact_aware_augmentation/outputs/gate1/tracks/lid_track_ep{e}.npz"
CLOSED = 0.7          # gripper openness (state) below which a gripper counts as closed (state-derived, not force)


def frame_from_axis(p, a) -> np.ndarray:
    a = np.asarray(a, float) / np.linalg.norm(a)
    u = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
    u /= np.linalg.norm(u)
    X = np.eye(4)
    X[:3, :3] = np.stack([u, np.cross(a, u), a], 1)
    X[:3, 3] = p
    return X


def video_q(Q: dict, lag: float, T: int) -> dict:
    return {s: np.stack([rc.interp_q(Q[s], t + lag) for t in range(T)]) for s in "RL"}


def tcp_poses(robot, g, side, times) -> list:
    return [rc.tcp_pose(robot, g, side, t) for t in times]


def axis_iou(cam, Ts, p, a, frames, masks, H=480, W=640) -> float:
    P0 = rc.cylinder_points(a, 16, 4) + p
    ious = []
    for T, t in zip(Ts, frames):
        uv, z = cam.project((T[:3, :3] @ P0.T).T + T[:3, 3])
        if (z < 0.05).any():
            ious.append(0.0)
            continue
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
        gt = masks[t] > 0
        ious.append((m.astype(bool) & gt).sum() / max(1, (m.astype(bool) | gt).sum()))
    return float(np.mean(ious)) if ious else 0.0


def dual_closed_frames(g, thr=CLOSED) -> list:
    both = (g["gR"] < thr) & (g["gL"] < thr)
    return [int(t) for t in np.flatnonzero(both)]


def joint_fit(robot, cam, g, giv, rec, dual, uv, lag, init, w_closure=1000.0) -> dict:
    """lid points p_R, p_L (TCP frames) from reprojection on giver/receiver frames + 3D closure on dual frames."""
    TR = tcp_poses(robot, g, "R", [t + lag for t in giv])
    TL = tcp_poses(robot, g, "L", [t + lag for t in rec])
    DR = tcp_poses(robot, g, "R", dual)
    DL = tcp_poses(robot, g, "L", dual)

    def proj(Ts, p):
        return cam.project(np.stack([T[:3, :3] @ p + T[:3, 3] for T in Ts]))[0]

    def res(x):
        pR, pL = x[:3], x[3:]
        r = [(proj(TR, pR) - uv[giv]).ravel(), (proj(TL, pL) - uv[rec]).ravel()]
        c = np.stack([(A[:3, :3] @ pR + A[:3, 3]) - (B[:3, :3] @ pL + B[:3, 3]) for A, B in zip(DR, DL)])
        r.append(w_closure * c.ravel())
        return np.concatenate(r)

    x0 = np.r_[init["R"], init["L"]]
    r = least_squares(res, x0=x0, loss="soft_l1", f_scale=5.0)
    pR, pL = r.x[:3], r.x[3:]
    eR = np.linalg.norm(proj(TR, pR) - uv[giv], axis=1)
    eL = np.linalg.norm(proj(TL, pL) - uv[rec], axis=1)
    cl = [np.linalg.norm((A[:3, :3] @ pR + A[:3, 3]) - (B[:3, :3] @ pL + B[:3, 3])) for A, B in zip(DR, DL)]
    return {"p_R": pR, "p_L": pL, "rms_px_R": float(np.sqrt(np.mean(eR ** 2))), "rms_px_L": float(np.sqrt(np.mean(eL ** 2))),
            "median_px_R": float(np.median(eR)), "median_px_L": float(np.median(eL)),
            "closure_mm_median": float(1000 * np.median(cl)), "closure_mm_max": float(1000 * np.max(cl)),
            "n_R": len(giv), "n_L": len(rec), "n_dual": len(dual)}


def closure_report(robot, g, X, dual) -> dict:
    tr, tilt, spin = [], [], []
    for t in dual:
        robot.fk(g["R"][t], g["L"][t])
        cR = robot.frame(robot.tcp["R"]) @ X["R"]
        cL = robot.frame(robot.tcp["L"]) @ X["L"]
        tr.append(1000 * np.linalg.norm(cR[:3, 3] - cL[:3, 3]))
        tilt.append(se3.axis_tilt_deg(cR[:3, 2], cL[:3, 2]))
        spin.append(abs(se3.spin_deg(cR[:3, :3], cL[:3, :3])))
    return {"translation_mm_median": float(np.median(tr)), "translation_mm_max": float(np.max(tr)),
            "axis_tilt_deg_median": float(np.median(tilt)), "spin_deg_median": float(np.median(spin))}


def sam2_key(F, gv, grip, lid_uv, lid_ok, keyframes, cam) -> str:
    h = hashlib.sha256()
    h.update(sg.MODEL.encode())
    h.update(hashlib.sha256(F.tobytes()).digest())
    for s in "RL":
        h.update(np.round(gv[s], 6).tobytes())
        h.update(np.round(grip[s], 6).tobytes())
    h.update(np.nan_to_num(np.round(lid_uv, 3)).tobytes())
    h.update(lid_ok.astype(np.uint8).tobytes())
    h.update(json.dumps(list(map(int, keyframes))).encode())
    h.update(json.dumps(cam.cfg, sort_keys=True).encode())
    h.update(Path(sg.__file__).read_bytes())
    return h.hexdigest()


@dataclass
class Recon:
    e: int
    fps: float
    T: int
    F: np.ndarray
    state: np.ndarray
    action: np.ndarray
    g: dict
    ga: dict
    ev: dict
    lead: int
    lid_uv: np.ndarray
    lid_ok: np.ndarray
    lag: float
    X: dict
    X_indep: dict
    table: np.ndarray
    z_table: float
    masks: dict
    plate: np.ndarray
    gv: dict
    grip_v: dict
    info: dict = field(default_factory=dict)

    def source_can(self, robot, t_state: float, X=None) -> np.ndarray:
        """source can pose at (fractional) state time under the joint model: table / giver / receiver."""
        X = self.X if X is None else X
        h = self.source_holder(int(round(t_state)))
        if h == "table":
            return self.table
        return rc.tcp_pose(robot, self.g, h, t_state) @ X[h]

    def source_holder(self, t: int) -> str:
        ev = self.ev
        return "table" if t < ev["right_close"] + 8 else ("R" if t < ev["right_open"] else "L")


def build(e: int, cache_dir: Path, device="cuda:0", keyframe_step=15, log=print, with_plate=True) -> Recon:
    t0 = time.time()
    times = {}
    ep = data.load(e, cams=(data.CAMS[0],))
    F = ep.frames[data.CAMS[0]]
    T = len(F)
    robot, cam = rb.Robot(), rb.Camera()
    g, ga, ev = rb.Robot.split(ep.state), rb.Robot.split(ep.action), ep.events
    lead = data.action_lead(ep.state, ep.action)
    z = np.load(str(LID_TRACK).format(e=e))
    lid_uv, lid_ok = z["uv"], z["status"] == "ok"
    times["load"] = time.time() - t0

    giv = [t for t in range(ev["right_close"] + 10, ev["left_close"] - 3) if lid_ok[t]]
    rec = [t for t in range(ev["right_open"] + 8, T) if lid_ok[t]]
    lagfit = rc.estimate_lag(robot, cam, g, "R", giv, lid_uv)
    lag = lagfit["lag"]
    ind = {"R": rc.fit_point(robot, cam, g, "R", giv, lid_uv, lag), "L": rc.fit_point(robot, cam, g, "L", rec, lid_uv, lag)}
    gv = video_q(g, lag, T)
    grip_v = {s: np.array([np.interp(t + lag, np.arange(T), g["g" + s]) for t in range(T)]) for s in "RL"}
    times["lag_fit"] = time.time() - t0 - times["load"]

    keyframes = list(range(0, T, keyframe_step)) + [T - 1]
    key = sam2_key(F, gv, grip_v, lid_uv, lid_ok, keyframes, cam)
    mpath = cache_dir / f"sam2_{key[:16]}.npz"
    t1 = time.time()
    if mpath.exists():
        masks = dict(np.load(mpath))
        sam_cached = True
    else:
        masks, prompts = sg.segment(F, robot, cam, gv, lid_uv, lid_ok, keyframes, device, grip=grip_v)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mpath, **masks)
        (cache_dir / f"sam2_{key[:16]}_prompts.json").write_text(json.dumps(prompts, default=float))
        sam_cached = False
    times["sam2"] = time.time() - t1

    # ---- axes: v1 independent (reported) and v2 joint
    t1 = time.time()
    sel = lambda fr: [t for t in fr[:: max(1, len(fr) // 12)] if masks["can"][t].any()]  # noqa: E731
    for s, fr in (("R", giv), ("L", rec)):
        ind[s].update(rc.fit_axis(robot, cam, g, s, ind[s]["p"], sel(fr), masks["can"], lag))
    X_indep = {s: frame_from_axis(ind[s]["p"], ind[s]["axis"]) for s in "RL"}
    dual = dual_closed_frames(g)
    jf = joint_fit(robot, cam, g, giv, rec, dual, lid_uv, lag, {"R": ind["R"]["p"], "L": ind["L"]["p"]})
    tm = dual[len(dual) // 2]
    robot.fk(g["R"][tm], g["L"][tm])
    R_rel = robot.frame(robot.tcp["L"])[:3, :3].T @ robot.frame(robot.tcp["R"])[:3, :3]   # R-frame -> L-frame
    fR, fL = sel(giv), sel(rec)
    TsR = tcp_poses(robot, g, "R", [t + lag for t in fR])
    TsL = tcp_poses(robot, g, "L", [t + lag for t in fL])
    best = (-1.0, None, 0.0, 0.0)
    for a in rc.fibonacci_dirs(300):
        iR = axis_iou(cam, TsR, jf["p_R"], a, fR, masks["can"])
        if iR < best[0] - 1.0:
            continue
        iL = axis_iou(cam, TsL, jf["p_L"], R_rel @ a, fL, masks["can"])
        if iR + iL > best[0]:
            best = (iR + iL, a, iR, iL)
    XR = frame_from_axis(jf["p_R"], best[1])
    XL = np.eye(4)
    XL[:3, :3] = R_rel @ XR[:3, :3]
    XL[:3, 3] = jf["p_L"]
    X = {"R": XR, "L": XL}
    times["attachment"] = time.time() - t1

    # ---- table phase (standing can): lid ray at the lid height recovered at the grasp
    tg = ev["right_close"] + 8
    T_att = rc.tcp_pose(robot, g, "R", tg) @ X["R"]
    pre = [t for t in range(0, ev["right_close"]) if lid_ok[t]]
    u = np.median(lid_uv[pre], axis=0)
    ray = cam.R @ np.linalg.solve(cam.K, np.r_[u, 1.0])
    s_ = (T_att[2, 3] - cam.t[2]) / ray[2]
    table = T_att.copy()
    table[:3, 3] = cam.t + s_ * ray
    z_table = float(table[2, 3] - rc.CAN_HEIGHT)          # declared prior: can stands on the table, 23 cm tall

    t1 = time.time()
    occ = (masks["R"] > 0) | (masks["L"] > 0) | (masks["can"] > 0)
    plate, never = rd.background_plate(F, occ) if with_plate else (np.zeros_like(F[0]), np.zeros(F.shape[1:3], bool))
    times["plate"] = time.time() - t1

    info = {"episode": e, "T": T, "fps": ep.fps, "events": ev, "action_lead": lead, "lag_frames": lag,
            "lag_rms_by_lag": lagfit["rms_by_lag"], "lid_ok_frac": float(lid_ok.mean()),
            "independent_fit": {s: {k: ind[s][k] for k in ("p", "rms_px", "n", "axis", "mean_iou")} for s in "RL"},
            "independent_closure": closure_report(robot, g, X_indep, dual),
            "joint_fit": jf, "joint_axis_iou": {"R": best[2], "L": best[3]},
            "joint_closure": closure_report(robot, g, X, dual), "dual_closed_frames": dual,
            "table_can_axis_tilt_from_vertical_deg": se3.axis_tilt_deg(table[:3, 2], [0, 0, -1.0]),
            "z_table_m": z_table, "plate_never_free_frac": float(never.mean()), "sam2_cache": mpath.name,
            "sam2_cached": sam_cached, "sam2_key": key, "keyframes": keyframes, "runtime_s": times,
            "X": {s: X[s].tolist() for s in "RL"}, "X_independent": {s: X_indep[s].tolist() for s in "RL"},
            "table_pose": table.tolist()}
    log(f"[recon ep{e}] lag {lag} | closure indep {info['independent_closure']} -> joint {info['joint_closure']} "
        f"| rms R {jf['rms_px_R']:.2f} L {jf['rms_px_L']:.2f} px | {time.time() - t0:.1f} s")
    return Recon(e, ep.fps, T, F, ep.state, ep.action, g, ga, ev, lead["k_best"], lid_uv, lid_ok, lag, X, X_indep,
                 table, z_table, masks, plate, gv, grip_v, info)
