"""Per-episode preprocessing for v3: lid tracks for all 10 source episodes, then the (unchanged) v2 reconstruction.

* Lid track: the Gate 1 image-only HSV tracker initialised from the human-verified frame-0 click of the 50-episode
  pool annotation file (pool episodes 0-9 == real_10 episodes 0-9, verified by state fingerprints). For episodes
  0/4/8 this reproduces the Gate 1 track files; v3 writes its own copies under outputs/cache.
* Reconstruction: sfa2.recon.build (joint closure-constrained attachment, keyed SAM2 cache, plate) - v2 code files are
  not modified; its lid-track path is pointed at the v3 cache, and three of its fitting functions are wrapped so that
  an episode whose receiver carry has no tracked lid (episode 6: 0/67 frames) falls back to a closure-only receiver
  attachment, flagged in rec.info (the wrappers are pass-through whenever image evidence exists).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V2 = HERE.parent / "v2"
sys.path.insert(0, str(V2))
from sfa2 import recon as recon2  # noqa: E402
from sfa2.recon import REPO, data, rb  # noqa: E402

sys.path.insert(0, str(REPO / "research/contact_aware_augmentation/gate1"))
ANN = REPO / "research/dataset_diagnostics/annotations/real_v00.csv"
CACHE = HERE / "outputs/cache"


def clicks() -> dict:
    return {int(r["episode_index"]): (float(r["u"]), float(r["v"])) for r in csv.DictReader(open(ANN))
            if r["dataset"] == "real_v00_pool" and r["verified"] == "1"}


def lid_track_path(e: int) -> Path:
    return CACHE / f"ep{e}" / f"lid_track_ep{e}.npz"


def ensure_lid_track(e: int, frames=None) -> dict:
    p = lid_track_path(e)
    if p.exists():
        return {"cached": True, "s": 0.0}
    import lid_tracker as lt
    t0 = time.time()
    if frames is None:
        frames = data.load(e, cams=(data.CAMS[0],)).frames[data.CAMS[0]]
    tr = lt.track(frames, clicks()[e])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, uv=tr.uv, centroid=tr.centroid, area=tr.area, status=np.array(tr.status),
                        flags=np.array(["|".join(f) for f in tr.flags]), init_click=np.array(clicks()[e]))
    return {"cached": False, "s": time.time() - t0}


MIN_FRAMES = 8
_fit_point, _fit_axis, _joint_fit = recon2.rc.fit_point, recon2.rc.fit_axis, recon2.joint_fit
FALLBACK = {}


def _fit_point_safe(robot, cam, g, side, frames, uv, lag):
    if len(frames) >= MIN_FRAMES:
        return _fit_point(robot, cam, g, side, frames, uv, lag)
    nan = float("nan")
    return {"p": np.array([0.0, 0.0, 0.05]), "rms_px": nan, "median_px": nan, "p90_px": nan, "n": len(frames),
            "std_m": [nan] * 3, "no_image_evidence": True}


def _fit_axis_safe(robot, cam, g, side, p_lid, frames, masks, lag, H=480, W=640):
    if len(frames) >= 1:
        return _fit_axis(robot, cam, g, side, p_lid, frames, masks, lag, H, W)
    return {"axis": np.array([0.0, 0.0, 1.0]), "mean_iou": float("nan")}


def _joint_fit_safe(robot, cam, g, giv, rec, dual, uv, lag, init, w_closure=1000.0):
    """v2 joint fit; if the receiver carry has < MIN_FRAMES tracked lid frames, the receiver lid point is taken from
    closure alone (giver attachment carried through the dual-closed frames) and flagged."""
    if len(rec) >= MIN_FRAMES:
        return _joint_fit(robot, cam, g, giv, rec, dual, uv, lag, init, w_closure)
    pR = _fit_point(robot, cam, g, "R", giv, uv, lag)
    ps = []
    for t in dual:
        robot.fk(g["R"][t], g["L"][t])
        TR, TL = robot.frame(robot.tcp["R"]), robot.frame(robot.tcp["L"])
        w = TR[:3, :3] @ pR["p"] + TR[:3, 3]
        ps.append(TL[:3, :3].T @ (w - TL[:3, 3]))
    pL = np.mean(ps, axis=0)
    FALLBACK["receiver_from_closure_only"] = True
    return {"p_R": pR["p"], "p_L": pL, "rms_px_R": pR["rms_px"], "rms_px_L": float("nan"), "median_px_R": pR["median_px"],
            "median_px_L": float("nan"), "closure_mm_median": float(1000 * np.median([np.linalg.norm(x - pL) for x in ps])),
            "closure_mm_max": float(1000 * max(np.linalg.norm(x - pL) for x in ps)), "n_R": len(giv), "n_L": len(rec),
            "n_dual": len(dual), "receiver_from_closure_only": True}


recon2.rc.fit_point, recon2.rc.fit_axis, recon2.joint_fit = _fit_point_safe, _fit_axis_safe, _joint_fit_safe


def build(e: int, device="cuda:0", with_plate=True, log=print):
    FALLBACK.clear()
    recon2.LID_TRACK = CACHE / "ep{e}" / "lid_track_ep{e}.npz"
    lt = ensure_lid_track(e)
    rec = recon2.build(e, CACHE / f"ep{e}", device=device, with_plate=with_plate, log=log)
    rec.info["lid_track"] = lt
    rec.info["init_click"] = clicks()[e]
    rec.info["receiver_from_closure_only"] = bool(FALLBACK.get("receiver_from_closure_only", False))
    return rec
