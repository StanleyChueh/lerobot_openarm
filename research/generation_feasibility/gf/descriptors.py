"""Hand-over descriptors and novelty relative to the source demonstrations.

The same functions are applied to (i) source demos, (ii) offline plans (predicted can pose) and, in the pilot,
(iii) executed rollouts with the simulator's can pose. Only (iii) can support a claim that a NEW successful hand-over
was produced; (ii) only shows what the generator asks for.

Descriptor families (each compared with its own scale):
  pass   : can centre at the receiver close (world, m)
  grasp  : receiver grasp relative to the can and to the giver grasp: axial separation (m) and relative azimuth (deg)
  path   : receiver grasp-point path from its approach start to 30 steps after the close, resampled to 40 points (m RMS)
  timing : receiver close minus giver arrival, and the dual-contact overlap (steps)
"""

from __future__ import annotations

import numpy as np


def _azimuth(v, axis):
    """Signed angle of v projected on the plane normal to axis (reference: world x projected)."""
    axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0, 0]) - axis[0] * axis
    if np.linalg.norm(ref) < 1e-6:
        ref = np.array([0, 1.0, 0]) - axis[1] * axis
    ref /= np.linalg.norm(ref)
    v = v - (v @ axis) * axis
    return np.degrees(np.arctan2(np.cross(ref, v) @ axis, ref @ v))


def handover_descriptor(TR, TL, can, t_lc, t_ga, t_la, t_release, n_path=40, t_close=None) -> dict:
    """TR/TL (T,4,4) grasp-frame poses, can (T,4,4) can poses, event indices of the same timeline.
    t_lc: the pass (can and grasps are read here); t_close: the receiver's close (path end = t_close + 30)."""
    t_close = t_lc if t_close is None else t_close
    Tc = can[t_lc]
    axis = Tc[:3, 2]
    ctr = Tc[:3, 3]
    rL = TL[t_lc][:3, 3] - ctr
    rR = TR[t_lc][:3, 3] - ctr
    daz = (_azimuth(rL, axis) - _azimuth(rR, axis)) % 360   # [0, 360): the sources sit near 180 (opposite sides)
    end = min(len(TL) - 1, t_close + 30)
    P = TL[t_la:end + 1, :3, 3]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if s[-1] < 1e-9:
        path = np.repeat(P[:1], n_path, 0)
    else:
        g = np.linspace(0, s[-1], n_path)
        path = np.stack([np.interp(g, s, P[:, k]) for k in range(3)], 1)
    return {"pass": ctr.copy(), "axial_L": float(rL @ axis), "axial_R": float(rR @ axis),
            "axial_sep": float((rL - rR) @ axis), "rel_azimuth_deg": float(daz),
            "radial_L": float(np.linalg.norm(rL - (rL @ axis) * axis)), "path": path,
            "t_close_minus_arrive": int(t_close - t_ga), "overlap": int(t_release - t_close)}


def _dist(a, b, fam):
    if fam == "pass":
        return float(np.linalg.norm(a["pass"] - b["pass"]))
    if fam == "grasp":   # 1 cm axial ~ 10 deg azimuth
        return float(np.hypot((a["axial_sep"] - b["axial_sep"]) / 0.01,
                              (a["rel_azimuth_deg"] - b["rel_azimuth_deg"]) / 10.0))
    if fam == "path":
        return float(np.sqrt(np.mean(np.sum((a["path"] - b["path"]) ** 2, 1))))
    if fam == "timing":
        return float(np.hypot(a["t_close_minus_arrive"] - b["t_close_minus_arrive"], a["overlap"] - b["overlap"]))
    raise KeyError(fam)


FAMILIES = ("pass", "grasp", "path", "timing")


def source_scale(src_desc: list[dict]) -> dict:
    """Median nearest-neighbour distance among the sources (leave-one-out), per family."""
    out = {}
    for fam in FAMILIES:
        nn = [min(_dist(a, b, fam) for j, b in enumerate(src_desc) if j != i) for i, a in enumerate(src_desc)]
        out[fam] = float(np.median(nn))
    return out


def novelty(desc: dict, src_desc: list[dict], scale: dict) -> dict:
    out = {}
    for fam in FAMILIES:
        d = min(_dist(desc, b, fam) for b in src_desc)
        out[fam] = {"nn_dist": d, "ratio": d / scale[fam] if scale[fam] > 0 else float("inf")}
    return out


def envelope(src_desc: list[dict]) -> dict:
    ax = [d["axial_sep"] for d in src_desc]
    az = [d["rel_azimuth_deg"] for d in src_desc]
    return {"axial_sep": (min(ax), max(ax)), "rel_azimuth_deg": (min(az), max(az)),
            "pass": np.stack([d["pass"] for d in src_desc])}


def outside_envelope(desc: dict, env: dict, pass_margin=0.03, axial_margin=0.01, az_margin=10.0) -> dict:
    """The pre-registered NOVEL test (config.CRITERIA S2)."""
    pass_far = bool(np.min(np.linalg.norm(env["pass"] - desc["pass"], axis=1)) >= pass_margin)
    lo, hi = env["axial_sep"]
    ax_out = bool(desc["axial_sep"] < lo - axial_margin or desc["axial_sep"] > hi + axial_margin)
    lo, hi = env["rel_azimuth_deg"]
    az_out = bool(desc["rel_azimuth_deg"] < lo - az_margin or desc["rel_azimuth_deg"] > hi + az_margin)
    return {"pass_far": pass_far, "axial_out": ax_out, "azimuth_out": az_out, "novel": pass_far or ax_out or az_out}
