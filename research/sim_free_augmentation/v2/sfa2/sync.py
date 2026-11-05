"""RGB / state / action synchronisation: the exported time map, alignment detectors and corruption controls.

Row convention (identical to the source dataset, measured, not assumed):
  * row i of a generated episode holds observation.state[i] = candidate (planned, NOT measured) state at state frame i,
    action[i] = candidate command at frame i, and image[i] = body_cam frame rendered to depict the candidate state at
    state time i + lag  (lag = the source episode's measured video-to-state lag, e.g. -1.75 frames on episode 0);
  * the command leads the state by `lead` frames (source: 2) because a_new(t) = q_new(t+lead) + r_src(tau(t));
  * rows whose image depicts a state time < 0 or whose command needs q_new beyond the last frame are TRIMMED, not
    padded: kept rows are ceil(-lag) .. T-1-lead.
A training consumer therefore sees exactly the image/state/action relation of the real recording. The relation is
re-measured on every generated clip by detectors that use only the exported arrays and the image.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from . import se3
from .recon import Recon, rb


def export_range(rec: Recon) -> tuple[int, int]:
    start = max(0, math.ceil(-rec.lag))
    stop = rec.T - rec.lead - max(0, math.ceil(rec.lag))
    return start, stop


def time_map(rec: Recon, cand: dict, j_of: np.ndarray) -> dict:
    a, b = export_range(rec)
    rows = np.arange(a, b)
    return {"row": rows - a, "candidate_frame": rows, "timestamp_s": (rows - a) / rec.fps,
            "image_depicts_state_time": rows + rec.lag, "source_state_time": np.interp(rows, np.arange(rec.T), cand["tau"]),
            "source_video_frame": j_of[rows], "source_episode": np.full(len(rows), rec.e)}


# ----------------------------------------------------------------------------- detectors
def action_lead(state16, action16, kmax=10) -> dict:
    """argmin_k mean |a(t) - s(t+k)| over arm joints, k in [-kmax, kmax] (negative = command lags the state)."""
    g_s, g_a = rb.Robot.split(state16), rb.Robot.split(action16)
    ok = np.isfinite(action16).all(1)
    err = {}
    T = len(state16)
    for k in range(-kmax, kmax + 1):
        t = np.arange(max(0, -k), min(T, T - k))
        t = t[ok[t]]
        err[k] = float(np.mean([np.abs(g_a[s][t] - g_s[s][t + k]).mean() for s in "RL"]))
    best = min(err, key=err.get)
    return {"lead": best, "err": err}


# ----------------------------------------------------------------------------- corruptions
def delay_actions(c: dict, k: int) -> dict:
    d = copy.deepcopy(c)
    a = np.full_like(c["action"], np.nan)
    a[k:] = c["action"][:-k]
    d["action"], d["valid"] = a, np.isfinite(a).all(1)
    return d


def delay_state(c: dict, k: int) -> dict:
    d = copy.deepcopy(c)
    s = c["state"].copy()
    s[k:] = c["state"][:-k]
    s[:k] = c["state"][0]
    d["state"] = s
    return d


def rotate_object(c: dict, deg: float, after: int, axis=(1.0, 0, 0)) -> dict:
    d = copy.deepcopy(c)
    for t in range(after, len(d["can"])):
        d["can"][t] = d["can"][t] @ se3.rot(axis, deg)
    return d


def shift_gripper(c: dict, side: str, k: int) -> dict:
    """move one gripper's command AND state timeline by k frames (positive = later)."""
    d = copy.deepcopy(c)
    i = rb.NAMES.index(f"{side}J8.pos")
    for key in ("state", "action"):
        x = c[key][:, i].copy()
        y = np.roll(x, k)
        if k > 0:
            y[:k] = x[0]
        else:
            y[k:] = x[-1]
        d[key][:, i] = y
    d["valid"] = np.isfinite(d["action"]).all(1)
    return d


def source_labels(c: dict, rec: Recon) -> dict:
    d = copy.deepcopy(c)
    d["state"], d["action"] = rec.state.copy(), rec.action.copy()
    return d
