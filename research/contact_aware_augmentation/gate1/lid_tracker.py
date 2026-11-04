"""Causal, image-only tracker for the Pringles lid centre in body_cam (CPU, OpenCV core only).

Uses no robot state, FK, gripper command or future frame, so a robot-geometry score computed on its output is not
circular. Initialised from the human-verified frame-0 click (`research/dataset_diagnostics/annotations/real_v00.csv`),
which is the only human label it consumes.

Per frame: HSV mask of the lid colour band -> connected components -> the component nearest the constant-velocity
prediction with a plausible area -> centre of a fitted ellipse (robust to partial occlusion of the rim) and the
centroid. Each frame gets a status:

    ok          one plausible component, no flag
    uncertain   any of: touches the image border, area ratio vs the running area outside [0.6, 1.6],
                solidity < 0.85, ellipse-centre vs centroid disagreement > 6 px, jump > 40 px from the prediction
    lost        no plausible component

Only `ok` frames may ever enter a ground-truth comparison or a score. This is an automatic label: it is NEVER ground
truth (see ANNOTATION_INSTRUCTIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

HSV_LO = (12, 35, 150)
HSV_HI = (40, 170, 255)
PARAMS = {"hsv_lo": HSV_LO, "hsv_hi": HSV_HI, "min_area": 400, "max_jump_px": 40.0, "area_ratio": (0.6, 1.6),
          "min_solidity": 0.85, "max_center_disagree_px": 6.0, "border_px": 2, "search_radius_px": 90.0,
          "area_ema": 0.2}


@dataclass
class Track:
    uv: np.ndarray            # (T, 2) ellipse centre, NaN where lost
    centroid: np.ndarray      # (T, 2)
    area: np.ndarray          # (T,)
    status: list = field(default_factory=list)   # "ok" | "uncertain" | "lost"
    flags: list = field(default_factory=list)    # list of flag names per frame

    def ok(self) -> np.ndarray:
        return np.array([s == "ok" for s in self.status])


def lid_mask(rgb: np.ndarray, p=PARAMS) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, p["hsv_lo"], p["hsv_hi"])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def track(frames: np.ndarray, init_uv, p=PARAMS) -> Track:
    T, H, W = frames.shape[:3]
    uv = np.full((T, 2), np.nan)
    cen = np.full((T, 2), np.nan)
    area = np.full(T, np.nan)
    status, flags = [], []
    last = [np.asarray(init_uv, float)]  # recent confident positions (for constant-velocity prediction)
    ref_area = None
    for t in range(T):
        pred = last[-1] if len(last) < 2 else last[-1] + (last[-1] - last[-2])
        m = lid_mask(frames[t], p)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        best, best_d = None, np.inf
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < p["min_area"]:
                continue
            d = np.linalg.norm(cents[i] - pred)
            if d < best_d and d < p["search_radius_px"] + np.sqrt(a / np.pi):
                best, best_d = i, d
        if best is None:
            status.append("lost"); flags.append(["no_component"])
            continue
        comp = (lab == best).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        a = float(stats[best, cv2.CC_STAT_AREA])
        hull = cv2.convexHull(c)
        solidity = a / max(cv2.contourArea(hull), 1.0)
        e_center = cv2.fitEllipse(c)[0] if len(c) >= 5 else tuple(cents[best])
        x, y, w, h = stats[best, :4]
        f = []
        if x <= p["border_px"] or y <= p["border_px"] or x + w >= W - p["border_px"] or y + h >= H - p["border_px"]:
            f.append("border")
        if ref_area is not None and not (p["area_ratio"][0] <= a / ref_area <= p["area_ratio"][1]):
            f.append("area_jump")
        if solidity < p["min_solidity"]:
            f.append("low_solidity")
        if np.linalg.norm(np.asarray(e_center) - cents[best]) > p["max_center_disagree_px"]:
            f.append("center_disagree")
        if t > 0 and best_d > p["max_jump_px"]:
            f.append("jump")
        uv[t], cen[t], area[t] = e_center, cents[best], a
        status.append("ok" if not f else "uncertain")
        flags.append(f)
        if not f:
            last = (last + [np.asarray(e_center, float)])[-2:]
            ref_area = a if ref_area is None else (1 - p["area_ema"]) * ref_area + p["area_ema"] * a
        elif ref_area is None:
            ref_area = a
    return Track(uv, cen, area, status, flags)
