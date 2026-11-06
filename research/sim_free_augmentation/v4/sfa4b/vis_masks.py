"""B4 occlusion masks for held-out body-camera frames (diagnostics only; never used to build training pairs).

object_only  : convex hulls of large HSV-red components within 150 px of the Gate-1 lid track, plus the tracked lid
               disk (radius 1.15 x equivalent lid radius), dilated 5 px, MINUS the gripper mask.
gripper_only : convex-hull silhouettes of both hands and all four fingers from FK of the recorded state through the
               nominal body camera, dilated 15 px (the inherited body-camera calibration has ~20-40 px offsets).
background   : the object_only mask translated, without reshaping, to the first offset (fixed order, nearest first,
               >= 100 px) where it lies fully inside the image and does not touch the object, any robot link or the
               gripper (all dilated 15 px); same pixel count as object_only. Falls back to an equal-area disk at the
               free location farthest from the robot if no translate fits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SFA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SFA / "v3"))
from sfa3.recon3 import rb  # noqa: E402

H, W = 480, 640
GRIP_LINKS = ("hand", "finger")
_K15 = np.ones((31, 31), np.uint8)
OFFSETS = sorted(((dx, dy) for dx in range(-560, 561, 20) for dy in range(-400, 401, 20) if np.hypot(dx, dy) >= 100),
                 key=lambda d: (np.hypot(*d), d))


def red_hulls(rgb, min_area=300):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(red)
    return [(lab == k, cen[k]) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_area]


def robot_masks(robot, cam, state16):
    g = rb.Robot.split(state16[None])
    sil = rb.link_silhouettes(robot, cam, g["R"][0], g["L"][0], g["gR"][0], g["gL"][0], H, W)
    grip = np.zeros((H, W), np.uint8)
    arm = np.zeros((H, W), np.uint8)
    for name, side, m, z, hull in sil:
        arm |= m
        if any(k in name for k in GRIP_LINKS):
            grip |= m
    return cv2.dilate(grip, _K15) > 0, cv2.dilate(arm, _K15) > 0


def object_mask(rgb, lid, area, grip):
    m = np.zeros((H, W), np.uint8)
    if lid is None:
        return m > 0
    for comp, c in red_hulls(rgb):
        if np.hypot(c[0] - lid[0], c[1] - lid[1]) <= 150:
            ys, xs = np.nonzero(comp)
            cv2.fillConvexPoly(m, cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)), 1)
    r = int(round(1.15 * np.sqrt(max(area, 1.0) / np.pi)))
    cv2.circle(m, (int(lid[0]), int(lid[1])), max(r, 6), 1, -1)
    m = cv2.dilate(m, np.ones((11, 11), np.uint8)) > 0
    return m & ~grip


def background_mask(obj, grip, arm):
    n = int(obj.sum())
    if n == 0:
        return obj.copy(), "empty"
    forbid = cv2.dilate((obj | grip | arm).astype(np.uint8), _K15) > 0
    ys, xs = np.nonzero(obj)
    for dx, dy in OFFSETS:
        y2, x2 = ys + dy, xs + dx
        if y2.min() < 0 or x2.min() < 0 or y2.max() >= H or x2.max() >= W:
            continue
        if forbid[y2, x2].any():
            continue
        out = np.zeros((H, W), bool)
        out[y2, x2] = True
        return out, f"shift({dx},{dy})"
    free = (~forbid).astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    out = np.zeros((H, W), np.uint8)
    cv2.circle(out, (int(cx), int(cy)), int(np.sqrt(n / np.pi)), 1, -1)
    return out > 0, "disk_fallback"


def all_masks(robot, cam, rgb, state16, lid, area):
    grip, arm = robot_masks(robot, cam, state16)
    obj = object_mask(rgb, lid, area, grip)
    bg, how = background_mask(obj, grip, arm)
    return {"object_only": obj, "gripper_only": grip, "background": bg}, how
