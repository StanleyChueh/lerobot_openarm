"""Pinhole camera: R = camera axes in world (columns), t = camera centre in world, K intrinsics."""
from __future__ import annotations
import numpy as np
import pinocchio as pin


def quat_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Camera:
    def __init__(self, cfg: dict):
        i, e = cfg["intrinsics"], cfg["extrinsics"]
        self.W, self.H = int(i["width"]), int(i["height"])
        self.K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
        self.R = quat_R(e["quat_wxyz"]); self.t = np.asarray(e["position"], float)
        self.cfg = cfg

    def copy(self):
        c = Camera.__new__(Camera); c.W, c.H, c.K, c.R, c.t, c.cfg = self.W, self.H, self.K.copy(), self.R.copy(), self.t.copy(), self.cfg
        return c

    def project(self, P):
        c = (np.atleast_2d(P) - self.t) @ self.R; z = c[:, 2]
        return np.stack([self.K[0, 0] * c[:, 0] / z + self.K[0, 2], self.K[1, 1] * c[:, 1] / z + self.K[1, 2]], 1), z

    def backproject_plane(self, uv, z_world):
        K = self.K
        r = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], 1) @ self.R.T
        lam = (z_world - self.t[2]) / r[:, 2]
        return self.t[None] + lam[:, None] * r

    def perturbed(self, p7):
        """p7 = (rotation vector in camera frame 3, translation in world 3, log focal scale 1)."""
        c = self.copy(); p7 = np.asarray(p7, float); s = np.exp(p7[6])
        c.R = self.R @ pin.exp3(p7[:3]); c.t = self.t + p7[3:6]; c.K[0, 0] *= s; c.K[1, 1] *= s
        return c
