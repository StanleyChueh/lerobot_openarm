"""v3 verifier = the frozen v2 verifier (sfa2/verify.py, unchanged) with the geometric corrections of the v2 review:

* can/table clearance uses the true lowest point of a (tilted) cylinder:
      z_min = min(z_lid, z_lid + h a_z) - r sqrt(1 - a_z^2)      (a = can axis lid -> bottom)
  v2 used the bottom-centre height, which ignores the rim and is wrong at the proposed tilts;
* collision / penetration statistics are evaluated at EVERY frame (v2: every third frame could miss transient
  interference in the 5-8-frame transfer);
* a signed inside-hull test between the two arms' distal links (points of one arm inside the convex hull - of 64
  vertices, an inner approximation - of a link of the other arm, with penetration depth), in addition to v2's sample nearest-neighbour clearance.
All thresholds remain source-relative where the source reconstruction itself is calibration-limited.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from .recon3 import rb
from sfa2 import verify as vf  # noqa: E402
from sfa2.recon import rc  # noqa: E402

DISTAL = ("link5", "link6", "link7", "hand", "left_finger", "right_finger")
TOL3 = dict(vf.TOL, collision_stride=1, hull_pen_excess_mm=5.0)


def cylinder_min_z(can) -> float:
    a = can[:3, 2]
    z0 = can[2, 3]
    return float(min(z0, z0 + rc.CAN_HEIGHT * a[2]) - rc.CAN_RADIUS * np.sqrt(max(0.0, 1 - a[2] ** 2)))


class HullSet:
    def __init__(self, robot):
        self.idx = {"R": [], "L": []}
        self.eq = {}
        self.sph = {}
        rng = np.random.default_rng(0)
        for i, (name, side, H, S) in enumerate(robot.links):
            if side in "RL" and name.split("_", 1)[1] in DISTAL:
                self.idx[side].append(i)
                Hs = H[rng.choice(len(H), size=min(64, len(H)), replace=False)]     # inner hull: conservative depth
                self.eq[i] = ConvexHull(Hs).equations
                c = Hs.mean(0)
                self.sph[i] = (c, float(np.linalg.norm(Hs - c, axis=1).max()))
        self.robot = robot

    def depth(self, qR, qL, gR, gL) -> float:
        """max depth (m) of any distal sample point of one arm inside a distal link hull of the other arm."""
        r = self.robot
        r.fk(qR, qL, gR, gL)
        pts = r.link_points("sample")
        best = 0.0
        for a, b in (("R", "L"), ("L", "R")):
            P = np.concatenate([pts[i][2] for i in self.idx[a]])
            for i in self.idx[b]:
                M = r.vd.oMg[i]
                loc = (P - M.translation) @ M.rotation
                c, rad = self.sph[i]
                loc = loc[np.einsum("ij,ij->i", loc - c, loc - c) < rad * rad]
                if not len(loc):
                    continue
                d = loc @ self.eq[i][:, :3].T + self.eq[i][:, 3]
                inside = (d <= 0).all(1)
                if inside.any():
                    best = max(best, float((-d[inside].max(1)).max()))
        return best


class Verifier3(vf.Verifier):
    def __init__(self, rec, robot, tol=TOL3):
        super().__init__(rec, robot, tol)
        self.hulls = HullSet(robot)
        g = self.src.g
        self.src_hull = np.array([self.hulls.depth(g["R"][t], g["L"][t], g["gR"][t], g["gL"][t]) for t in range(rec.T)])
        self.src_cyl_min = min(cylinder_min_z(c) for c in self.src_can)

    def __call__(self, cand: dict) -> dict:
        out = super().__call__(cand)
        res, tol, rec = out["checks"], self.tol, self.rec
        # corrected cylinder support height (replaces v2's bottom-centre can check; robot part unchanged)
        cmin = min(cylinder_min_z(c) for c in cand["can"])
        zc = min(rec.z_table, self.src_cyl_min) - tol["table_slack_m"]
        tc = res["table_clearance"]
        robot_ok = tc["value"]["robot_min_z"] >= tc["threshold"]["robot"]
        res["table_clearance"] = {"ok": bool(robot_ok and cmin >= zc),
                                  "value": {"robot_min_z": tc["value"]["robot_min_z"], "can_min_z_cylinder": cmin},
                                  "threshold": {"robot": tc["threshold"]["robot"], "can": zc},
                                  "note": "can: lowest rim point of the tilted cylinder (v2 used the bottom centre)"}
        g = rb.Robot.split(cand["state"])
        ex = 0.0
        for t in range(len(cand["state"])):
            d = self.hulls.depth(g["R"][t], g["L"][t], g["gR"][t], g["gL"][t])
            j = int(np.clip(np.round(cand["tau"][t]), 0, rec.T - 1))
            ex = max(ex, 1000 * (d - self.src_hull[j]))
        res["hand_hand_hull_penetration"] = {"ok": bool(ex <= tol["hull_pen_excess_mm"]), "value": ex,
                                             "threshold": tol["hull_pen_excess_mm"],
                                             "note": "excess over the source depth at tau(t), every frame"}
        failed = [k for k, v in res.items() if isinstance(v, dict) and "ok" in v and not v["ok"]]
        out["ok"], out["failed"] = not failed, failed
        return out
