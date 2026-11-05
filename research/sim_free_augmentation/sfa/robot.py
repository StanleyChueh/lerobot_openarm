"""Robot kinematics, camera projection and link geometry (CPU, pinocchio; no physics).

* URDF: model/openarm_description.urdf (follower), visual STL meshes -> per-link convex hulls.
* Camera: body_cam extrinsic FITTED in IsaacLab (silhouette chamfer, inherited, unvalidated on held-out keypoints),
  intrinsics fx = fy = 607.22, c = (326.4, 238.7). Optionally refined per episode by `refine_camera` (silhouette IoU
  against SAM2 robot masks); the refinement and its held-out residual are reported, never assumed.
* Dataset conventions (LeRobot v3, ethanCSL/openarm_pringles_lerobot_real_10_v00): 16-D interleaved RJ1, LJ1, ...,
  RJ8, LJ8 in motor radians; grippers raw: right open -1.298, left open 1.2121, closed 0 (calibration.json).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[3]
URDF = REPO / "model/openarm_description.urdf"
MESH_DIR = REPO / "model/meshes"
NAMES = [f"{s}J{i}.pos" for i in range(1, 9) for s in ("R", "L")]
GRIP_OPEN_RAW = {"R": -1.298, "L": 1.2121}
FINGER_OPEN = 0.044
SIDE = {"R": "right", "L": "left"}

BODY_CAM = {"pos": [0.05621, -0.00305, 0.83062], "quat_wxyz": [0.18720, -0.68244, 0.69543, -0.12490],
            "fx": 607.22, "fy": 607.22, "cx": 326.4, "cy": 238.7, "W": 640, "H": 480,
            "source": "IsaacLab@4aa5e026 body_cam OffsetCfg (silhouette fit on real_v00, inherited)"}


def quat_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Camera:
    def __init__(self, cam: dict = BODY_CAM, R: np.ndarray | None = None, t: np.ndarray | None = None):
        self.cfg = dict(cam)
        self.R = quat_R(cam["quat_wxyz"]) if R is None else np.asarray(R, float)   # camera axes in world
        self.t = np.asarray(cam["pos"], float) if t is None else np.asarray(t, float)
        self.K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])

    def to_cam(self, P):
        return (np.atleast_2d(P) - self.t) @ self.R

    def project(self, P):
        c = self.to_cam(P)
        z = c[:, 2]
        uv = np.stack([self.K[0, 0] * c[:, 0] / z + self.K[0, 2], self.K[1, 1] * c[:, 1] / z + self.K[1, 2]], 1)
        return uv, z

    def perturbed(self, rvec, dt):
        """camera rotated by rvec (axis-angle, camera frame) and moved by dt (world)."""
        return Camera(self.cfg, self.R @ pin.exp3(np.asarray(rvec, float)), self.t + np.asarray(dt, float))


def _stl(path: str) -> np.ndarray:
    b = open(path, "rb").read()
    if b[:5] == b"solid" and b"facet" in b[:400]:
        return np.array([list(map(float, l.split()[1:4])) for l in b.decode(errors="ignore").splitlines()
                         if l.strip().startswith("vertex")])
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


class Robot:
    def __init__(self):
        self.m = pin.buildModelFromUrdf(str(URDF))
        self.d = self.m.createData()
        self.vis = pin.buildGeomFromUrdf(self.m, str(URDF), pin.GeometryType.VISUAL, package_dirs=[str(MESH_DIR)])
        self.vd = pin.GeometryData(self.vis)
        self.links = []          # (name, side, hull vertices in the geometry frame, sample points)
        rng = np.random.default_rng(0)
        for g in self.vis.geometryObjects:
            V = _stl(g.meshPath) * g.meshScale
            H = V[ConvexHull(V).vertices]
            S = V[rng.choice(len(V), size=min(400, len(V)), replace=False)]
            side = "R" if g.name.startswith("openarm_right_") else ("L" if g.name.startswith("openarm_left_") else "B")
            self.links.append((g.name.replace("openarm_", "").rsplit("_", 1)[0], side, H, S))
        self.qidx = {s: [self.m.idx_qs[self.m.getJointId(f"openarm_{SIDE[s]}_joint{i}")] for i in range(1, 8)] for s in "RL"}
        self.vidx = {s: [self.m.idx_vs[self.m.getJointId(f"openarm_{SIDE[s]}_joint{i}")] for i in range(1, 8)] for s in "RL"}
        self.fidx = {s: [self.m.idx_qs[self.m.getJointId(f"openarm_{SIDE[s]}_finger_joint{i}")] for i in (1, 2)] for s in "RL"}
        self.hand = {s: self.m.getFrameId(f"openarm_{SIDE[s]}_hand") for s in "RL"}
        self.tcp = {s: self.m.getFrameId(f"openarm_{SIDE[s]}_hand_tcp") for s in "RL"}
        self.lo = {s: self.m.lowerPositionLimit[self.qidx[s]].copy() for s in "RL"}
        self.hi = {s: self.m.upperPositionLimit[self.qidx[s]].copy() for s in "RL"}
        self.vel = {s: self.m.velocityLimit[self.vidx[s]].copy() for s in "RL"}

    # ------------------------------------------------------------------ dataset <-> configuration
    @staticmethod
    def split(x16: np.ndarray) -> dict:
        """(T,16) dataset vector -> {"R": (T,7), "L": (T,7), "gR": (T,), "gL": (T,)} (grippers as openness 0..1)."""
        x16 = np.atleast_2d(x16)
        out = {s: x16[:, [NAMES.index(f"{s}J{i}.pos") for i in range(1, 8)]] for s in "RL"}
        for s in "RL":
            out["g" + s] = np.clip(x16[:, NAMES.index(f"{s}J8.pos")] / GRIP_OPEN_RAW[s], 0, 1)
        return out

    @staticmethod
    def merge(R7, L7, gR_raw, gL_raw) -> np.ndarray:
        T = len(R7)
        x = np.zeros((T, 16))
        for i in range(7):
            x[:, NAMES.index(f"RJ{i + 1}.pos")] = R7[:, i]
            x[:, NAMES.index(f"LJ{i + 1}.pos")] = L7[:, i]
        x[:, NAMES.index("RJ8.pos")] = gR_raw
        x[:, NAMES.index("LJ8.pos")] = gL_raw
        return x

    def q_full(self, qR, qL, gR=1.0, gL=1.0):
        q = pin.neutral(self.m)
        q[self.qidx["R"]] = qR
        q[self.qidx["L"]] = qL
        for s, g in (("R", gR), ("L", gL)):
            q[self.fidx[s]] = FINGER_OPEN * float(np.clip(g, 0, 1))
        return q

    def fk(self, qR, qL, gR=1.0, gL=1.0):
        pin.framesForwardKinematics(self.m, self.d, self.q_full(qR, qL, gR, gL))
        pin.updateGeometryPlacements(self.m, self.d, self.vis, self.vd)

    def frame(self, fid) -> np.ndarray:
        return self.d.oMf[fid].homogeneous.copy()

    def link_points(self, which="hull"):
        """after fk(): list of (name, side, world points) for every visual link."""
        out = []
        for (name, side, H, S), oM in zip(self.links, self.vd.oMg):
            P = H if which == "hull" else S
            out.append((name, side, (oM.rotation @ P.T).T + oM.translation))
        return out

    def tcp_traj(self, Q: dict, side: str) -> np.ndarray:
        T = []
        for t in range(len(Q["R"])):
            pin.framesForwardKinematics(self.m, self.d, self.q_full(Q["R"][t], Q["L"][t]))
            T.append(self.frame(self.tcp[side]))
        return np.stack(T)

    # ------------------------------------------------------------------ inverse kinematics
    def ik(self, side, T_target, q0, iters=80, damp=1e-3, margin=0.02, frame="tcp"):
        """DLS IK to the TCP pose, clamped inside URDF limits - margin. Returns (q, pos_err_m, rot_err_deg)."""
        lo, hi = self.lo[side] + margin, self.hi[side] - margin
        q = np.clip(np.asarray(q0, float), lo, hi)
        fid = self.tcp[side] if frame == "tcp" else self.hand[side]
        target = pin.SE3(T_target[:3, :3], T_target[:3, 3])
        other = "L" if side == "R" else "R"
        for _ in range(iters):
            x = self.q_full(**{f"q{side}": q, f"q{other}": np.zeros(7)})
            pin.framesForwardKinematics(self.m, self.d, x)
            err = pin.log6(self.d.oMf[fid].inverse() * target).vector
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
                break
            J = pin.computeFrameJacobian(self.m, self.d, x, fid, pin.LOCAL)[:, self.vidx[side]]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err), lo, hi)
        x = self.q_full(**{f"q{side}": q, f"q{other}": np.zeros(7)})
        pin.framesForwardKinematics(self.m, self.d, x)
        cur = self.d.oMf[fid]
        e = pin.log6(cur.inverse() * target).vector
        return q, float(np.linalg.norm(cur.translation - target.translation)), float(np.degrees(np.linalg.norm(e[3:])))


def link_silhouettes(robot: Robot, cam: Camera, qR, qL, gR, gL, H=480, W=640, skip_body=True):
    """Convex-hull silhouettes of every visual link: list of (name, side, mask uint8, mean depth, hull pts 2D)."""
    import cv2

    robot.fk(qR, qL, gR, gL)
    out = []
    for name, side, P in robot.link_points("hull"):
        if skip_body and side == "B":
            continue
        uv, z = cam.project(P)
        if (z < 0.05).any():
            continue
        hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, hull, 1)
        out.append((name, side, m, float(np.median(z)), hull))
    return out


def seg_seg_dist(p1, q1, p2, q2) -> float:
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 or e < 1e-12:
        return float(np.linalg.norm(r))
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


ARM_CHAIN = ["link2", "link3", "link4", "link5", "link6", "link7"]


def arm_clearance(robot: Robot, qR, qL, link_radius=0.045) -> float:
    """min surface distance between the two arms' link2..link7 capsule chains (approximate geometry)."""
    pin.framesForwardKinematics(robot.m, robot.d, robot.q_full(qR, qL))
    pts = {s: [robot.d.oMf[robot.m.getFrameId(f"openarm_{SIDE[s]}_{n}")].translation.copy() for n in ARM_CHAIN]
           for s in "RL"}
    d = min(seg_seg_dist(pts["R"][i], pts["R"][i + 1], pts["L"][j], pts["L"][j + 1])
            for i in range(len(ARM_CHAIN) - 1) for j in range(len(ARM_CHAIN) - 1))
    return d - 2 * link_radius
