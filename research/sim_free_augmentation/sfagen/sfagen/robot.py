"""Robot-agnostic kinematics from a URDF + a mapping from the LeRobot state/action vector to URDF joints.

Config (YAML `robot:` block):
  urdf: path/to/robot.urdf
  package_dirs: [path/to/meshes]               # where mesh files referenced by the URDF are found
  arms:                                         # one or more arms, any names (e.g. R, L, arm)
    R:
      joints: [openarm_right_joint1, ...]      # URDF joint names, in the order of `state_indices`
      state_indices: [0, 2, 4, ...]            # their positions in observation.state / action
      tcp_frame: openarm_right_hand_tcp         # URDF frame of the tool centre point
      gripper:                                  # optional
        state_index: 14                         # gripper entry of state/action
        open_raw: -1.298                        # raw value when fully open
        closed_raw: 0.0                         # raw value when closed
        finger_joints: [openarm_right_finger_joint1, openarm_right_finger_joint2]
        finger_open: 0.044                      # finger joint value when open (closed = 0)
        close_threshold: 0.5                    # openness below which the gripper counts as closed
Every visual geometry is assigned to the arm whose joint is its nearest kinematic ancestor (else to the base 'B')."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial import ConvexHull


def read_mesh(path: str) -> np.ndarray:
    """triangle soup (N*3, 3) from STL (binary/ascii); other formats through trimesh if installed."""
    p = str(path)
    if p.lower().endswith(".stl"):
        b = open(p, "rb").read()
        if b[:5] == b"solid" and b"facet" in b[:400]:
            return np.array([list(map(float, l.split()[1:4])) for l in b.decode(errors="ignore").splitlines()
                             if l.strip().startswith("vertex")])
        n = struct.unpack("<I", b[80:84])[0]
        a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        return a["v"].reshape(-1, 3).astype(float)
    import trimesh  # obj / dae / glb
    m = trimesh.load(p, force="mesh")
    return np.asarray(m.vertices)[np.asarray(m.faces)].reshape(-1, 3).astype(float)


class Robot:
    def __init__(self, cfg: dict, base_dir: Path | None = None):
        base = Path(base_dir or ".")
        rp = lambda p: str((base / p).resolve()) if not Path(p).is_absolute() else str(p)  # noqa: E731
        self.cfg = cfg
        urdf = rp(cfg["urdf"]); pdirs = [rp(p) for p in cfg.get("package_dirs", [])]
        self.m = pin.buildModelFromUrdf(urdf); self.d = self.m.createData()
        self.vis = pin.buildGeomFromUrdf(self.m, urdf, pin.GeometryType.VISUAL, package_dirs=pdirs)
        self.vd = pin.GeometryData(self.vis)
        try:
            self.col = pin.buildGeomFromUrdf(self.m, urdf, pin.GeometryType.COLLISION, package_dirs=pdirs)
        except Exception:
            self.col = self.vis
        self.arms = list(cfg["arms"])
        A = cfg["arms"]
        jid = lambda n: self.m.getJointId(n)  # noqa: E731
        for s in self.arms:
            for n in A[s]["joints"]:
                assert jid(n) < self.m.njoints, f"joint {n} not in URDF"
        self.qidx = {s: [self.m.idx_qs[jid(n)] for n in A[s]["joints"]] for s in self.arms}
        self.vidx = {s: [self.m.idx_vs[jid(n)] for n in A[s]["joints"]] for s in self.arms}
        self.sidx = {s: list(A[s]["state_indices"]) for s in self.arms}
        self.tcp = {s: self.m.getFrameId(A[s]["tcp_frame"]) for s in self.arms}
        self.grip = {s: A[s].get("gripper") for s in self.arms}
        self.fidx = {s: [self.m.idx_qs[jid(n)] for n in (self.grip[s] or {}).get("finger_joints", [])] for s in self.arms}
        self.lo = {s: self.m.lowerPositionLimit[self.qidx[s]].copy() for s in self.arms}
        self.hi = {s: self.m.upperPositionLimit[self.qidx[s]].copy() for s in self.arms}
        self.vel = {s: self.m.velocityLimit[self.vidx[s]].copy() for s in self.arms}
        # arm ownership of every joint: nearest ancestor joint that is listed for an arm
        own = {}
        for s in self.arms:
            for n in A[s]["joints"] + list((self.grip[s] or {}).get("finger_joints", [])):
                own[jid(n)] = s
        def owner(j):
            while j > 0:
                if j in own:
                    return own[j]
                j = self.m.parents[j]
            return "B"
        self.geom_arm = [owner(g.parentJoint) for g in self.vis.geometryObjects]
        self.col_arm = [owner(g.parentJoint) for g in self.col.geometryObjects]
        self.joint_owner = owner
        rng = np.random.default_rng(0); self.links = []
        for g, s in zip(self.vis.geometryObjects, self.geom_arm):
            V = read_mesh(g.meshPath) * g.meshScale
            H = V[ConvexHull(V).vertices] if len(V) >= 4 else V
            self.links.append((g.name, s, H))

    # ------------------------------------------------------------------ dataset vector <-> configuration
    def split(self, x):
        """(T, D) state/action -> {arm: (T, n) joints, 'g'+arm: (T,) gripper openness 0..1}."""
        x = np.atleast_2d(np.asarray(x, float)); out = {}
        for s in self.arms:
            out[s] = x[:, self.sidx[s]]
            g = self.grip[s]
            if g:
                out["g" + s] = np.clip((x[:, g["state_index"]] - g.get("closed_raw", 0.0)) / (g["open_raw"] - g.get("closed_raw", 0.0)), 0, 1)
        return out

    def q_full(self, q: dict, g: dict | None = None):
        x = pin.neutral(self.m)
        for s in self.arms:
            if s in q:
                x[self.qidx[s]] = q[s]
            if self.fidx[s]:
                x[self.fidx[s]] = self.grip[s].get("finger_open", 0.04) * float(np.clip((g or {}).get(s, 1.0), 0, 1))
        return x

    def fk(self, q: dict, g: dict | None = None):
        pin.framesForwardKinematics(self.m, self.d, self.q_full(q, g))
        pin.updateGeometryPlacements(self.m, self.d, self.vis, self.vd)

    def frame(self, fid):
        return self.d.oMf[fid].homogeneous.copy()

    def tcp_traj(self, Q: dict, side: str):
        T = []
        for t in range(len(Q[side])):
            self.fk({s: Q[s][t] for s in self.arms})
            T.append(self.frame(self.tcp[side]))
        return np.stack(T)

    def tcp_from_state(self, S, side):
        sp = self.split(S); return self.tcp_traj({s: sp[s] for s in self.arms}, side)

    def link_points(self):
        """(name, arm, world hull vertices) at the last fk() call."""
        out = []
        for (n, s, H), M in zip(self.links, self.vd.oMg):
            out.append((n, s, H @ M.rotation.T + M.translation))
        return out

    def finger_axis(self, side):
        """world direction in which the first finger joint of `side` moves (parallel-gripper closing axis), at the last fk()."""
        if not self.fidx[side]:
            return None
        jn = self.grip[side]["finger_joints"][0]; jid = self.m.getJointId(jn); jm = self.m.joints[jid]
        sn = jm.shortname(); loc = {"JointModelPX": [1, 0, 0], "JointModelPY": [0, 1, 0], "JointModelPZ": [0, 0, 1]}.get(sn)
        if loc is None:
            try:
                loc = np.asarray(jm.extract().axis).ravel()
            except Exception:
                loc = [0, 1, 0]
        return self.d.oMi[jid].rotation @ np.asarray(loc, float)

    def gripper_events(self, A):
        """command events per arm with a gripper: {'closes': [...], 'opens': [...]} (frame indices)."""
        out = {}
        sp = self.split(A)
        for s in self.arms:
            if not self.grip[s]:
                continue
            closed = sp["g" + s] < self.grip[s].get("close_threshold", 0.5)
            ch = np.flatnonzero(np.diff(closed.astype(int)))
            out[s] = {"closes": [int(c + 1) for c in ch if closed[c + 1]], "opens": [int(c + 1) for c in ch if not closed[c + 1]]}
        return out
