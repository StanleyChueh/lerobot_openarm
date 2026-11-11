"""Battery-rig reconstruction v2 (calibrated camera) + the charging STATION as a manipulated object.

* Camera: the battery-rig body camera fitted by render-and-compare (calib_battery.py -> calib_battery.json); the
  battery-v00 export used the camera inherited from the Pringles rig, which is 20-40 px off on this rig.
* Station (no CAD model, no manual label): segmented in frame 0 as the plate-difference foreground that is not robot
  (calibrated mesh silhouette) and not battery, connected to the left TCP at its grasp. Its footprint is back-projected
  to the grasp height and summarised by an oriented box (PCA extents, height from grasp height above the table).
  The box is a proxy geometry only: it gives the station pixels a depth and an occlusion order; the station's
  APPEARANCE is always the real pixels (forward-warped with the box's rigid motion).
  Pose: table pose until the left grasp (+3 frames), then attached to the left TCP (X_S = T_L(t_g)^-1 T_table).
* Plate: temporal median over pixels free of robot, battery AND station (all dilated).
`StationRecon` exposes the station in MGW's object ('can') slot, so the MGW renderer is used unchanged."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "battery/bsfa"), str(HERE.parent / "render_v4"), "/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation"]
import pinocchio as pin  # noqa: E402
import brecon, mesh  # noqa: E402
from sfa.robot import Camera  # noqa: E402

CALIB = HERE / "calib_battery.json"
CACHE = HERE / "cache_bat"


def calibrated_camera(path=CALIB):
    p = np.array(json.loads(Path(path).read_text())["params7"]); c0 = Camera(); c = Camera(); s = np.exp(p[6])
    c.R = c0.R @ pin.exp3(p[:3]); c.t = c0.t + p[3:6]; c.K = c0.K.copy(); c.K[0, 0] *= s; c.K[1, 1] *= s
    return c


def mgw_calib_file(path=CALIB):
    """MGW reads a 21-vector (camera 7 + joint offsets 14, zero here)."""
    out = CACHE / "mgw_calib.json"; CACHE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"params": json.loads(Path(path).read_text())["params7"] + [0.0] * 14}))
    return out


_CAM = None


def _cam():
    global _CAM
    if _CAM is None:
        _CAM = calibrated_camera()
    return _CAM


brecon.Camera = lambda: _cam()       # every BRecon built through this module uses the calibrated camera
brecon.CACHE = CACHE


class BRecon2(brecon.BRecon):
    """battery reconstruction on the calibrated camera (separate cache)."""


def box_points(ext, spacing=0.003, rng=np.random.default_rng(3)):
    """surface samples of a box [-ex/2,ex/2] x [-ey/2,ey/2] x [0,h] (local frame, origin = bottom centre)."""
    ex, ey, h = ext; pts = []
    for (a, b, fix_axis, fix_val, ua, ub) in ((ex, ey, 2, h, 0, 1), (ex, ey, 2, 0.0, 0, 1), (ex, h, 1, ey / 2, 0, 2),
                                              (ex, h, 1, -ey / 2, 0, 2), (ey, h, 0, ex / 2, 1, 2), (ey, h, 0, -ex / 2, 1, 2)):
        n = max(50, int(a * b / spacing ** 2)); P = np.zeros((n, 3))
        P[:, ua] = (rng.random(n) - (0.5 if ua < 2 else 0.0)) * a; P[:, ub] = (rng.random(n) - (0.5 if ub < 2 else 0.0)) * b
        P[:, fix_axis] = fix_val; pts.append(P)
    return np.concatenate(pts)


class StationRecon(BRecon2):
    def __init__(self, e, device="cuda:0"):
        super().__init__(e, device=device)
        self.tLg = self.ev["L"]["closes"][0]
        self.tLo = self.ev["L"]["opens"][0] if self.ev["L"]["opens"] else self.T - 1
        cp = CACHE / f"ep{e}_station.npz"
        if cp.exists():
            z = np.load(cp); self.st_table, self.st_ext = z["table"], z["ext"]; self.st_info = json.loads(str(z["info"]))
            self.plate = z["plate"]
        else:
            self.st_table, self.st_ext, self.st_info = self._station_on_table()
        self.XS = np.linalg.inv(self.tcp("L", self.tLg + 3)) @ self.st_table
        if not cp.exists():
            self.plate = self._plate_station()
        self._station_masks()
        self._st_ready = True
        if not cp.exists():
            np.savez_compressed(cp, table=self.st_table, ext=self.st_ext, plate=self.plate, info=json.dumps(self.st_info))

    # ------------------------------------------------------------------ station geometry
    def _fg(self, j, plate, thr=30):
        fg = np.abs(self.F[j].astype(np.int16) - plate.astype(np.int16)).max(2) > thr
        return cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    def _station_on_table(self):
        TL = self.tcp("L", self.tLg)
        # table height: the lowest arm point over the 10 real demos (fingertips reach the table) - 5 mm; TCP heights are
        # biased (the TCP frame sits above the fingertips), so they are not used for the table
        z_tab = float(json.loads((HERE.parent / "battery/cache/envelope.json").read_text())["min_z_m"]) - 0.005
        h = float(np.clip(2 * (TL[2, 3] - z_tab), 0.02, 0.07))
        zs = z_tab + 0.75 * h                                                 # back-projection plane: near the top face
        uv, _ = self.cam.project(TL[:3, 3][None]); u0, v0 = uv[0]
        r, b = self.silhouettes(0)
        fg = self._fg(0, self.plate) & ~(cv2.dilate(r.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~b
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8))
        near = np.zeros(fg.shape, np.uint8); cv2.circle(near, (int(u0), int(v0)), 45, 1, -1)
        far = np.zeros(fg.shape, np.uint8); cv2.circle(far, (int(u0), int(v0)), 130, 1, -1)
        keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 60 and (near[lab == k] > 0).any()]
        m = np.isin(lab, keep) & (far > 0)
        info = {"tcp_grasp_xyz": TL[:3, 3].tolist(), "tcp_grasp_px": [float(u0), float(v0)], "seg_px": int(m.sum()), "z_table": z_tab}
        if m.sum() < 300:
            raise RuntimeError(f"station not segmented in frame 0 (ep {self.e}, {int(m.sum())} px)")
        ys, xs = np.nonzero(m)
        P = self._backproject(np.stack([xs, ys], 1).astype(float), zs)
        c = P[:, :2].mean(0); U = np.linalg.svd(P[:, :2] - c)[2]
        proj = (P[:, :2] - c) @ U.T
        lo, hi = np.percentile(proj, 3, 0), np.percentile(proj, 97, 0)
        ext_xy = hi - lo; c = c + ((lo + hi) / 2) @ U
        T = np.eye(4); T[:3, 0] = [U[0, 0], U[0, 1], 0]; T[:3, 1] = [U[1, 0], U[1, 1], 0]; T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
        if T[2, 2] < 0:
            T[:3, 1] *= -1; T[:3, 2] = np.cross(T[:3, 0], T[:3, 1])
        T[:3, 3] = [c[0], c[1], z_tab]
        info.update({"box_extent_m": [float(ext_xy[0]), float(ext_xy[1]), h], "centre_xy_m": c.tolist(),
                     "centre_vs_tcp_xy_mm": float(1000 * np.linalg.norm(c - TL[:2, 3]))})
        return T, np.array([ext_xy[0], ext_xy[1], h]), info

    def station_pose(self, t_state):
        t = int(round(t_state))
        if t < self.tLg + 3:
            return self.st_table
        return self.tcp("L", min(t_state, self.tLo)) @ self.XS

    def _station_sil(self, Tst, dil=1):
        P = (Tst[:3, :3] @ box_points(self.st_ext, 0.004).T).T + Tst[:3, 3]
        uv, zc = self.cam.project(P); ok = (zc > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        m = np.zeros((480, 640), np.uint8); m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return cv2.dilate(m, np.ones((dil, dil), np.uint8)) > 0 if dil > 1 else m > 0

    def _plate_station(self, dil=15):
        occ = np.zeros((self.T, 480, 640), bool)
        for j in range(self.T):
            r, b = self.silhouettes(j)
            s = self._station_sil(self.station_pose(j + self.lag), dil=31)
            occ[j] = (cv2.dilate((r | b).astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0) | s
        plate = np.zeros((480, 640, 3), np.uint8)
        F = torch.tensor(self.F, device=self.dev); O = torch.tensor(occ, device=self.dev)
        for c in range(3):
            x = F[..., c].half().masked_fill(O, float("nan"))
            med = torch.nanmedian(x.view(self.T, -1), dim=0).values.view(480, 640)
            plate[..., c] = torch.nan_to_num(med, nan=0).clamp(0, 255).byte().cpu().numpy()
        never = occ.all(0); self.st_info["plate_never_free_frac"] = float(never.mean())
        del F, O; torch.cuda.empty_cache()
        return cv2.inpaint(plate, never.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)

    def _station_masks(self):
        """MGW object slot = station: real station pixels = foreground inside the dilated box silhouette, minus robot."""
        self.masks["bat"] = self.masks["can"].copy()
        st = np.zeros((self.T, 480, 640), np.uint8)
        for j in range(self.T):
            if j > self.tLo:
                continue
            box = self._station_sil(self.station_pose(j + self.lag), dil=21)
            robot = (self.masks["R"][j] > 0) | (self.masks["L"][j] > 0)
            st[j] = box & self._fg(j, self.plate, thr=25) & ~robot & ~(self.masks["bat"][j] > 0)
        self.masks["can"] = st

    # MGW object-slot interface
    def source_can(self, robot, t_state):
        if not getattr(self, "_st_ready", False):          # during the battery reconstruction: the battery
            return super().source_can(robot, t_state)
        return self.station_pose(t_state)
