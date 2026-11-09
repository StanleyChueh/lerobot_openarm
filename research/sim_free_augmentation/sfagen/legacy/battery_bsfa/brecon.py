"""Per-episode reconstruction for the battery-insertion task (no SAM2 needed):
  * video/state lag: scanned by agreement of the URDF mesh render with the image foreground;
  * clean background plate: temporal median over frames where the pixel is free of the (mesh-rendered, dilated) robot
    and of the battery; never-free pixels are Telea-inpainted;
  * battery: 18650 cell (r = 9.25 mm, L = 65 mm). Resting pose on the table = the right TCP position at the grasp
    (kinematics) refined by the bright blob in frame 0 (centroid + principal axis, back-projected to the plane at the
    grasp height). Attached to the right TCP from the grasp to the first right release, then to the left TCP (station).
Interface mirrors sfa2.recon.Recon so the MGW renderer can be used unchanged."""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), "/home/stanley/Stanley_ws/sfa_hf_export/render_v4"]
import bdata, mesh  # noqa: E402
from sfa.robot import Robot, Camera  # noqa: E402

BAT_R, BAT_L = 0.00925, 0.065
CACHE = HERE.parent / "cache"


def interp_rows(X, t):
    T = len(X); t = np.clip(t, 0, T - 1); i = np.floor(t).astype(int); j = np.minimum(i + 1, T - 1); f = (t - i)[..., None]
    return X[i] * (1 - f) + X[j] * f


class BRecon:
    def __init__(self, e, device="cuda:0", lag=None, verbose=True):
        self.e, self.dev = e, device
        ep = bdata.load(e); self.ep = ep
        self.F = ep.frames[bdata.CAMS[0]]; self.T = len(self.F)
        self.state, self.action = ep.state, ep.action
        self.robot, self.cam = Robot(), Camera()
        self.G = mesh.Geometry(self.robot, device=device, spacing=0.003)
        self.ev = bdata.grip_events(ep.action)
        self.tg = self.ev["R"]["closes"][0]; self.tr = self.ev["R"]["opens"][0]
        self.g = Robot.split(self.state)
        CACHE.mkdir(parents=True, exist_ok=True)
        cp = CACHE / f"ep{e}_recon.npz"
        if cp.exists() and lag is None:
            z = np.load(cp); self.lag = float(z["lag"]); self.plate = z["plate"]; self.table = z["table"]; self.info = json.loads(str(z["info"]))
            self._video_states(); self._attach(); self._masks()
            return
        self.info = {}
        self.lag = -1.75 if lag is None else lag
        self._video_states()
        self.table = self._battery_on_table()
        self._attach()
        self.plate = self._plate()
        if lag is None:
            base = self.lag; scores = self._lag_scan_scores(); best = max(scores, key=scores.get)
            # adopt a different lag only if it clearly improves silhouette agreement (the scan is flat on this rig)
            self.lag = best if scores[best] > scores.get(base, -1) + 0.01 else base
            self.info["lag"] = self.lag; self._video_states()
            if self.lag != base:
                self.plate = self._plate()
        self._masks()
        np.savez_compressed(cp, lag=self.lag, plate=self.plate, table=self.table, info=json.dumps(self.info))

    # ------------------------------------------------------------ states at video time j (state time j + lag)
    def _video_states(self):
        ts = np.arange(self.T) + self.lag
        self.gv = {s: interp_rows(self.g[s], ts) for s in "RL"}
        self.grip_v = {s: interp_rows(self.g["g" + s][:, None], ts)[:, 0] for s in "RL"}

    def tcp(self, side, t_state):
        q = {s: interp_rows(self.g[s], np.array([t_state]))[0] for s in "RL"}
        self.robot.fk(q["R"], q["L"])
        return self.robot.frame(self.robot.tcp[side])

    # ------------------------------------------------------------ battery
    def _battery_on_table(self):
        Tg = self.tcp("R", self.tg)
        c0 = Tg[:3, 3].copy()
        uv0, _ = self.cam.project(c0[None]); u0, v0 = uv0[0]
        img = self.F[0]; hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        bright = (hsv[..., 2] > 150) & (hsv[..., 1] < 70)
        roi = np.zeros(bright.shape, np.uint8); cv2.circle(roi, (int(u0), int(v0)), 70, 1, -1)
        n, lab, st, cen = cv2.connectedComponentsWithStats((bright & (roi > 0)).astype(np.uint8))
        info = {"tcp_grasp_xyz": c0.tolist(), "tcp_grasp_px": [float(u0), float(v0)]}
        z = c0[2]
        if n > 1:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); ys, xs = np.nonzero(lab == k)
            P = self._backproject(np.stack([xs, ys], 1).astype(float), z)
            c = P.mean(0); d = np.linalg.svd(P[:, :2] - c[:2])[2][0]
            axis = np.array([d[0], d[1], 0.0]); axis /= np.linalg.norm(axis)
            info.update({"blob_px": int(len(xs)), "blob_centroid_px": [float(xs.mean()), float(ys.mean())],
                         "blob_extent_m": float(np.ptp((P[:, :2] - c[:2]) @ d)), "blob_vs_tcp_xy_mm": float(1000 * np.linalg.norm(c[:2] - c0[:2]))})
        else:
            c = c0; axis = Tg[:3, 0] - Tg[:3, 2] * (Tg[:3, 0] @ Tg[:3, 2]); axis[2] = 0; axis /= np.linalg.norm(axis); info["blob_px"] = 0
        c[2] = z
        zax = axis; xax = np.array([0, 0, 1.0]); yax = np.cross(zax, xax)
        T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2] = xax, yax, zax; T[:3, 3] = c - zax * BAT_L / 2
        self.info["battery"] = info
        return T

    def _backproject(self, uv, z):
        K = self.cam.K
        r = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], 1) @ self.cam.R.T
        lam = (z - self.cam.t[2]) / r[:, 2]
        return self.cam.t[None] + lam[:, None] * r

    def _attach(self):
        self.XR = np.linalg.inv(self.tcp("R", self.tg + 3)) @ self.table
        Tb_r = self.tcp("R", self.tr) @ self.XR
        self.XL = np.linalg.inv(self.tcp("L", self.tr)) @ Tb_r

    def holder(self, t):
        return "table" if t < self.tg + 3 else ("R" if t < self.tr else "L")

    def source_can(self, robot, t_state):
        h = self.holder(int(round(t_state)))
        if h == "table":
            return self.table
        return self.tcp(h, t_state) @ (self.XR if h == "R" else self.XL)

    # ------------------------------------------------------------ rendering helpers (robot + battery silhouettes)
    def _bat_points(self):
        rng = np.random.default_rng(2); n = 6000
        th = 2 * np.pi * rng.random(n); zz = BAT_L * rng.random(n)
        return np.stack([BAT_R * np.cos(th), BAT_R * np.sin(th), zz], 1)

    def silhouettes(self, j, battery=True, sides=False):
        Ms = self.G.placements(self.gv["R"][j], self.gv["L"][j], self.grip_v["R"][j], self.grip_v["L"][j])
        _, lab, _ = mesh.splat(self.G.world(Ms), self.G.gid, self.cam, dev=self.dev)
        side = np.array(self.G.side + ["X"])[lab.cpu().numpy()]
        robot = (side == "R") | (side == "L")
        if sides:
            return side
        bat = np.zeros_like(robot)
        if battery:
            Tb = self.source_can(self.robot, j + self.lag); P = (Tb[:3, :3] @ self._bat_points().T).T + Tb[:3, 3]
            uv, zc = self.cam.project(P); ok = (zc > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
            bat[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = True
            bat = cv2.dilate(bat.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        return robot, bat

    def _plate(self, dil=15):
        occ = np.zeros((self.T, 480, 640), bool)
        for j in range(self.T):
            r, b = self.silhouettes(j)
            occ[j] = cv2.dilate((r | b).astype(np.uint8), np.ones((dil, dil), np.uint8)) > 0
        plate = np.zeros((480, 640, 3), np.uint8)
        F = torch.tensor(self.F, device=self.dev); O = torch.tensor(occ, device=self.dev)
        for c in range(3):
            x = F[..., c].half().masked_fill(O, float("nan"))
            med = torch.nanmedian(x.view(self.T, -1), dim=0).values.view(480, 640)
            plate[..., c] = torch.nan_to_num(med, nan=0).clamp(0, 255).byte().cpu().numpy()
        never = occ.all(0); self.info["plate_never_free_frac"] = float(never.mean())
        del F, O; torch.cuda.empty_cache()
        return cv2.inpaint(plate, never.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)

    def _lag_scan_scores(self, lags=np.arange(-4.0, 1.51, 0.25)):
        js = np.linspace(40, self.T - 40, 40).astype(int)
        fg = [cv2.morphologyEx((np.abs(self.F[j].astype(np.int16) - self.plate.astype(np.int16)).max(2) > 25).astype(np.uint8),
                               cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0 for j in js]
        best, scores = None, {}
        for L in lags:
            self.lag = float(L); self._video_states(); s = []
            for j, f in zip(js, fg):
                r, _ = self.silhouettes(j, battery=False)
                roi = cv2.dilate(r.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
                ff = f & roi; s.append((ff & r).sum() / max(1, (ff | r).sum()))
            scores[float(L)] = float(np.mean(s))
        self.info["lag_scan"] = scores
        return scores

    def _masks(self):
        self.masks = {"R": np.zeros((self.T, 480, 640), np.uint8), "L": np.zeros((self.T, 480, 640), np.uint8),
                      "can": np.zeros((self.T, 480, 640), np.uint8)}
        for j in range(self.T):
            _, b = self.silhouettes(j)
            fg = np.abs(self.F[j].astype(np.int16) - self.plate.astype(np.int16)).max(2) > 20
            self.masks["can"][j] = (cv2.dilate(b.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & fg
            # arm masks: z-buffered URDF mesh silhouettes (this dataset has no SAM2 masks; the black arm on the dark table
            # is not separable by background difference). Plays the role SAM2 masks play in the Pringles pipeline.
            sd = self.silhouettes(j, battery=False, sides=True)
            self.masks["R"][j] = sd == "R"; self.masks["L"][j] = sd == "L"
        self.masks["R"][self.masks["can"] > 0] = 0; self.masks["L"][self.masks["can"] > 0] = 0
