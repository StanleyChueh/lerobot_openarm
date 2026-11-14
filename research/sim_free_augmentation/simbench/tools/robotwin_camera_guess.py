"""Initial guess for the RoboTwin head camera, as a user of a real rig would have it: the nominal mount + a seeded
error. Truth = embodiment config (static_camera_list head_camera, world frame, SAPIEN axes forward/left/up) moved into
the robot root frame (robot_pose, quaternion wxyz); demo_clean has random_head_camera_dis = 0, so truth is exact.
Guess = truth rotated by DEG about a random axis and shifted by M metres in a random direction (seed fixed) — the
plan's "extrinsic guess = truth + a seeded offset <= 10 cm / 10 deg". Intrinsics: D435 fovy 37 deg, square pixels,
principal point at the centre (public sensor spec). Truth goes to TRUTH_JSON only; sfagen sees the guess.
Usage: robotwin_camera_guess.py <aloha-agilex dir> <camera config yml> TRUTH_JSON [--seed 0 --deg 8 --m 0.08]"""
import argparse, json
from pathlib import Path
import numpy as np, yaml
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser(); ap.add_argument("emb"); ap.add_argument("camcfg"); ap.add_argument("truth")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--deg", type=float, default=8.0); ap.add_argument("--m", type=float, default=0.08)
ap.add_argument("--width", type=int, default=640); ap.add_argument("--height", type=int, default=480)
a = ap.parse_args()
cfg = yaml.safe_load((Path(a.emb) / "config.yml").read_text())
head = next(c for c in cfg["static_camera_list"] if c["name"] == "head_camera")
fovy = yaml.safe_load(Path(a.camcfg).read_text())[head["type"]]["fovy"]
rp = np.asarray(cfg["robot_pose"][0], float); R_wb = Rot.from_quat(rp[3:], scalar_first=True).as_matrix(); p_wb = rp[:3]
f = np.asarray(head["forward"], float); f /= np.linalg.norm(f); l = np.asarray(head["left"], float); l /= np.linalg.norm(l); up = np.cross(f, l)
R_wc = np.stack([-l, -up, f], 1)                      # OpenCV axes (x right, y down, z forward) in the world
R = R_wb.T @ R_wc; t = R_wb.T @ (np.asarray(head["position"], float) - p_wb)
fy = a.height / 2 / np.tan(np.radians(fovy) / 2)
K = {"width": a.width, "height": a.height, "fx": round(float(fy), 3), "fy": round(float(fy), 3), "cx": a.width / 2, "cy": a.height / 2}
rng = np.random.default_rng(a.seed); ax = rng.normal(size=3); ax /= np.linalg.norm(ax); dt = rng.normal(size=3); dt /= np.linalg.norm(dt)
Rg = R @ Rot.from_rotvec(ax * np.radians(a.deg)).as_matrix(); tg = t + dt * a.m
q = lambda M: Rot.from_matrix(M).as_quat(scalar_first=True).round(6).tolist()  # noqa: E731
Path(a.truth).parent.mkdir(parents=True, exist_ok=True)
Path(a.truth).write_text(json.dumps({"position": t.tolist(), "quat_wxyz": q(R), "intrinsics": K, "source": "RoboTwin embodiment config",
                                     "guess_offset": {"seed": a.seed, "deg": a.deg, "m": a.m}}, indent=1))
print(yaml.safe_dump({"camera": {"intrinsics": K, "extrinsics": {"position": tg.round(5).tolist(), "quat_wxyz": q(Rg)},
                                 "note": f"initial guess = nominal RoboTwin head-camera mount + seeded error (seed {a.seed}, {a.deg} deg, {a.m} m); refined by `sfagen calibrate --search`"}},
                     default_flow_style=None, sort_keys=False))
