"""Evaluation only (after sfagen ran): fitted / initial camera vs simulator truth. Rotation error (deg), centre error
(cm), focal ratio, and reprojection error (px) of a 3-D grid over the arena (table plane +- 15 cm).
Usage: camera_error.py CONFIG TRUTH_JSON [WORK_DIR]"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
from sfagen.camera import Camera
from scipy.spatial.transform import Rotation as Rot

ctx = Context(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None); tj = json.load(open(sys.argv[2]))
truth = Camera({"intrinsics": tj["intrinsics"], "extrinsics": tj})
g = np.stack(np.meshgrid(np.linspace(0.45, 0.95, 11), np.linspace(-0.35, 0.35, 15), np.linspace(0.74, 0.90, 3)), -1).reshape(-1, 3)
uv_t, _ = truth.project(g); vis = (uv_t[:, 0] >= 0) & (uv_t[:, 0] < 640) & (uv_t[:, 1] >= 0) & (uv_t[:, 1] < 480)
out = {}
for name, c in (("initial", ctx.cam0), ("fitted", ctx.cam)):
    uv, _ = c.project(g); e = np.linalg.norm(uv - uv_t, axis=1)[vis]
    out[name] = {"rot_err_deg": float(np.degrees(Rot.from_matrix(c.R.T @ truth.R).magnitude())), "centre_err_cm": float(100 * np.linalg.norm(c.t - truth.t)),
                 "focal_ratio": float(c.K[0, 0] / truth.K[0, 0]), "reproj_px_median": float(np.median(e)), "reproj_px_p95": float(np.percentile(e, 95))}
print(json.dumps(out))
