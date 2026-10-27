"""Estimate the initial can XY of every evaluation episode (not logged) from its first body_cam frame.

The evaluation and sim_20_v00 episodes share one fixed body_cam (visual_domain.py: correlation >= 0.9997), so the
per-pixel median of the 200 evaluation first frames is the empty scene. The can is the largest blob of
|frame - background|; its silhouette centroid maps to table XY by a homography fitted on the 20 server-recorded
sim_20_v00 starts (recorded poses in IsaacLab/logs/demos/openarm_pringles_lerobot_sim_v00.hdf5), validated
leave-one-out. (A red-colour blob was tried first and rejected: it lands on different parts of the label
depending on the can's visible face, LOO 3.4 cm.) Generated starts cannot be used -- their cameras are randomized.

Also decodes key frames of every evaluation episode for the human-review contact sheets.
"""
import json
import sys

import cv2
import numpy as np
import pandas as pd

from common import DATA, DATASETS, REPO

sys.path.insert(0, str(REPO))
from tools.dataset_diagnostics.loader import load_dataset  # noqa: E402
from tools.dataset_diagnostics.positions import fit_validated_transform, apply_homography  # noqa: E402

CAM = "observation.images.body_cam"
sim20 = json.loads((DATA / "sim20_start_poses.json").read_text())
SETS = {"sim_20": DATASETS["eval_manual"].parent / "openarm_pringles_lerobot_sim_20_v00",
        "eval_manual": DATASETS["eval_manual"], "eval_generated": DATASETS["eval_generated"]}
views = {k: load_dataset(root=p, label=k, offline=True) for k, p in SETS.items()}

first = {k: [v.read_frame(CAM, e, 0) for e in range(v.info["total_episodes"])] for k, v in views.items()}
bg = np.median(np.stack(first["eval_manual"] + first["eval_generated"]).astype(np.int16), 0).astype(np.int16)


def silhouette(rgb, thr=35):
    m = (np.abs(rgb.astype(np.int16) - bg).max(-1) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == i)
    return {"u": float(xs.mean()), "v": float(ys.mean()), "area": int(len(xs))}


rows = []
for k, imgs in first.items():
    for ep, rgb in enumerate(imgs):
        s = silhouette(rgb) or {"u": np.nan, "v": np.nan, "area": 0}
        r = {"set": k, "episode": ep, **s}
        if k == "sim_20":
            r["x_true"], r["y_true"] = sim20["xy"][ep]
        rows.append(r)
df = pd.DataFrame(rows)
assert views["sim_20"].episodes["length"].tolist() == sim20["lengths"], "sim_20 dataset != recorded HDF5"

fit = df[df.set == "sim_20"]
T = fit_validated_transform(fit[["u", "v"]].to_numpy(), fit[["x_true", "y_true"]].to_numpy(), camera=CAM,
                            frame_id="isaac_env", max_allowed_m=0.02,
                            source="sim_20_v00 first frames (background-subtracted silhouette centroid) + recorded poses")
xy = apply_homography(T.H, df[["u", "v"]].to_numpy())
df["x_est"], df["y_est"] = xy[:, 0], xy[:, 1]
df["err_m"] = np.hypot(df.x_est - df.x_true, df.y_est - df.y_true)
a_lo, a_hi = fit.area.min(), fit.area.max()
df["area_in_fit_range"] = df.area.between(0.9 * a_lo, 1.1 * a_hi)
val = {
    "method": "background-subtracted silhouette centroid + homography",
    "fit_points": int(T.n_points), "loo_rmse_m": T.heldout_rmse_m, "loo_max_m": T.heldout_max_m, "valid(<=2cm)": bool(T.valid),
    "fit_x_range": [float(fit.x_true.min()), float(fit.x_true.max())],
    "fit_y_range": [float(fit.y_true.min()), float(fit.y_true.max())],
    "fit_area_range": [int(a_lo), int(a_hi)],
    "eval_area_outliers": df[(df.set != "sim_20") & ~df.area_in_fit_range][["set", "episode", "area"]].to_dict("records"),
    "eval_outside_fit_hull_frac": float(((df.set != "sim_20") & ((df.x_est < fit.x_true.min() - 0.01) | (df.x_est > fit.x_true.max() + 0.01) |
                                                                 (df.y_est < fit.y_true.min() - 0.01) | (df.y_est > fit.y_true.max() + 0.01))).sum() / 200),
}
print(json.dumps(val, indent=1))
df.to_csv(DATA / "start_positions.csv", index=False)
json.dump({"validation": val, "H": T.H.tolist()}, open(DATA / "start_position_estimator.json", "w"), indent=1)

frames = {}
for k in ["eval_manual", "eval_generated"]:
    v = views[k]
    for ep, L in enumerate(v.episodes["length"].to_numpy()):
        idx = [0, L // 4, L // 2, (3 * L) // 4, L - 1]
        frames[f"{k}__{ep}"] = np.stack([v.read_frame(CAM, ep, int(i))[::4, ::4] for i in idx])
np.savez_compressed(DATA / "eval_keyframes.npz", **frames)
