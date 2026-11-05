"""v3.2 figures: calibration diagnostics, per-camera synchronisation curves, wrist-camera pose trajectories, provenance
fractions and an object-tracking overlay clip.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/fig_threecam.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from gen_wrist import CACHE, load_cand  # noqa: E402
from sfa3 import recon3  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402
from sfa2 import render2  # noqa: E402
from sfa32 import render_wrist as rw, wristcam as wc  # noqa: E402
from sfa.recover import interp_q, cylinder_points  # noqa: E402
from sfa.render import label, write_mp4  # noqa: E402

FIG = HERE / "outputs/figures"
COL = {"source": "#52514e", "gen": "#1baf7a", "R": "#eb6834", "L": "#2a78d6"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e1e0d9", "axes.grid": True,
                     "grid.color": "#e1e0d9", "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})


def fig_calibration():
    c = json.loads(wc.CALIB.read_text())
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for s in "RL":
        cv = c[s]["lag_curve_median_abs_angle_diff_deg"]
        x = sorted(float(k) for k in cv)
        axs[0].plot(x, [cv[str(k) if str(k) in cv else f"{k}"] for k in x], marker="o", color=COL[s], label=f"{wc.CAM_OF[s].split('.')[-1]} (min {c[s]['lag_frames']})")
    axs[0].set_xlabel("assumed video-to-state lag (frames)")
    axs[0].set_ylabel("median |rot angle(image) - rot angle(FK)| (deg)")
    axs[0].set_title("Wrist-camera synchronisation (mount-independent)", loc="left")
    axs[0].legend(fontsize=8)
    lab = ["nominal", "fitted"]
    for i, s in enumerate("RL"):
        f = c[s]["can_silhouette_fit"] if s == "L" else c[s]["can_silhouette_fit_constrained"]
        v = [f["iou_val_nominal"], f["iou_val_fitted"] if s == "L" else f["iou_val"]]
        axs[1].bar(np.arange(2) + (i - 0.5) * 0.38, v, 0.36, color=COL[s], label=wc.CAM_OF[s].split(".")[-1])
        he = c["mount_crosscheck_handeye_residual_deg_val"][s]
        hv = [he["nominal"], he["can_silhouette_fit"] if s == "L" else f["handeye_val_deg"]]
        axs[2].bar(np.arange(2) + (i - 0.5) * 0.38, hv, 0.36, color=COL[s], label=wc.CAM_OF[s].split(".")[-1])
    axs[1].set_xticks(range(2), lab)
    axs[1].set_ylabel("held-out can-silhouette IoU (eps 5-9)")
    axs[1].set_title("Mount fit: can silhouette agreement", loc="left")
    axs[1].legend(fontsize=8)
    axs[2].set_xticks(range(2), lab)
    axs[2].set_ylabel("held-out hand-eye rotation residual (deg)")
    axs[2].set_title("Mount check: image vs FK rotation", loc="left")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "wrist_calibration.png", dpi=150)
    plt.close(fig)


def fig_camera_poses(E=0, key="init_x-6__C__s1.0"):
    robot = rb.Robot()
    ep = data.load(E, cams=())
    cand = load_cand(E, key)
    g = rb.Robot.split(ep.state)
    T = len(ep.state)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for s in "RL":
        X, lag = wc.mount(s), wc.lag(s)
        Ps = np.array([wc.cam_pose(robot, interp_q(g["R"], t + lag), interp_q(g["L"], t + lag), s, X) for t in range(T)])
        Pn = np.array([wc.cam_pose(robot, interp_q(cand["q"]["R"], t + lag), interp_q(cand["q"]["L"], t + lag), s, X) for t in range(T)])
        axs[0].plot(100 * Ps[:, 0, 3], 100 * Ps[:, 1, 3], color=COL[s], ls="--", lw=1.2, label=f"{s} wrist cam, source")
        axs[0].plot(100 * Pn[:, 0, 3], 100 * Pn[:, 1, 3], color=COL[s], lw=2, label=f"{s} wrist cam, generated")
        d = 100 * np.linalg.norm(Pn[:, :3, 3] - Ps[:, :3, 3], axis=1)
        ang = [np.degrees(np.linalg.norm(cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0])) for a, b in zip(Ps, Pn)]
        axs[1].plot(d, color=COL[s], label=f"{s}")
        axs[2].plot(ang, color=COL[s], label=f"{s}")
    axs[0].scatter(*(100 * cand["can"][0][:2, 3]), marker="o", color=COL["gen"], s=60, label="new can")
    axs[0].set_aspect("equal")
    axs[0].set_title(f"ep{E} {key}\nwrist camera positions, top view (cm)", loc="left")
    axs[0].legend(fontsize=7)
    axs[1].set_title("camera displacement generated vs source (cm)", loc="left")
    axs[2].set_title("camera rotation generated vs source (deg)", loc="left")
    for a in axs[1:]:
        a.set_xlabel("frame")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"wrist_camera_poses_ep{E}_{key}.png", dpi=150)
    plt.close(fig)


def overlay_clip(E=3, key="init_x-3__C__s1.0"):
    """generated right wrist view with the commanded can outline (green), the detected can hull (red) and provenance."""
    robot = rb.Robot()
    rec = recon3.build(E, with_plate=False, log=lambda *a: None)
    cand = load_cand(E, key)
    gen = np.load(CACHE / f"ep{E}" / f"wristR_{key}.npy", mmap_mode="r")
    lab = np.load(CACHE / f"ep{E}" / f"wristlab_{key}.npz")["R"]
    X, lag = wc.mount("R"), wc.lag("R")
    P = cylinder_points([0, 0, 1.0], 24, 6)
    frames = []
    for i in range(max(0, rec.ev["right_close"] - 45), rec.ev["right_close"] + 10):
        im = np.asarray(gen[i]).copy()
        Tn = wc.cam_pose(robot, interp_q(cand["q"]["R"], i + lag), interp_q(cand["q"]["L"], i + lag), "R", X)
        can = render2.can_at(cand["can"], i + lag)
        uv, z = wc.camera("R", Tn).project((can[:3, :3] @ P.T).T + can[:3, 3])
        f = z > 0.05
        if f.sum() > 8:
            cv2.polylines(im, [cv2.convexHull(uv[f].astype(np.float32)).astype(np.int32)], True, (0, 255, 0), 2)
        m = rw.can_mask(im).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (255, 0, 0), 2)
        prov = cv2.resize(rw_colors()[lab[i]], (320, 240), interpolation=cv2.INTER_NEAREST)
        frames.append(np.concatenate([label(cv2.resize(im, (320, 240)), f"gen R wrist t={i}: green=commanded can, red=detected"),
                                      label(prov, "provenance")], 1))
    write_mp4(HERE / "outputs/clips" / f"ep{E}_{key}_right_wrist_object_overlay.mp4", frames, crf=22)


def rw_colors():
    from gen_wrist import COLORS
    return COLORS


def fig_provenance():
    s = json.loads((HERE / "outputs/episode_summary.json").read_text())
    names = list(rw.LAB)
    fam = {}
    for E, d in s.items():
        for k, v in d.items():
            if not isinstance(v, dict) or "provenance_frac" not in v:
                continue
            f = "identity" if k.startswith("identity") else ("new initial position" if k.startswith("init") else "original position")
            for side in "RL":
                fam.setdefault((f, side), []).append([v["provenance_frac"][side][n] for n in names])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    keys = sorted(fam)
    cols = ["#1baf7a", "#2a78d6", "#eda100", "#6250d6", "#e34948", "#52514e"]
    bottom = np.zeros(len(keys))
    for j, n in enumerate(names):
        v = np.array([np.mean(fam[k], axis=0)[j] for k in keys])
        ax.barh(np.arange(len(keys)), v, left=bottom, color=cols[j], label=n)
        bottom += v
    ax.set_yticks(range(len(keys)), [f"{f} | {'right' if s_ == 'R' else 'left'} wrist (n={len(fam[(f, s_)])})" for f, s_ in keys])
    ax.set_xlabel("fraction of generated wrist pixels (all frames)")
    ax.set_title("Pixel provenance of generated wrist views", loc="left")
    ax.legend(fontsize=7, ncol=6, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.tight_layout()
    fig.savefig(FIG / "wrist_provenance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_calibration()
    fig_camera_poses()
    if (HERE / "outputs/episode_summary.json").exists():
        fig_provenance()
    if (CACHE / "ep3" / "wristR_init_x-3__C__s1.0.npy").exists():
        overlay_clip()
    print("ok")
