"""Target-position heatmaps + metrics: 10 real sources vs synthetic revisions, with the other real recordings as context.
Targets (target_positions.py, forward kinematics of recorded/published joint states, no annotation):
  Pringles hand-over point = right TCP at the left-gripper close; battery station pick = left TCP at the left-gripper
  close; battery insertion point = right TCP at the first right-gripper open.
Usage: target_heatmaps.py <out_dir>  (reads target_positions.json, which must contain the revisions listed in RUNS)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
HERE = Path(__file__).resolve().parent
SRC_C, SYN_C, CTX_C = "#2a78d6", "#eb6834", "#8f8e8a"; INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
BLUE = LinearSegmentedColormap.from_list("b", [SURF, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
ORANGE = LinearSegmentedColormap.from_list("o", [SURF, "#fbe0d3", "#f5ae8c", "#eb6834", "#c24e1f", "#8a3510", "#5e230a"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
RUNS = {"pringles": [("handover", "Hand-over point", ["synthetic_v3", "synthetic_v4"])],
        "battery": [("station_pick", "Station pick point", ["synthetic_v00", "synthetic_v01"]),
                    ("insertion", "Insertion point", ["synthetic_v00", "synthetic_v01"]),
                    ("battery_grasp", "Battery position (right hand at the battery grasp; object, not target)", ["synthetic_v00", "synthetic_v01"])]}


def density(P, ext, cell, sigma):
    nx, ny = int(round((ext[0][1] - ext[0][0]) / cell)), int(round((ext[1][1] - ext[1][0]) / cell))
    H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=(nx, ny), range=ext); H = gaussian_filter(H, sigma / cell, mode="constant"); return H.T / H.sum()


def hull(P):
    return float(ConvexHull(P).volume) if len(np.unique(P.round(3), axis=0)) >= 3 else 0.0


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    r = json.loads((HERE / "target_positions.json").read_text()); M = {"definition": r["definition"]}
    for task, items in RUNS.items():
        for key, label, revs in items:
            S = 100 * np.array([v[key] for _, v in sorted(r[task]["source"].items(), key=lambda x: int(x[0]))])[:, :2]
            P = 100 * np.array([v[key] for v in r[task]["pool"].values()])[:, :2]
            Ys = {rv: 100 * np.array([v[key] for v in r[task][rv].values()])[:, :2] for rv in revs if rv in r[task]}
            pad = 4.0; allp = np.r_[S, P, *Ys.values()]
            ext = ((np.floor(allp[:, 0].min() - pad), np.ceil(allp[:, 0].max() + pad)), (np.floor(allp[:, 1].min() - pad), np.ceil(allp[:, 1].max() + pad)))
            panels = [("10 real sources", S, BLUE)] + [(f"sources + {rv.split('_')[1]} ({len(Y)} synthetic)", np.r_[S, Y], ORANGE) for rv, Y in Ys.items()]
            D = [density(X, ext, 0.25, 1.0) for _, X, _ in panels]; vmax = max(d.max() for d in D)
            fig, axs = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.6), constrained_layout=True)
            for ax, (name, X, cmap), d in zip(axs, panels, D):
                ax.imshow(d, origin="lower", extent=(*ext[0], *ext[1]), cmap=cmap, norm=PowerNorm(1.0, 0, vmax), interpolation="bilinear")
                ax.scatter(P[:, 0], P[:, 1], s=16, facecolor="none", edgecolor=CTX_C, lw=0.9, label=f"{len(P)} other real recordings")
                if X is not S:
                    Y = X[len(S):]; ax.scatter(Y[:, 0], Y[:, 1], s=20, color=SYN_C, edgecolor=SURF, lw=1.0, zorder=5, label="synthetic")
                ax.scatter(S[:, 0], S[:, 1], s=30, color=SRC_C, edgecolor=SURF, lw=1.3, zorder=6, label="source")
                ax.grid(color=GRID, lw=0.5, alpha=0.6); ax.set_xlabel("world x (cm)"); ax.set_ylabel("world y (cm)"); ax.set_aspect("equal")
                ax.set_title(name); ax.legend(loc="upper left", fontsize=7)
            fig.suptitle(f"{task.capitalize()}: {label} (top-down, same colour scale per figure)", fontweight="bold", fontsize=11)
            fig.savefig(out / f"target_{task}_{key}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
            nn = lambda A, B: np.array([np.min(np.linalg.norm(B - a, axis=1)) for a in A])  # noqa: E731
            m = {"n_source": len(S), "n_other_real": len(P), "other_real_hull_cm2": hull(P),
                 "source": {"hull_cm2": hull(S), "other_real_nn_median_cm": float(np.median(nn(P, S))), "other_real_within_2cm": int((nn(P, S) <= 2).sum())}}
            for rv, Y in Ys.items():
                X = np.r_[S, Y]; prov_shift = None
                m[rv] = {"n": len(Y), "hull_cm2": hull(X), "other_real_nn_median_cm": float(np.median(nn(P, X))),
                         "other_real_within_2cm": int((nn(P, X) <= 2).sum()), "new_targets_gt_1cm_from_every_source": int((nn(Y, S) > 1.0).sum())}
            M[f"{task}/{key}"] = m
    (out / "target_metrics.json").write_text(json.dumps(M, indent=1)); print(json.dumps({k: v for k, v in M.items() if k != "definition"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
