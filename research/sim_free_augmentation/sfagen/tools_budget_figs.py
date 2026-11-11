"""Per-budget diversity figures for the Pringles source-budget datasets: can (pick-up) position and hand-over position of
the N real sources, their synthetic episodes, and the 40 other real recordings (never used for generation).
Writes logs/budget/positions_{N}.png, logs/budget/positions_overview.png and logs/budget/diversity.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
S = Path(__file__).resolve().parent; sys.path.insert(0, str(S))
from sfagen.context import Context
ctx = Context(S / "configs/openarm_pringles_handover_dense.yaml"); rob = ctx.robot; D = ctx.work
sel = json.loads((S / "logs/budget_selection.json").read_text())
POOL = Path.home() / ".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d/data"


def positions(Sx, Ax):
    ev = rob.gripper_events(Ax); TR = rob.tcp_from_state(Sx, "R")[:, :3, 3]
    return {"can": 100 * TR[ev["R"]["closes"][0], :2], "handover": 100 * TR[[c for c in ev["L"]["closes"] if c > ev["R"]["closes"][0]][0], :2]}


src = {e: positions(*[x.astype(float) for x in ctx.src.arrays(e)]) for e in range(10)}
syn = {}
for e, k, _ in sel["9"]["episodes"]:
    z = np.load(D / f"candidates/ep{e}/{k}.npz"); syn[(e, k)] = positions(z["state"].astype(float), z["action"].astype(float))
pdf = pd.read_parquet(sorted(POOL.rglob("*.parquet"))); pool = {"can": [], "handover": []}
for e, d in pdf.groupby("episode_index"):
    if int(e) < 10:
        continue
    d = d.sort_values("frame_index")
    try:
        p = positions(np.stack(d["observation.state"].to_numpy()).astype(float), np.stack(d["action"].to_numpy()).astype(float))
    except (IndexError, KeyError):
        continue
    for k in p:
        pool[k].append(p[k])
pool = {k: np.array(v) for k, v in pool.items()}


def hull(P):
    P = np.unique(np.round(P, 2), axis=0)
    return float(ConvexHull(P).volume) if len(P) >= 3 else 0.0


def stats(P, k):
    d = np.min(np.linalg.norm(pool[k][:, None] - P[None], axis=2), axis=1)
    return {"distinct_positions": int(len(np.unique(np.round(P, 1), axis=0))), "x_range_cm": float(np.ptp(P[:, 0])), "y_range_cm": float(np.ptp(P[:, 1])),
            "hull_area_cm2": hull(P), "unseen_real_within_2cm": int((d <= 2).sum()), "unseen_real_total": len(pool[k]), "median_dist_unseen_real_cm": float(np.median(d))}


INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"
C_POOL, C_SRC, C_SYN = "#b9b8b3", "#0b0b0b", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
lim = {k: (np.concatenate([pool[k], np.array([v[k] for v in syn.values()])]).min(0) - 2, np.concatenate([pool[k], np.array([v[k] for v in syn.values()])]).max(0) + 2) for k in pool}


def panel(ax, n, k, title=True, legend=False):
    sub = sel[str(n)]["sources"]; sp = np.array([syn[(e, kk)][k] for e, kk, _ in sel[str(n)]["episodes"]]); rp = np.array([src[e][k] for e in sub])
    ax.scatter(*pool[k].T, s=26, c=C_POOL, marker="o", lw=0, label=f"other real recordings ({len(pool[k])}, unseen)")
    ax.scatter(*sp.T, s=22, c=C_SYN, marker="o", edgecolors=SURF, linewidths=0.8, label=f"synthetic ({len(sp)})")
    ax.scatter(*rp.T, s=60, c=C_SRC, marker="X", edgecolors=SURF, linewidths=0.8, label=f"real source demos ({len(rp)})")
    ax.set_xlim(lim[k][0][0], lim[k][1][0]); ax.set_ylim(lim[k][0][1], lim[k][1][1]); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.5)
    st = stats(np.concatenate([rp, sp]), k)
    if title:
        ax.set_title(f"N={n}: {k.replace('can', 'can pick-up').replace('handover', 'hand-over')} | {st['unseen_real_within_2cm']}/{st['unseen_real_total']} unseen within 2 cm", fontsize=8.5)
    if legend:
        ax.legend(loc="upper left", fontsize=7)
    return st


div = {}
for n in range(1, 10):
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    div[n] = {"sources": sel[str(n)]["sources"], "synthetic_episodes": len(sel[str(n)]["episodes"]),
              "families": {f: sum(k.startswith(p) for _, k, _ in sel[str(n)]["episodes"]) for f, p in (("object_position", "can"), ("interaction_location", "interact"))},
              "sources_only": {k: stats(np.array([src[e][k] for e in sel[str(n)]["sources"]]), k) for k in pool}}
    div[n]["sources_plus_synthetic"] = {k: panel(ax, n, k, legend=(k == "can")) for ax, k in zip(axs, pool)}
    for ax in axs:
        ax.set_xlabel("x (cm, robot base frame)"); ax.set_ylabel("y (cm)")
    fig.suptitle(f"openarm_pringles_lerobot_real_to_sim_{n}_3cam_v00: positions from {n} real source demo{"s" if n > 1 else ""} + {len(sel[str(n)]['episodes'])} synthetic episodes", fontweight="bold", fontsize=10)
    fig.savefig(S / f"logs/budget/positions_{n}.png", dpi=140); plt.close(fig)
fig, axs = plt.subplots(2, 9, figsize=(27, 6.6), constrained_layout=True)
for i, n in enumerate(range(1, 10)):
    for j, k in enumerate(pool):
        panel(axs[j, i], n, k, title=False, legend=(i == 0 and j == 0)); axs[j, i].set_xticklabels([]); axs[j, i].set_yticklabels([])
        s = div[n]["sources_plus_synthetic"][k]
        axs[j, i].set_title(f"N={n} {'can' if k == 'can' else 'hand-over'}: {s['unseen_real_within_2cm']}/{s['unseen_real_total']} within 2 cm", fontsize=9)
fig.suptitle("Pringles: can pick-up (top) and hand-over (bottom) positions vs number of real source demos (black X = sources, orange = synthetic, grey = 40 unseen real recordings)", fontweight="bold", fontsize=12)
fig.savefig(S / "logs/budget/positions_overview.png", dpi=110); plt.close(fig)
(S / "logs/budget/diversity.json").write_text(json.dumps(div, indent=1))
for n, d in div.items():
    print(n, d["synthetic_episodes"], d["families"], {k: (d["sources_only"][k]["unseen_real_within_2cm"], v["unseen_real_within_2cm"], v["distinct_positions"], round(v["x_range_cm"], 1), round(v["y_range_cm"], 1))
                                                     for k, v in d["sources_plus_synthetic"].items()})
