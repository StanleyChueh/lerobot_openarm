"""Source-demo vs synthetic-dataset distribution for the RoboTwin pipeline-v1 datasets (R02 follow-up).
Top-down maps (robot frame) of the source object positions, the candidate grid and the synthetic object positions;
synthetic episodes per source demo; shift distances; handover variation. Reads provenance + frozen source positions.
Usage: dataset_distribution.py RUN_DIR DATA_DIR CONFIG_DIR OUT_PREFIX"""
import json, sys, collections
from pathlib import Path
import numpy as np, yaml, matplotlib
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.target_plan import auto_targets
matplotlib.use("Agg"); import matplotlib.pyplot as plt

run, data, cdir, out = map(Path, sys.argv[1:5])
TASKS = ["place_empty_cup", "stack_blocks_two", "handover_block"]
fig, ax = plt.subplots(3, 3, figsize=(17, 15)); summary = {}
for j, t in enumerate(TASKS):
    root = data / t / "synthetic" if (data / t / "synthetic").exists() else data / t
    prov = json.loads((root / "meta/sfa_provenance_merged.json").read_text())["episodes"]
    src = {int(k): v for k, v in json.loads((run / "positions" / f"{t}.json").read_text())[t].items()}
    groups = {g.stem.split(f"{t}_")[1]: yaml.safe_load(g.read_text()) for g in cdir.glob(f"{t}_*.yaml") if not g.stem.endswith("base")}
    grid = set()
    for c in groups.values():
        ot = c["variations"]["object_targets"]
        pts = auto_targets(ot, {int(k): np.array(v) for k, v in ot["source_positions_xy_m"].items()}) if ot["positions_xy_m"] == "auto" else ot["positions_xy_m"]
        grid |= {tuple(np.round(p, 4)) for p in pts}
    per = collections.Counter(p["source_real_episode"] for p in prov); fam = collections.Counter(p["family"] for p in prov)
    sh = np.array([100 * np.hypot(*p["shift_m"][:2]) for p in prov])
    a = ax[0, j]; cmap = plt.get_cmap("tab10")
    g = np.array(sorted(grid)); a.scatter(g[:, 1], g[:, 0], s=12, c="lightgrey", marker="s", label=f"target lattice from source demos ({len(g)}, 3 cm)")
    syn = []
    for p in prov:
        e = p["source_real_episode"]; s0 = np.array(src[e]); s1 = s0 + np.array(p["shift_m"][:2])
        if p["family"] in ("object_workspace", "object_position"):
            syn.append(s1); a.plot([s0[1], s1[1]], [s0[0], s1[0]], color=cmap(e % 10), lw=.5, alpha=.4)
            a.scatter(s1[1], s1[0], s=14, color=cmap(e % 10), alpha=.8)
    for e, p in src.items():
        a.scatter(p[1], p[0], s=160, color=cmap(e % 10), edgecolor="k", marker="*", zorder=5)
        a.annotate(f"d{e}", (p[1], p[0]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    a.set_title(f"{t}: {len(src)} source demos (stars) -> {len(prov)} synthetic"
                + (f"\n{len(syn)} moved-object positions (dots)" if syn else "\n"))
    a.set_xlabel("robot y (m, left +)"); a.set_ylabel("robot x (m, forward)"); a.invert_xaxis(); a.set_aspect("equal"); a.grid(alpha=.3)
    a.legend(loc="lower left", fontsize=8)
    b = ax[1, j]; ks = sorted(src); b.bar([f"d{k}" for k in ks], [per.get(k, 0) for k in ks], color=[cmap(k % 10) for k in ks])
    b.set_title(f"synthetic episodes per source demo (used {len(per)}/{len(src)})"); b.set_ylabel("episodes")
    c = ax[2, j]
    if t == "handover_block":
        loc = collections.Counter(tuple(np.round(100 * np.array(p["shift_m"][:2])).astype(int)) for p in prov if p["family"] == "interaction_location")
        tim = collections.Counter(p["variation"] for p in prov if p["family"] == "interaction_timing")
        for (dx, dy), n in loc.items():
            c.scatter(dy, dx, s=60 * n, alpha=.6, color="C0"); c.annotate(str(n), (dy, dx), ha="center", va="center", fontsize=8)
        c.scatter(0, 0, marker="*", s=200, color="k", label="source handover point")
        c.set_title(f"handover location shifts (cm), n={sum(loc.values())}; timing: " + ", ".join(f"{k} x{v}" for k, v in sorted(tim.items())))
        c.set_xlabel("dy (cm)"); c.set_ylabel("dx (cm)"); c.invert_xaxis(); c.set_aspect("equal"); c.grid(alpha=.3); c.legend(fontsize=8)
    else:
        c.hist(sh, bins=np.arange(0, 13, 1), color="C1", edgecolor="k"); c.set_xlabel("object shift from its source demo (cm)")
        c.set_ylabel("episodes"); c.set_title(f"shift distance: median {np.median(sh):.1f} cm, max {sh.max():.1f} cm")
    xy = np.array(syn) if syn else None
    summary[t] = {"source_demos": len(src), "source_demos_used": len(per), "synthetic_episodes": len(prov), "families": dict(fam),
                  "per_source": {int(k): per.get(k, 0) for k in ks}, "shift_cm": {"min": float(sh.min()), "median": float(np.median(sh)), "max": float(sh.max())},
                  "source_xy_range_m": {"x": [min(p[0] for p in src.values()), max(p[0] for p in src.values())], "y": [min(p[1] for p in src.values()), max(p[1] for p in src.values())]},
                  "synthetic_xy_range_m": None if xy is None else {"x": [float(xy[:, 0].min()), float(xy[:, 0].max())], "y": [float(xy[:, 1].min()), float(xy[:, 1].max())]},
                  "families_note": "object_workspace = moved object; interaction_* = handover location/timing", "distinct_synthetic_targets": None if xy is None else len({tuple(np.round(r, 3)) for r in xy}), "grid_targets": len(g)}
fig.suptitle("RoboTwin pipeline-v1 datasets: source demos vs synthetic distribution (robot frame, top-down)", fontsize=14)
fig.tight_layout(); fig.savefig(f"{out}.png", dpi=90); Path(f"{out}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
