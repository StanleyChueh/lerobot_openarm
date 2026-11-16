"""R04 density report: training-set object-position density (real demos + synthetic episodes) before (R03 / v03r2) and
after (R04 N=100 even selection). Density at a point = number of training episodes within 4 cm. Evenness over the
target lattice (3 cm points within 12 cm of a source): coefficient of variation and p90/p10 of density over lattice
points that are covered in the AFTER set; OpenArm also per evaluation region (mean over the region's eval starts).
Usage: density_report.py OUT_PREFIX   (paths are fixed to the R03/R04 runs)"""
import json, sys
from pathlib import Path
import numpy as np, yaml, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
WT = Path(__file__).resolve().parents[4]; SF = WT / "research/sim_free_augmentation/sfagen"; SB = WT / "research/sim_free_augmentation/simbench"
sys.path.insert(0, str(SF)); from sfagen.target_plan import auto_targets
RUNS = WT / "artifacts/simbench_runs"; R = 0.04; out = Path(sys.argv[1])


def dens(train, pts):
    return np.array([(np.linalg.norm(train - p, axis=1) < R).sum() for p in pts])


def even(d):
    d = d[d > 0] if (d > 0).any() else d
    return {"mean": float(d.mean()), "cv": float(d.std() / max(d.mean(), 1e-9)), "p90_over_p10": float(np.percentile(d, 90) / max(np.percentile(d, 10), 1e-9)),
            "min": int(d.min()), "max": int(d.max())}


rows = {}; panels = []
for t in ("place_empty_cup", "stack_blocks_two", "handover_block"):
    src = {int(k): np.array(v) for k, v in json.loads((RUNS / f"s1e/positions/{t}.json").read_text())[t].items()}; S = np.array(list(src.values()))
    c = yaml.safe_load(sorted((SB / "configs/robotwin").glob(f"{t}_[A-Z]*.yaml"))[0].read_text())["variations"]["object_targets"]
    lat = auto_targets(c, src)
    before = [src[p["source_real_episode"]] + (np.array(p["shift_m"][:2]) if p["family"] == "object_workspace" else 0)
              for p in json.loads((WT / f"artifacts/simbench_data/robotwin_sfagen_v1r3/{t}/synthetic/meta/sfa_provenance_merged.json").read_text())["episodes"]]
    after = [np.array(r["pos"]) for r in json.loads((RUNS / f"s1g/{t}/selection.json").read_text())["selection"]]
    B, A = np.vstack([S, before]), np.vstack([S, after]); dB, dA = dens(B, lat), dens(A, lat)
    rows[t] = {"synthetic_before": len(before), "synthetic_after": len(after), "lattice_points": len(lat),
               "before": even(dB), "after": even(dA)}
    panels.append((t, lat, S, dB, dA, len(before), len(after)))
# OpenArm: published v03r2 loose vs R04 N=100 (pipeline v1 candidates)
cfg = yaml.safe_load((SF / "configs/openarm_pringles_v03r2_loose.yaml").read_text())["variations"]["object_targets"]
src = {int(k): np.array(v) for k, v in cfg["source_positions_xy_m"].items()}; S = np.array(list(src.values()))
P = np.array(json.loads((SF / "results/v02/planned_targets.json").read_text())["eval_xy_m"])
grp = [("front-right", 0, 4), ("rear-right", 4, 8), ("rear-left", 8, 11), ("front-left", 11, 16), ("middle", 16, 20)]
prov = json.loads(Path("/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03r2_run/loose_copied/dataset/meta/sfa_provenance.json").read_text())["episodes"]
before = [src[p["source_real_episode"]] + (np.array(p["shift_m"][:2]) if p["family"] == "object_workspace" else 0) for p in prov]
sel = RUNS / "r04_openarm/selection.json"
if sel.exists():
    after = [np.array(r["pos"]) for r in json.loads(sel.read_text())["selection"]]
    lat = np.array(cfg["positions_xy_m"]); B, A = np.vstack([S, before]), np.vstack([S, after])
    reg = lambda T: {n: float(np.mean(dens(T, P[a:b]))) for n, a, b in grp}  # noqa: E731
    rb, ra = reg(B), reg(A)
    rows["openarm_pringles"] = {"synthetic_before": len(before), "synthetic_after": len(after), "lattice_points": len(lat),
                                "before": even(dens(B, lat)), "after": even(dens(A, lat)),
                                "eval_regions_before": rb, "eval_regions_after": ra,
                                "region_max_over_min_before": max(rb.values()) / max(min(rb.values()), 1e-9),
                                "region_max_over_min_after": max(ra.values()) / max(min(ra.values()), 1e-9)}
    panels.append(("openarm_pringles", lat, S, dens(B, lat), dens(A, lat), len(before), len(after)))
out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
fig, ax = plt.subplots(len(panels), 2, figsize=(12, 4.6 * len(panels)), squeeze=False)
for i, (t, lat, S, dB, dA, nb, na) in enumerate(panels):
    vmax = max(dB.max(), dA.max())
    for j, (d, lab) in enumerate(((dB, f"before ({nb} synthetic + 10 real)"), (dA, f"after: even N={na} (+10 real)"))):
        a = ax[i, j]; sc = a.scatter(lat[:, 1], lat[:, 0], c=d, cmap="viridis", vmin=0, vmax=vmax, s=60, marker="s")
        a.scatter(S[:, 1], S[:, 0], marker="*", s=150, c="red", edgecolor="k", label="real demos")
        e = even(d); a.set_title(f"{t}: {lab}\ndensity within 4 cm: CV {e['cv']:.2f}, p90/p10 {e['p90_over_p10']:.1f}", fontsize=10)
        a.invert_xaxis(); a.set_aspect("equal"); a.set_xlabel("robot y (m)"); a.set_ylabel("robot x (m)")
        if t == "openarm_pringles":
            a.scatter(P[:, 1], P[:, 0], facecolors="none", edgecolors="w", s=40, label="eval starts")
        a.legend(fontsize=7, loc="lower left"); fig.colorbar(sc, ax=a, fraction=.04)
fig.tight_layout(); fig.savefig(out.with_suffix(".png"), dpi=80)
print(json.dumps(rows, indent=1))
