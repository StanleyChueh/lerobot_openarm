"""R04 even per-position quota (results/r04/PROTOCOL.md). General: any robot/task, any episode groups.
Object position per episode: real demo -> source position; object_workspace -> target; interaction_* -> source position.
Cell = nearest point of the `step` lattice. Each cell holds at most Q episodes in the final source + synthetic set; real
demos are always kept and count first. Cells with the fewest candidates are filled first; within a cell the least-used
source demo wins, then the lowest distortion score, then the smaller shift. Writes selection.json and per-group
`<group>_export` dirs exactly like select_export.py (so `sfagen build --work-dir <group>_export` exports the selection).
Amendment 1 (N mode): `N=100` selects exactly N synthetic episodes: repeatedly the cell with the lowest count, ties by
largest distance to every occupied position (real demos + picks so far) = farthest-point sampling within each round.
Usage: balanced_select.py TASK_RUN_DIR GROUP_CONFIG [Q=2 | N=100] [STEP=0.03]   (refuses to overwrite)"""
import collections, json, os, shutil, sys
from pathlib import Path
import numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.distortion import score_file

run = Path(sys.argv[1]).resolve(); cfg = yaml.safe_load(Path(sys.argv[2]).read_text())
arg = sys.argv[3] if len(sys.argv) > 3 else "Q=2"; mode, val = (arg.split("=") if "=" in arg else ("Q", arg)); Q = N = int(val); step = float(sys.argv[4]) if len(sys.argv) > 4 else .03
src = {int(k): np.asarray(v, float) for k, v in cfg["variations"]["object_targets"]["source_positions_xy_m"].items()}
cell = lambda p: tuple(int(v) for v in np.round(np.asarray(p, float) / step))  # noqa: E731
groups = sorted(p for p in run.iterdir() if p.is_dir() and (p / "candidates").exists() and not p.name.endswith("_export"))
rows = []
for g in groups:
    for cj in sorted((g / "candidates").glob("ep*/checks.json")):
        e = int(cj.parent.name[2:])
        for k, v in json.loads(cj.read_text()).items():
            f = g / "body" / f"{e}_{k}.json"
            if not (v["pass"] and f.exists()):
                continue
            pos = src[e] + np.asarray(v["shift_m"][:2]) if v["family"] in ("object_workspace", "object_position") else src[e]
            rows.append({"group": g.name, "source": e, "key": k, "family": v["family"], "score": score_file(f),
                         "shift_cm": 100 * float(np.hypot(*v["shift_m"][:2])), "pos": [float(x) for x in pos], "cell": cell(pos)})
count = collections.Counter(cell(p) for p in src.values())                  # real demos count first
by_cell = collections.defaultdict(list)
for r in rows:
    by_cell[r["cell"]].append(r)
used = collections.Counter(); keep = []
pick = lambda pool: min(pool, key=lambda r: (used[r["source"]], r["score"], r["shift_cm"], r["key"]))  # noqa: E731
if mode == "Q":
    for c in sorted(by_cell, key=lambda c: (len(by_cell[c]), c)):            # scarce cells first
        pool = list(by_cell[c])
        while count[c] < Q and pool:
            r = pick(pool); pool.remove(r); keep.append(r); used[r["source"]] += 1; count[c] += 1
    assert all(n <= max(Q, sum(cell(p) == c for p in src.values())) for c, n in count.items())
else:                                                                         # N mode: rounds + farthest point
    pools = {c: list(v) for c, v in by_cell.items()}; centre = {c: np.array(c, float) * step for c in pools}
    occ = np.array(list(src.values())); dist = {c: float(np.linalg.norm(occ - centre[c], axis=1).min()) for c in pools}
    while len(keep) < N and any(pools.values()):
        c = min((c for c in pools if pools[c]), key=lambda c: (count[c], -dist[c], c))
        r = pick(pools[c]); pools[c].remove(r); keep.append(r); used[r["source"]] += 1; count[c] += 1
        p = np.asarray(r["pos"]); dist = {k: min(v, float(np.linalg.norm(centre[k] - p))) for k, v in dist.items()}
    assert len(keep) == min(N, len(rows))
assert not (run / "selection.json").exists(), "selection exists"
fam = collections.Counter(r["family"] for r in keep); fam_all = collections.Counter(r["family"] for r in rows)
stats = {"rule": (f"even quota Q={Q} per {step * 100:.0f} cm cell" if mode == "Q" else f"N={N} synthetic, rounds over {step * 100:.0f} cm cells, farthest-point") + " (real demos count first); least-used source, then distortion",
         "per_cell_counts_incl_real": dict(collections.Counter(n for n in count.values() if n)), "shortfall": max(0, N - len(keep)) if mode == "N" else None,
         "candidates": len(rows), "selected": len(keep), "families_selected": dict(fam), "families_available": dict(fam_all),
         "cells_with_episodes": len([c for c, n in count.items() if n]), "per_source": {int(k): used.get(k, 0) for k in sorted(src)}}
(run / "selection.json").write_text(json.dumps({**stats, "selection": [{k: v for k, v in r.items() if k != "cell"} for r in keep]}, indent=1))
for g in groups:
    d = run / f"{g.name}_export"; d.mkdir(); (d / "body").mkdir()
    for f in ("calibration.json", "table_height.json", "lowest_arm.json"):
        shutil.copy2(g / f, d / f)
    for s in ("cache", "candidates"):
        os.symlink(g.resolve() / s, d / s)
    for r in (r for r in keep if r["group"] == g.name):
        for ext in (".npy", ".json", ".fingerprint"):
            os.link(g / "body" / f"{r['source']}_{r['key']}{ext}", d / "body" / f"{r['source']}_{r['key']}{ext}")
print(json.dumps(stats))
