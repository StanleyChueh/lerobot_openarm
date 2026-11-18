"""Before rendering: pick N passing candidates with the R04 N-mode rule (rounds over 3 cm cells, real demos count first,
farthest point, least-used source demo, then the smaller shift) - balanced_select.py without the distortion score, which
needs renders. Lets a long-episode task render only what it will export; balanced_select.py then runs on the renders.
Usage: preselect_render.py RUN_WORK_DIR CONFIG N [STEP=0.03]  -> prints "episode key" lines"""
import collections, json, sys
from pathlib import Path
import numpy as np, yaml

w, cfg, N = Path(sys.argv[1]), yaml.safe_load(Path(sys.argv[2]).read_text()), int(sys.argv[3]); step = float(sys.argv[4]) if len(sys.argv) > 4 else .03
src = {int(k): np.asarray(v, float) for k, v in cfg["variations"]["object_targets"]["source_positions_xy_m"].items()}
cell = lambda p: tuple(int(v) for v in np.round(np.asarray(p, float) / step))  # noqa: E731
rows = []
for cj in sorted(w.glob("candidates/ep*/checks.json")):
    e = int(cj.parent.name[2:])
    for k, v in json.loads(cj.read_text()).items():
        if v["pass"] and v["family"] in ("object_workspace", "object_position"):
            p = src[e] + np.asarray(v["shift_m"][:2]); rows.append({"source": e, "key": k, "pos": p, "shift": float(np.hypot(*v["shift_m"][:2])), "cell": cell(p)})
count = collections.Counter(cell(p) for p in src.values()); pools = collections.defaultdict(list)
for r in rows:
    pools[r["cell"]].append(r)
centre = {c: np.array(c, float) * step for c in pools}; occ = np.array(list(src.values()))
dist = {c: float(np.linalg.norm(occ - centre[c], axis=1).min()) for c in pools}; used = collections.Counter(); keep = []
while len(keep) < N and any(pools.values()):
    c = min((c for c in pools if pools[c]), key=lambda c: (count[c], -dist[c], c))
    r = min(pools[c], key=lambda r: (used[r["source"]], r["shift"], r["key"])); pools[c].remove(r); keep.append(r); used[r["source"]] += 1; count[c] += 1
    dist = {k: min(v, float(np.linalg.norm(centre[k] - r["pos"]))) for k, v in dist.items()}
print(f"# passing {len(rows)} in {len(set(r['cell'] for r in rows))} cells; picked {len(keep)}", file=sys.stderr)
for r in keep:
    print(r["source"], r["key"])
