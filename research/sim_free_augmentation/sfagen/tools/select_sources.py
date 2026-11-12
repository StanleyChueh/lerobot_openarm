"""C1: source-only demo selection rule (frozen before any coverage number; results/fewsrc/SELECTION.md).

Greedy farthest-point (k-center) over the demos' recorded object positions, i.e. the FK of the grasping arm's TCP at its
first gripper close (the same position proxy as the coverage tool). Start: the demo nearest the centroid. Uses only the
source demos, never the evaluation pool.
usage: select_sources.py CONFIG OUT.json"""
import json, sys
from pathlib import Path
import numpy as np


def kcenter(P, n):
    """Indices of n points: nearest-to-centroid first, then repeatedly the point farthest from those chosen (ties: lowest index)."""
    P = np.asarray(P, float); sel = [int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    while len(sel) < min(n, len(P)):
        i = int(np.argmax(d)); sel.append(i); d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return sel


def main(cfg, out):
    from _provenance import HERE, git_state
    sys.path.insert(0, str(HERE))
    from sfagen.context import Context
    ctx = Context(HERE / cfg); r = ctx.robot; E = ctx.episodes(); pos = {}
    for e in E:
        S, A = ctx.src.arrays(e); ev = r.gripper_events(A)
        for name, o in ctx.cfg["objects"].items():
            t = ev[o["grasped_by"]]["closes"][o.get("grasp_index", 0)]; pos[e] = r.tcp_from_state(S[[t]], o["grasped_by"])[0, :2, 3].tolist()
            break                                                      # first configured object
    order = [E[i] for i in kcenter([pos[e] for e in E], len(E))]
    res = {"rule": "k-center over recorded object positions (FK at first grasp close), start nearest the centroid", "git": git_state(),
           "source_dataset": ctx.cfg["source"]["repo_id"], "revision": ctx.cfg["source"]["revision"], "positions_xy_m": pos, "order": order,
           "subsets": {str(n): sorted(order[:n]) for n in (1, 2, 4, len(E))}}
    Path(out).write_text(json.dumps(res, indent=1)); return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2])["subsets"]))
