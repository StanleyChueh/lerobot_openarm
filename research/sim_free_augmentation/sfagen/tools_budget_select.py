"""Per-budget episode selection for the source-budget datasets: sources 0..N-1 (nested, fixed), dense grid, interaction
shifts capped at 4.3 cm, replay checks with the speed/acceleration/table limits recomputed from the N sources only.
Writes logs/budget_selection.json: {N: {"sources": [...], "episodes": [[e, key, {subset-envelope ratios}], ...]}}."""
import json, sys
from pathlib import Path
import numpy as np
S = Path(__file__).resolve().parent; sys.path.insert(0, str(S))
from sfagen.context import Context
from sfagen.checks import Checker
from sfagen.why import reasons
ctx = Context(S / "configs/openarm_pringles_handover_dense.yaml"); ck = Checker(ctx); cs = ck.cfg; W = ctx.work; CAP = 4.3
src = {}
for e in range(10):
    Sx, Ax = ctx.src.arrays(e); v, a = ck.kin(Ax); src[e] = {"v": v, "a": a, "z": float(ck.lowest(Sx).min())}
cands = []
for e in range(10):
    for key, c in json.loads((W / f"candidates/ep{e}/checks.json").read_text()).items():
        npz = W / f"candidates/ep{e}/{key}.npz"
        if "error" in c or not npz.exists() or (set(reasons(c, ctx.cfg.get("checks", {}), 0)) - {"speed", "accel", "table"}):
            continue
        if c["family"] == "interaction_location" and 100 * np.linalg.norm(c["shift_m"]) > CAP + 1e-6:
            continue
        v, a = ck.kin(np.load(npz)["action"].astype(float)); cands.append((e, key, v, a, c["lowest_arm_z_m"]))
out = {}
for n in range(1, 10):
    sub = list(range(n)); ev = np.max([src[e]["v"] for e in sub], 0); ea = np.max([src[e]["a"] for e in sub], 0); mz = min(src[e]["z"] for e in sub)
    sel = []
    for e, key, v, a, lz in cands:
        if e not in sub:
            continue
        sr, ar = float((v / ev).max()), float((a / ea).max())
        if sr <= cs["speed_ratio"] and ar <= cs["accel_ratio"] and lz >= mz - cs["table_tol_m"]:
            sel.append([e, key, {"speed_ratio_to_subset_max": sr, "accel_ratio_to_subset_max": ar, "subset_lowest_arm_z_m": mz,
                                 "envelope_sources": sub}])
    out[n] = {"sources": sub, "episodes": sel}
    fam = [k for _, k, _ in sel]; print(n, "sources", sub, "episodes", len(sel), "| object", sum(k.startswith("can") for k in fam), "| hand-over", sum(k.startswith("interact") for k in fam), flush=True)
(S / "logs/budget_selection.json").write_text(json.dumps(out, indent=1))
