"""Source-demo budget study: how many real source demos does sfagen need?

Every synthetic episode is generated from exactly ONE source demo, so the synthetic set for any subset S of the sources is
the union of the per-source candidates. The only cross-episode quantities in the replay checks are the speed and
acceleration envelope (max over source demos) and the table-clearance limit (lowest arm point over source demos); here they
are recomputed from the demos IN S ONLY, so a 1-demo dataset never benefits from limits learned on other demos. All other
checks are relative to the candidate's own source. Camera calibration and table height are treated as rig calibration
(done once per rig) and reused; see the report for that assumption.

For every subset of the 10 sources (all 1,023) we score coverage of the OTHER real recordings of the task (never used for
generation): within 2 cm of a training position, median distance to the nearest training position, convex-hull area, for
`sources only` vs `sources + synthetic`. Positions by forward kinematics at gripper command events (as coverage.py).
usage: tools_source_budget.py pringles|battery"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.spatial import ConvexHull
S = Path(__file__).resolve().parent; sys.path.insert(0, str(S))
from sfagen.context import Context
from sfagen.checks import Checker
from sfagen.why import reasons
task = sys.argv[1]
CFG = {"pringles": "openarm_pringles_handover", "battery": "openarm_battery_insertion"}[task]
POOL = {"pringles": (Path.home() / ".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d/data", list(range(10))),
        "battery": (Path("/home/stanley/Stanley_ws/sfa_hf_export/battery/pool/data"), [15, 20, 21, 22, 23, 30, 32, 36, 45, 53])}[task]
SUFFIX = (sys.argv[2] if len(sys.argv) > 2 else "") + __import__("os").environ.get("TAG", ""); CFG = CFG + (sys.argv[2] if len(sys.argv) > 2 else ""); ctx = Context(S / f"configs/{CFG}.yaml"); W = ctx.work; ck = Checker(ctx); rob = ctx.robot; cfgc = {**ck.cfg}
ENV_DEP = {"speed", "accel", "table"}


def positions(Sx, Ax):
    ev = rob.gripper_events(Ax); TR = rob.tcp_from_state(Sx, "R")[:, :3, 3]; TL = rob.tcp_from_state(Sx, "L")[:, :3, 3]
    if task == "pringles":
        return {"can": TR[ev["R"]["closes"][0], :2], "handover": TR[[c for c in ev["L"]["closes"] if c > ev["R"]["closes"][0]][0], :2]}
    return {"battery": TR[ev["R"]["closes"][0], :2], "station": TL[ev["L"]["closes"][0], :2], "insertion": TR[ev["R"]["opens"][0], :2]}


# ---- sources: envelope pieces + positions
src = {}
for e in ctx.episodes():
    Sx, Ax = ctx.src.arrays(e); v, a = ck.kin(Ax)
    src[e] = {"v": v, "a": a, "z": float(ck.lowest(Sx).min()), "pos": positions(Sx, Ax)}
    print("source", e, flush=True)
# ---- candidates: envelope-independent pass + own kinematics + positions
cands = []
for e in ctx.episodes():
    p = W / f"candidates/ep{e}/checks.json"
    for key, c in json.loads(p.read_text()).items():
        npz = W / f"candidates/ep{e}/{key}.npz"
        if "error" in c or not npz.exists():
            continue
        r = set(reasons(c, ctx.cfg.get("checks", {}), c.get("real_lowest_arm_z_m", 0)))
        if r - ENV_DEP:
            continue                                            # fails a check that does not depend on the source set
        cap = float(__import__("os").environ.get("MAX_F2_CM", "inf"))
        if c["family"] == "interaction_location" and 100 * np.linalg.norm(c["shift_m"]) > cap + 1e-6:
            continue                                            # visual-quality cap on interaction shifts (MAX_F2_CM)
        z = np.load(npz); v, a = ck.kin(z["action"].astype(float))
        try:
            pos = positions(z["state"].astype(float), z["action"].astype(float))
        except (IndexError, KeyError):
            continue
        cands.append({"e": e, "key": key, "family": c["family"], "v": v, "a": a, "lowz": c["lowest_arm_z_m"], "pos": pos})
print("candidates passing all source-independent checks:", len(cands), flush=True)
# ---- pool (other real recordings)
pdf = pd.read_parquet(sorted(POOL[0].rglob("*.parquet"))); pool = {}
for e, d in pdf.groupby("episode_index"):
    if int(e) in POOL[1]:
        continue
    d = d.sort_values("frame_index")
    try:
        p = positions(np.stack(d["observation.state"].to_numpy()).astype(float), np.stack(d["action"].to_numpy()).astype(float))
    except (IndexError, KeyError):
        continue
    for k, v in p.items():
        pool.setdefault(k, []).append(v)
pool = {k: 100 * np.array(v) for k, v in pool.items()}; keys = list(pool)


def hull(P):
    return float(ConvexHull(P).volume) if len(P) >= 3 and len(np.unique(np.round(P, 2), axis=0)) >= 3 else 0.0


def score(points, k):
    P = pool[k]; X = 100 * np.array(points)
    d = np.min(np.linalg.norm(P[:, None] - X[None], axis=2), axis=1)
    return {"within_2cm": int((d <= 2).sum()), "median_nn_cm": float(np.median(d)), "hull_cm2": hull(X)}


cs = ck.cfg; rows = []
E = ctx.episodes()
for n in range(1, len(E) + 1):
    for sub in itertools.combinations(E, n):
        env_v = np.max([src[e]["v"] for e in sub], 0); env_a = np.max([src[e]["a"] for e in sub], 0); minz = min(src[e]["z"] for e in sub)
        ok = [c for c in cands if c["e"] in sub and (c["v"] / np.maximum(env_v, 1e-9)).max() <= cs["speed_ratio"]
              and (c["a"] / np.maximum(env_a, 1e-9)).max() <= cs["accel_ratio"] and c["lowz"] >= minz - cs["table_tol_m"]]
        r = {"n": n, "subset": list(sub), "synthetic": len(ok), **{f"syn_{f}": sum(c["family"] == f for c in ok) for f in ("object_position", "interaction_location")}}
        for k in keys:
            so = score([src[e]["pos"][k] for e in sub], k); sa = score([src[e]["pos"][k] for e in sub] + [c["pos"][k] for c in ok], k)
            r.update({f"{k}_src_{m}": v for m, v in so.items()}); r.update({f"{k}_aug_{m}": v for m, v in sa.items()})
        rows.append(r)
df = pd.DataFrame(rows)
out = {"task": task, "n_pool": {k: len(v) for k, v in pool.items()}, "candidates_source_independent_pass": len(cands), "per_n": {}}
for n, g in df.groupby("n"):
    nested = g[g.subset.apply(lambda s: s == E[:n])].iloc[0]
    d = {"subsets": len(g), "synthetic_mean": float(g.synthetic.mean()), "synthetic_min": int(g.synthetic.min()), "synthetic_max": int(g.synthetic.max()),
         "nested_first_n": {"subset": nested.subset, "synthetic": int(nested.synthetic)}}
    for k in keys:
        for m in ("within_2cm", "median_nn_cm", "hull_cm2"):
            for side in ("src", "aug"):
                col = g[f"{k}_{side}_{m}"]; d[f"{k}_{side}_{m}"] = {"mean": float(col.mean()), "min": float(col.min()), "max": float(col.max())}
            d[f"{k}_nested_{m}"] = {"src": float(nested[f"{k}_src_{m}"]), "aug": float(nested[f"{k}_aug_{m}"])}
        best = g.loc[g[f"{k}_aug_within_2cm"].idxmax()]; d[f"{k}_best_subset"] = {"subset": best.subset, "aug_within_2cm": int(best[f"{k}_aug_within_2cm"])}
    out["per_n"][int(n)] = d
(S / f"logs/source_budget_{task}{SUFFIX}.json").write_text(json.dumps(out, indent=1, default=float))
df.to_parquet(S / f"logs/source_budget_{task}{SUFFIX}_all_subsets.parquet")
for n, d in out["per_n"].items():
    print(n, f"syn {d['synthetic_mean']:.1f} [{d['synthetic_min']},{d['synthetic_max']}]", " | ".join(
        f"{k}: within2cm src {d[f'{k}_src_within_2cm']['mean']:.1f} -> aug {d[f'{k}_aug_within_2cm']['mean']:.1f} (best {d[f'{k}_aug_within_2cm']['max']:.0f})" for k in keys))
