"""Do v00 (13d7231a) synthetic labels stay inside URDF ∪ source envelope, esp. those covering rear-right? (Claude audit)
usage: python check_v00_limits.py <v00_snapshot_dir> <config.yaml> <results_dir> [out.json]
"""
import json, sys, glob, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, pandas as pd, yaml

snap, cfg, res = Path(sys.argv[1]), yaml.safe_load(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
       for j in ET.parse(cfg["robot"]["urdf"]).getroot().iter("joint") if j.find("limit") is not None and j.get("type") == "revolute"}
cols = {i: n for s in cfg["robot"]["arms"].values() for n, i in zip(s["joints"], s["state_indices"])}
prov = json.loads((snap / "meta/sfa_provenance.json").read_text())
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(snap / "data/*/*.parquet")))]).sort_values("index")
arr = lambda g, k: np.stack(g[k].to_numpy()).astype(float)
E = {e: (arr(g, "observation.state"), arr(g, "action")) for e, g in df.groupby("episode_index")}
real = {p["source_real_episode"] if "source_real_episode" in p else p.get("real_episode", p["episode_index"]): p["episode_index"]
        for p in prov["episodes"] if not p["synthetic"]}
cov = json.loads((res / "eval_v00/coverage.json").read_text()); realxy = np.array(cov["real_xy_cm"])
P = np.array(json.loads((res / "v02/planned_targets.json").read_text())["eval_xy_m"]) * 100
rows = []
for p in prov["episodes"]:
    if not p["synthetic"]: continue
    s, a = E[p["episode_index"]]; S0, A0 = E[real[p["source_real_episode"]]]
    worst = (0.0, None)
    for X, Y, f in ((s, S0, "state"), (a, A0, "action")):
        for i, n in cols.items():
            ex = max(min(lim[n][0], Y[:, i].min()) - X[:, i].min(), X[:, i].max() - max(lim[n][1], Y[:, i].max()))
            if ex > worst[0]: worst = (ex, f"{f}:{n}")
    d = (np.array(p["can_initial_world_m"][:2]) - np.array(p["source_can_initial_world_m"][:2])) * 100
    xy = realxy[p["source_real_episode"]] + d
    rows.append(dict(ep=p["episode_index"], variation=p["variation"], family=p["family"], xy_cm=xy.round(2).tolist(),
                     excess_deg=round(float(np.degrees(worst[0])), 3), where=worst[1],
                     nn_rear_right_cm=round(float(np.linalg.norm(P[4:8] - xy, axis=1).min()), 2)))
bad = [r for r in rows if r["excess_deg"] > 1e-4]
rr = sorted(rows, key=lambda r: r["nn_rear_right_cm"])[:6]
out = dict(n_synthetic=len(rows), n_envelope_violations=len(bad), max_excess_deg=max([r["excess_deg"] for r in rows]),
           violations=bad, closest_to_rear_right=rr)
print(json.dumps({k: v for k, v in out.items() if k != "violations"}, indent=1)); print("violations:", len(bad))
for r in bad[:15]: print(r)
if len(sys.argv) > 4: Path(sys.argv[4]).write_text(json.dumps(out, indent=1))
