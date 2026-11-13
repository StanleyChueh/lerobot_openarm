"""Independent export audit (Claude): counts, no real episodes in synthetic-only, fixed duration, URDF∪source envelope.

Reads parquet + provenance directly and parses URDF limits from XML (does not reuse sfagen's checker).
usage: python check_exports.py <run_dir> <config.yaml> [out.json]
"""
import json, sys, glob, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, pandas as pd, yaml

run, cfg_path = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load(cfg_path.read_text())
urdf = ET.parse(cfg["robot"]["urdf"] if "robot" in cfg else cfg["source"]["urdf"])
lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
       for j in urdf.getroot().iter("joint") if j.find("limit") is not None and j.get("type") == "revolute"}
arms = (cfg.get("robot") or {}).get("arms") or cfg["arms"]
cols = {i: n for s in arms.values() for n, i in zip(s["joints"], s["state_indices"])}

def load(root):
    df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root / "data/*/*.parquet")))])
    return df.sort_values("index"), json.loads((root / "meta/sfa_provenance.json").read_text())

def arr(df, k): return np.stack(df[k].to_numpy()).astype(float)

out = {}
for cond in ["copied", "right_wrist"]:
    syn, prov = load(run / cond / "dataset"); mix, mprov = load(run / cond / "dataset_mixed")
    nreal = len(mprov["real_episodes"])
    real = mix[mix.episode_index < nreal]
    R = {e: (arr(g, "observation.state"), arr(g, "action")) for e, g in real.groupby("episode_index")}
    src_of = {e["export_episode_index"]: e["source_real_episode"] for e in prov["episodes"]}
    res = dict(syn_episodes=int(syn.episode_index.nunique()), syn_frames=len(syn),
               mix_episodes=int(mix.episode_index.nunique()), mix_frames=len(mix),
               mix_is_synthetic_real=float(mix[mix.episode_index < nreal]["is_synthetic"].map(lambda v: np.ravel(v)[0]).max()),
               mix_is_synthetic_syn=float(mix[mix.episode_index >= nreal]["is_synthetic"].map(lambda v: np.ravel(v)[0]).min()),
               duration_mismatch=[], envelope_violations=[], identical_to_real=[])
    for e, g in syn.groupby("episode_index"):
        s, a = arr(g, "observation.state"), arr(g, "action"); S0, A0 = R[src_of[e]]
        if len(s) != len(S0): res["duration_mismatch"].append(int(e))
        if s.shape == S0.shape and np.array_equal(s, S0) and np.array_equal(a, A0): res["identical_to_real"].append(int(e))
        for X, Y, f in ((s, S0, "state"), (a, A0, "action")):
            for i, n in cols.items():
                lo, hi = min(lim[n][0], Y[:, i].min()), max(lim[n][1], Y[:, i].max())
                ex = max(lo - X[:, i].min(), X[:, i].max() - hi)
                if ex > 1e-6: res["envelope_violations"].append(dict(ep=int(e), field=f, joint=n, excess_deg=float(np.degrees(ex))))
    # mixed synthetic part must equal synthetic-only arrays
    ms = mix[mix.episode_index >= nreal]
    res["mixed_syn_equals_syn"] = bool(np.array_equal(arr(ms, "action"), arr(syn, "action")) and np.array_equal(arr(ms, "observation.state"), arr(syn, "observation.state")))
    out[cond] = res
print(json.dumps(out, indent=1))
if len(sys.argv) > 3: Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
