"""C4: per-N few-source candidate datasets from the 10-demo capped F1W work dir (configs/openarm_pringles_fewsrc.yaml).

select     For N in the frozen subsets (results/fewsrc/selection.json): every candidate of a chosen demo that passes all
           checks except speed/accel/table, re-checked against the speed/accel/table envelope of those N demos only;
           then at most PER_DEMO variants per demo, spread by k-center over the target positions (nearest-to-centroid
           first). Writes results/fewsrc/datasets.json and prints the per-demo render key lists (union over N).
setup N [--wrist]  Work dir configs/work/pringles_fewsrc{N} (symlinks to candidates, body/wrist renders and recon cache of the
           base work dir) + configs/pringles_fewsrc{N}.yaml (episodes = the N demos), so `sfagen build` exports it.
usage: fewsrc_build.py select | setup N"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np
import yaml
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tools"))
from select_sources import kcenter
BASE = HERE / "configs/openarm_pringles_fewsrc.yaml"; WBASE = HERE / "configs/openarm_pringles_fewsrc_wristrender.yaml"; OUT = HERE / "results/fewsrc/datasets.json"; PER_DEMO = None   # None: every variant passing the N-demo envelope (16 was a compute cap; it cut N=2 coverage 24 -> 19)


def select():
    from sfagen.context import Context
    from sfagen.checks import Checker
    from sfagen.why import reasons
    ctx = Context(BASE); ck = Checker(ctx); cs = ck.cfg; W = ctx.work; subsets = json.loads((HERE / "results/fewsrc/selection.json").read_text())["subsets"]
    src = {}
    for e in ctx.episodes():
        S, A = ctx.src.arrays(e); v, a = ck.kin(A); src[e] = {"v": v, "a": a, "z": float(ck.lowest(S).min())}
    cands = {}
    for e in ctx.episodes():
        for key, c in json.loads((W / f"candidates/ep{e}/checks.json").read_text()).items():
            npz = W / f"candidates/ep{e}/{key}.npz"
            if "error" in c or not npz.exists() or (set(reasons(c, ctx.cfg.get("checks", {}), c.get("real_lowest_arm_z_m", 0))) - {"speed", "accel", "table"}):
                continue
            v, a = ck.kin(np.load(npz)["action"].astype(float)); cands.setdefault(e, []).append((key, v, a, c["lowest_arm_z_m"], np.asarray(c["shift_m"])[:2]))
    out = {}
    for n, sub in subsets.items():
        ev = np.max([src[e]["v"] for e in sub], 0); ea = np.max([src[e]["a"] for e in sub], 0); mz = min(src[e]["z"] for e in sub); eps = []
        for e in sub:
            ok = [(k, float((v / ev).max()), float((a / ea).max()), d) for k, v, a, z, d in cands.get(e, [])
                  if (v / ev).max() <= cs["speed_ratio"] + 1e-9 and (a / ea).max() <= cs["accel_ratio"] + 1e-9 and z >= mz - cs["table_tol_m"]]
            keep = ok if PER_DEMO is None else ([ok[i] for i in sorted(kcenter([d for *_, d in ok], PER_DEMO))] if ok else [])
            eps += [[e, k, {"speed_ratio_to_subset_max": sr, "accel_ratio_to_subset_max": ar, "subset_lowest_arm_z_m": mz, "envelope_sources": sub}] for k, sr, ar, _ in keep]
        out[n] = {"sources": sub, "per_demo_cap": PER_DEMO or "none", "passing_before_cap": {str(e): sum(1 for k, v, a, z, d in cands.get(e, []) if (v / ev).max() <= cs["speed_ratio"] + 1e-9 and (a / ea).max() <= cs["accel_ratio"] + 1e-9 and z >= mz - cs["table_tol_m"]) for e in sub},
                  "episodes": eps}
        print("N", n, "sources", sub, "episodes", len(eps), out[n]["passing_before_cap"], flush=True)
    OUT.write_text(json.dumps(out, indent=1))
    keys = {}
    for d in out.values():
        for e, k, _ in d["episodes"]:
            keys.setdefault(e, set()).add(k)
    (HERE / "results/fewsrc/render_keys.json").write_text(json.dumps({str(e): sorted(v) for e, v in sorted(keys.items())}, indent=1))
    print("render union", sum(len(v) for v in keys.values()), {e: len(v) for e, v in sorted(keys.items())})


def setup(n, wrist=False):
    """wrist=True: the rendered-wrist ablation variant (base config with wrist_render), own work dir, shared renders."""
    sel = json.loads(OUT.read_text())[str(n)]; base = yaml.safe_load((WBASE if wrist else BASE).read_text()); D = Path(base["work_dir"])
    name = f"pringles_fewsrc{n}" + ("_wristrender" if wrist else ""); W = D.parent / name
    if W.exists():
        shutil.rmtree(W)
    W.mkdir(parents=True)
    for f in ("calibration.json", "table_height.json"):
        shutil.copy(D / f, W / f)
    for d in ("cache", "body", "wrist"):
        (D / d).mkdir(exist_ok=True); os.symlink(os.path.realpath(D / d), W / d)
    for e in sel["sources"]:
        (W / f"candidates/ep{e}").mkdir(parents=True); full = json.loads((D / f"candidates/ep{e}/checks.json").read_text()); out = {}
        for ee, key, env in sel["episodes"]:
            if ee == e:
                c = dict(full[key]); c.update({"speed_ratio_to_real_max": env["speed_ratio_to_subset_max"], "accel_ratio_to_real_max": env["accel_ratio_to_subset_max"],
                                               "real_lowest_arm_z_m": env["subset_lowest_arm_z_m"], "envelope_sources": env["envelope_sources"], "pass": True})
                out[key] = c; os.symlink(D / f"candidates/ep{e}/{key}.npz", W / f"candidates/ep{e}/{key}.npz")
        (W / f"candidates/ep{e}/checks.json").write_text(json.dumps(out, indent=1, default=float))
    c = dict(base); c["source"] = {**base["source"], "episodes": sel["sources"]}; c["work_dir"] = str(W)
    (HERE / f"configs/{name}.yaml").write_text(("# ABLATION: wrist views re-rendered although the registered wrist gate failed (results/wrist/RESULTS.md).\n" if wrist else "") + f"# Few-source candidate dataset: {n} real demo(s) {sel['sources']} (frozen k-center rule), F1W capped at 11 cm, "
                                                         f"<= {sel['per_demo_cap']} variants per demo, replay limits from these {n} demos only.\n" + yaml.safe_dump(c, sort_keys=False, width=160))
    print("ready", n, len(sel["episodes"]), "episodes from", sel["sources"], "->", W)


if __name__ == "__main__":
    select() if sys.argv[1] == "select" else setup(int(sys.argv[2]), "--wrist" in sys.argv)
