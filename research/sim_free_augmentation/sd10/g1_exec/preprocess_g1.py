"""Cold, pool-keyed preprocessing of the 23-source union in capture-ID order (one process per source, 600 s cap).

Lid track (frozen Gate-1 HSV tracker from the verified pool click) -> SAM2.1-small masks -> frozen v2 joint
attachment fit with the registered fixed lag -1.625 and fixed inherited camera -> standing-can pose -> background plate.
Writes STORE/assets/pool_epXX_geometry.pkl (+ plate npz) and outputs/preprocess.json. Shared captures are built once,
so every condition uses identical asset hashes; each condition is charged its own ten sources' standalone time.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import g1lib as G

CHILD = r"""
import json, sys, time, numpy as np, torch
sys.path.insert(0, {exec_dir!r})
import g1lib as G
t0 = time.time()
rec = G.build_source({e}, device="cuda:0", with_plate=True)
t1 = time.time()
G.save_light(rec, G.STORE / "assets" / "pool_ep{e:02d}_geometry.pkl")
np.savez_compressed(G.STORE / "assets" / "pool_ep{e:02d}_plate.npz", plate=rec.plate)
info = {{k: rec.info[k] for k in ("events", "action_lead", "lag_frames", "lag_rms_by_lag", "lid_ok_frac", "joint_fit",
       "joint_axis_iou", "joint_closure", "independent_closure", "table_can_axis_tilt_from_vertical_deg", "z_table_m",
       "plate_never_free_frac", "sam2_cache", "sam2_cached", "runtime_s", "receiver_from_closure_only", "init_click",
       "lid_track_sha256", "table_pose")}}
out = {{"episode": {e}, "build_s": t1 - t0, "asset_sha256": G.asset_hash(rec), "lead": rec.lead, "lag": rec.lag,
       "T": rec.T, "info": info, "gpu": torch.cuda.get_device_name(0), "sam_dtype": "bfloat16",
       "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}}
print("PRE" + json.dumps(G.jsonable(out)))
"""


def main():
    G.check_seals()
    G.STORE.mkdir(parents=True, exist_ok=True)
    assert not (G.STORE / "cache").exists() or not any((G.STORE / "cache").iterdir()), "cold cache required"
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None); env.pop("LD_LIBRARY_PATH", None)
    res, t_all = [], time.time()
    for e in G.UNION:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-B", "-c", CHILD.format(exec_dir=str(G.EXEC), e=e)], capture_output=True,
                               text=True, env=env, timeout=G.CFG["compute"]["source_preprocessing_cap_s"])
            line = [l for l in p.stdout.splitlines() if l.startswith("PRE")]
            r = json.loads(line[0][3:]) if line else {"episode": e, "error": p.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            r = {"episode": e, "error": "TIMEOUT_600s"}
        r["wall_s"] = time.time() - t0
        res.append(r)
        print(e, round(r["wall_s"], 1), "ERROR" if "error" in r else f"closure {r['info']['joint_closure']['translation_mm_median']:.1f} mm "
              f"rms R {r['info']['joint_fit']['rms_px_R']:.2f} fallback {r['info']['receiver_from_closure_only']}", flush=True)
    per = {r["episode"]: r for r in res}
    charge = {c: {"standalone_preprocess_s": sum(per[e]["wall_s"] for e in ids),
                  "failed_sources": [e for e in ids if "error" in per[e]]} for c, ids in G.CFG["conditions"].items()}
    out = {"sources": res, "union_wall_s": time.time() - t_all, "per_condition_charge": charge,
           "cold_cache": True, "lag_fixed": G.FIXED_LAG, "camera_refit": False}
    G.OUT.mkdir(parents=True, exist_ok=True)
    (G.OUT / "preprocess.json").write_text(json.dumps(G.jsonable(out), indent=1))
    print(json.dumps({"union_wall_s": out["union_wall_s"], "charge": charge}, indent=1))


if __name__ == "__main__":
    main()
