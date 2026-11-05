"""One-time source preprocessing for all 10 episodes on both GPUs, benchmarked (lid track, SAM2.1, reconstruction,
background plate). Writes outputs/bench/preprocess.json and a 1 Hz nvidia-smi log.

    cd research/sim_free_augmentation/v3
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/preprocess_all.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

CHILD = r"""
import sys, json, time, torch
sys.path.insert(0, {here!r})
from sfa3 import recon3
t0 = time.time(); lt = recon3.ensure_lid_track({e}); t1 = time.time()
r = recon3.build({e}, device={dev!r}, with_plate=True, log=lambda *a: None)
out = {{"episode": {e}, "device": {dev!r}, "lid_track_s": t1 - t0, "lid_cached": lt["cached"], "total_s": time.time() - t0,
       "recon_runtime_s": r.info["runtime_s"], "sam2_cached": r.info["sam2_cached"],
       "peak_vram_gb": torch.cuda.max_memory_allocated({dev!r}) / 1e9 if torch.cuda.is_initialized() else 0.0,
       "lag": r.info["lag_frames"], "lead": r.info["action_lead"]["k_best"], "events": r.info["events"],
       "lid_ok_frac": r.info["lid_ok_frac"], "joint_closure_mm": r.info["joint_closure"]["translation_mm_median"],
       "independent_closure_mm": r.info["independent_closure"]["translation_mm_median"],
       "rms_px": [r.info["joint_fit"]["rms_px_R"], r.info["joint_fit"]["rms_px_L"]],
       "axis_iou": r.info["joint_axis_iou"], "plate_never_free": r.info["plate_never_free_frac"]}}
print("PRE", json.dumps(out))
"""


def run(e, dev):
    t = time.time()
    p = subprocess.run([sys.executable, "-c", CHILD.format(here=str(HERE), e=e, dev=dev)], capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("PRE ")]
    if not line:
        return {"episode": e, "error": p.stderr[-3000:], "wall_s": time.time() - t}
    r = json.loads(line[0][4:])
    r["wall_s"] = time.time() - t
    print(f"ep{e} on {dev}: {r['wall_s']:.1f} s (sam2 {r['recon_runtime_s']['sam2']:.1f} s, cached={r['sam2_cached']}) "
          f"closure {r['independent_closure_mm']:.0f}->{r['joint_closure_mm']:.0f} mm lag {r['lag']}", flush=True)
    return r


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "uncached"
    out = HERE / "outputs/bench"
    out.mkdir(parents=True, exist_ok=True)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                            "-l", "1"], stdout=open(out / f"nvidia_smi_preprocess_{tag}.csv", "w"))
    t0 = time.time()
    lanes = {"cuda:0": [0, 2, 4, 6, 8], "cuda:1": [1, 3, 5, 7, 9]}
    with ThreadPoolExecutor(2) as ex:
        futs = {d: ex.submit(lambda d=d: [run(e, d) for e in lanes[d]]) for d in lanes}
        res = [r for d in lanes for r in futs[d].result()]
    wall = time.time() - t0
    smi.terminate()
    (out / f"preprocess_{tag}.json").write_text(json.dumps({"wall_s": wall, "episodes": res, "lanes": lanes}, indent=1))
    print(f"all 10 episodes preprocessed in {wall:.1f} s wall-clock")


if __name__ == "__main__":
    main()
