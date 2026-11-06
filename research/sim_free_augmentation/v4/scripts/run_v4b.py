"""Run every frozen v4B training job (5 arms x 3 paired seeds x 1000 updates), two concurrent jobs per GPU.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/run_v4b.py [--lanes 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
TR = json.loads((HERE / "configs/v4b.json").read_text())["training"]
OUT = HERE / "outputs/v4b"


def gpu_sampler(stop, samples):
    while not stop.is_set():
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout
            samples.append((time.time(), [tuple(map(float, l.split(","))) for l in q.strip().splitlines()]))
        except Exception:
            pass
        stop.wait(5)


def run(arm, seed, gpu):
    out = OUT / "train" / f"{arm}_s{seed}"
    if (out / "checkpoints/last/pretrained_model/model.safetensors").exists():
        return {"arm": arm, "seed": seed, "skipped": "exists"}
    log = OUT / "logs" / f"train_{arm}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", HF_DATASETS_CACHE=str(OUT / "cache/hf_datasets"))
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    t0 = time.time()
    rc = subprocess.run([str(REPO / ".venv/bin/python"), str(HERE / "scripts/train_v4b.py"), "--arm", arm, "--seed", str(seed)],
                        stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    txt = log.read_text(errors="ignore")
    losses = [float(x) for x in re.findall(r"step:\S+ .*?loss:([0-9.]+)", txt)]
    mem = [float(x) for x in re.findall(r"mem_gb:([0-9.]+)", txt)]
    return {"arm": arm, "seed": seed, "gpu": gpu, "rc": rc, "wall_s": time.time() - t0, "loss_log": losses,
            "peak_gpu_mem_gb_logged": max(mem) if mem else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=4)
    a = ap.parse_args()
    jobs = [(arm, s) for s in TR["seeds"] for arm in TR["arms"]]
    stop, samples = threading.Event(), []
    th = threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True)
    th.start()
    t0 = time.time()
    res = []
    with ThreadPoolExecutor(a.lanes) as ex:
        for r in ex.map(lambda x: run(*x[1], x[0] % 2), enumerate(jobs)):
            res.append(r)
            print({k: r.get(k) for k in ("arm", "seed", "rc", "wall_s", "skipped")} | {"final_loss": (r.get("loss_log") or [None])[-1]}, flush=True)
    stop.set()
    snap = Path.home() / ".cache/huggingface/hub/models--lerobot--smolvla_base/snapshots"
    pre = {str(p.relative_to(snap)): hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.glob("*/model.safetensors")}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "research/sim_free_augmentation/v4"], cwd=REPO, capture_output=True, text=True).stdout
    gpu = {}
    for _, rows in samples:
        for i, m, u in rows:
            g = gpu.setdefault(int(i), {"max_mem_mib": 0, "util": []})
            g["max_mem_mib"] = max(g["max_mem_mib"], m)
            g["util"].append(u)
    gpu = {i: {"max_mem_mib": g["max_mem_mib"], "mean_util_pct": sum(g["util"]) / len(g["util"])} for i, g in gpu.items()}
    (OUT / "train_runs.json").write_text(json.dumps({"runs": res, "wall_s": time.time() - t0, "lanes": a.lanes, "git_head": git,
                                                     "v4_dirty_files_at_launch": dirty.splitlines(), "pretrained_sha256": pre,
                                                     "gpu_usage": gpu, "config": TR}, indent=1))


if __name__ == "__main__":
    main()
