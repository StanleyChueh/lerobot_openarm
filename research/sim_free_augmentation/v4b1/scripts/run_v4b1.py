"""Run the frozen v4B.1 training jobs (5 arms x 3 paired seeds x 1000 updates), two concurrent jobs per GPU.
Pre-launch provenance (git HEAD, dirty status, config/spec/input hashes, dependency revision) is written BEFORE any
worker starts; terminal state is written separately afterwards.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/run_v4b1.py [--lanes 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
CFG = json.loads((HERE / "configs/v4b1.json").read_text())
OUT = HERE / "outputs"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*c, cwd=REPO):
    return subprocess.run(["git", *c], cwd=cwd, capture_output=True, text=True).stdout.strip()


def gpu_sampler(stop, samples):
    while not stop.is_set():
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10).stdout
            samples.append([tuple(map(float, l.split(","))) for l in q.strip().splitlines()])
        except Exception:
            pass
        stop.wait(5)


def run(arm, seed, gpu):
    out = OUT / "train" / f"{arm}_s{seed}"
    if (out / "checkpoints/last/pretrained_model/model.safetensors").exists():
        return {"arm": arm, "seed": seed, "skipped": "exists"}
    log = OUT / "logs" / f"train_{arm}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    t0 = time.time()
    rc = subprocess.run([str(REPO / ".venv/bin/python"), str(HERE / "scripts/train_v4b1.py"), "--arm", arm, "--seed", str(seed)],
                        stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
    txt = log.read_text(errors="ignore")
    return {"arm": arm, "seed": seed, "gpu": gpu, "rc": rc, "wall_s": time.time() - t0,
            "loss_log": [float(x) for x in re.findall(r"step:\S+ .*?loss:([0-9.]+)", txt)],
            "warmup_autoscale_lines": [l.strip() for l in txt.splitlines() if "warmup" in l.lower() and "scal" in l.lower()][:2]}


def lerobot_rev():
    """installed LeRobot source revision from the wheel's PEP 610 direct_url.json (not a git query of the venv)."""
    import glob
    import lerobot
    site = Path(lerobot.__file__).resolve().parents[1]
    f = sorted(glob.glob(str(site / "lerobot-*.dist-info/direct_url.json")))
    return json.loads(Path(f[0]).read_text()) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=4)
    a = ap.parse_args()
    import lerobot
    OUT.mkdir(parents=True, exist_ok=True)
    pre = {"time": time.time(), "git_head": git("rev-parse", "HEAD"), "dirty_tracked": git("status", "--porcelain", "--untracked-files=no").splitlines(),
           "v4b1_dirty": git("status", "--porcelain", "--", "research/sim_free_augmentation/v4b1").splitlines(),
           "spec_sha256": sha(HERE / "V4B1_EXPERIMENT_SPEC.md"), "config_sha256": sha(HERE / "configs/v4b1.json"),
           "v4b_config_sha256": sha(HERE.parent / "v4/configs/v4b.json"), "frozen_inputs_sha256": sha(HERE.parent / "v4/configs/frozen_inputs_sha256.json"),
           "lerobot_version": getattr(lerobot, "__version__", None), "lerobot_installed_from": lerobot_rev(),
           "jobs": [(arm, s) for s in CFG["seeds"] for arm in CFG["arms"]]}
    (OUT / "prelaunch.json").write_text(json.dumps(pre, indent=1))
    stop, samples = threading.Event(), []
    threading.Thread(target=gpu_sampler, args=(stop, samples), daemon=True).start()
    t0 = time.time()
    res = []
    with ThreadPoolExecutor(a.lanes) as ex:
        for r in ex.map(lambda x: run(*x[1], x[0] % 2), enumerate(pre["jobs"])):
            res.append(r)
            print({k: r.get(k) for k in ("arm", "seed", "rc", "wall_s", "skipped")}, flush=True)
    stop.set()
    gpu = {}
    for rows in samples:
        for i, m, u in rows:
            g = gpu.setdefault(int(i), {"max_mem_mib": 0, "util": []})
            g["max_mem_mib"] = max(g["max_mem_mib"], m)
            g["util"].append(u)
    (OUT / "train_runs.json").write_text(json.dumps({"runs": res, "wall_s": time.time() - t0, "lanes": a.lanes,
                                                     "gpu_usage": {i: {"max_mem_mib": g["max_mem_mib"], "mean_util_pct": sum(g["util"]) / len(g["util"])}
                                                                   for i, g in gpu.items()},
                                                     "terminal_git_head": git("rev-parse", "HEAD")}, indent=1))


if __name__ == "__main__":
    main()
