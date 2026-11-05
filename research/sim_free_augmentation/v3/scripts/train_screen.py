"""Bounded SmolVLA training screen (configs/train_screen.json): 3 arms x 3 seeds x 1000 updates, identical
preprocessing, both GPUs. Uses the official `lerobot-train` entry point on LOCAL datasets; nothing is pushed.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/train_screen.py
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
CFG = json.loads((HERE / "configs/train_screen.json").read_text())


def run(arm, seed, gpu):
    ds = CFG["arms"][arm]
    out = HERE / "outputs/train" / f"{arm.replace('+', 'p')}_s{seed}"
    if (out / "checkpoints/last/pretrained_model/model.safetensors").exists():
        return {"arm": arm, "seed": seed, "skipped": "exists"}
    if out.exists():
        import shutil
        shutil.rmtree(out)
    cmd = [str(REPO / ".venv/bin/lerobot-train"), "--policy.path=lerobot/smolvla_base", f"--dataset.repo_id=local/{ds}",
           f"--dataset.root={HERE / 'outputs/lerobot' / ds}", f"--steps={CFG['steps']}", f"--batch_size={CFG['batch_size']}",
           f"--seed={seed}", f"--num_workers={CFG['num_workers']}", "--log_freq=100", "--save_checkpoint=true",
           f"--save_freq={CFG['steps']}", f"--output_dir={out}", "--policy.push_to_hub=false", "--wandb.enable=false",
           f"--rename_map={json.dumps(CFG['camera_rename'])}"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", HF_DATASETS_CACHE=str(HERE / "outputs/cache/hf_datasets"))
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    t0 = time.time()
    log = HERE / "outputs/logs" / f"train_{arm.replace('+', 'p')}_s{seed}.log"
    rc = subprocess.run(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env).returncode
    losses = [float(l.split("loss:")[1].split()[0]) for l in open(log) if " loss:" in l and "step:" in l]
    return {"arm": arm, "seed": seed, "gpu": gpu, "rc": rc, "wall_s": time.time() - t0, "loss_log": losses}


def main():
    jobs = [(a, s) for s in CFG["seeds"] for a in CFG["arms"]]
    res = []
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(run, a, s, i % 2) for i, (a, s) in enumerate(jobs)]
        for f in futs:
            r = f.result()
            res.append(r)
            print(r if "skipped" in r else {k: r[k] for k in ("arm", "seed", "rc", "wall_s")} | {"final_loss": r["loss_log"][-1:] }, flush=True)
    (HERE / "outputs/train_screen_runs.json").write_text(json.dumps({"runs": res, "wall_s": time.time() - t0, "config": CFG}, indent=1))


if __name__ == "__main__":
    main()
