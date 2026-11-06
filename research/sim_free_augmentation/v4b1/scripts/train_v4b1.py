"""One v4B.1 SmolVLA run: the frozen v4B trainer path (official lerobot_train.train with the v4B TwoPart view and
StratifiedBatches of the BASE arm) plus the arm's intervention (sfa4b1/interventions.py).

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
        ../../../.venv/bin/python scripts/train_v4b1.py --arm RpPos_PhaseMask --seed 0 [--steps 1000] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa4b1 import interventions as iv  # noqa: E402

mx = iv.mx
V31 = HERE.parent / "v3_1"
TR = mx.CFG["training"]
REPO = HERE.parents[2]


def lerobot_rev():
    """installed LeRobot source revision from the wheel's PEP 610 direct_url.json (not a git query of the venv)."""
    import glob
    import lerobot
    site = Path(lerobot.__file__).resolve().parents[1]
    f = sorted(glob.glob(str(site / "lerobot-*.dist-info/direct_url.json")))
    return json.loads(Path(f[0]).read_text()) if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(iv.CFG["arms"]))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=iv.CFG["steps"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else HERE / "outputs/train" / f"{a.arm}_s{a.seed}"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = iv.CFG["arms"][a.arm]["base"]
    import lerobot
    import torch
    from lerobot.datasets import factory
    from lerobot.scripts import lerobot_train as T

    # pre-launch provenance (written BEFORE training starts)
    git = lambda *c: subprocess.run(["git", *c], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    pre = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "base_arm": base, "spec": iv.CFG["arms"][a.arm],
           "p_drop": iv.CFG["p_drop"], "git_head": git("rev-parse", "HEAD"),
           "v4b1_dirty": git("status", "--porcelain", "--", "research/sim_free_augmentation/v4b1").splitlines(),
           "lerobot_version": getattr(lerobot, "__version__", None), "lerobot_path": str(Path(lerobot.__file__).parent),
           "lerobot_installed_from": lerobot_rev()}
    sel = mx.selection()
    info = {"prelaunch": pre}

    def make_ds(cfg):
        ds_r = factory.make_dataset(cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2.dataset.repo_id = "local/" + TR["synthetic_dataset"]
        cfg2.dataset.root = str(V31 / "outputs/lerobot" / TR["synthetic_dataset"])
        ds_s = factory.make_dataset(cfg2)
        ds = iv.TwoPartV41(ds_r, ds_s, a.arm, a.seed)
        info["ranges"] = (mx.episode_ranges(ds_r.meta), mx.episode_ranges(ds_s.meta))
        return ds, None

    def make_dl(cfg, dataset, eval_dataset, step, parallel_dims):
        rr, sr = info["ranges"]
        real, syn = mx.frame_pools(base, sel, rr, sr, dataset.nR)
        sampler = mx.StratifiedBatches(base, sel, real, syn, cfg.steps, cfg.batch_size, cfg.seed, TR["sampler_seed"], TR["real_fraction"])
        info["sampler"] = {"base_arm": base, "n_real_per_batch": sampler.n_real, "real_source_weights": [float(x) for x in sampler.w],
                           "schedule_sha256": mx.schedule_digest(sampler), "selection_keys": sel["keys"].get(base, {})}
        (out.parent / f"{out.name}_run.json").write_text(json.dumps(info | {"ranges": None}, indent=1, default=str))
        dl = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=cfg.num_workers,
                                         pin_memory=parallel_dims.device_type == "cuda", collate_fn=None,
                                         prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None, persistent_workers=False)
        return dl, None

    applier = iv.Applier(a.arm, T._preprocess_dataset_batch, out.parent / f"{out.name}_budget.json")
    T._preprocess_dataset_batch = applier
    T.make_train_eval_datasets = make_ds
    T.make_dataloaders = make_dl
    sys.argv = [sys.argv[0], f"--policy.path={TR['pretrained']}", f"--dataset.repo_id=local/{TR['real_dataset']}",
                f"--dataset.root={V31 / 'outputs/lerobot' / TR['real_dataset']}", f"--steps={a.steps}",
                f"--batch_size={TR['batch_size']}", f"--seed={a.seed}", f"--num_workers={TR['num_workers']}", "--log_freq=50",
                "--save_checkpoint=true", f"--save_freq={a.steps}", f"--output_dir={out}", "--policy.push_to_hub=false",
                "--wandb.enable=false", f"--rename_map={json.dumps(TR['camera_rename'])}"]
    T.train()
    applier.dump()


if __name__ == "__main__":
    main()
