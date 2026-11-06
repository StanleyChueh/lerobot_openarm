"""One v4B SmolVLA run through the OFFICIAL lerobot-train loop (lerobot.scripts.lerobot_train.train), with exactly two
substitutions: the dataset is the read-only real+synthetic TwoPart view (real-only normalisation statistics), and the
dataloader uses the common-random-number StratifiedBatches sampler of sfa4b/mixture.py.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
        ../../../.venv/bin/python scripts/train_v4b.py --arm RpPos --seed 0 [--steps 1000] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa4b import mixture as mx  # noqa: E402

V31 = HERE.parent / "v3_1"
TR = mx.CFG["training"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(TR["arms"]))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=TR["steps"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else HERE / "outputs/v4b/train" / f"{a.arm}_s{a.seed}"

    import torch
    from lerobot.datasets import factory
    from lerobot.scripts import lerobot_train as T
    from lerobot.utils.collate import lerobot_collate_fn

    sel = mx.selection()
    info = {}

    def make_ds(cfg):
        ds_r = factory.make_dataset(cfg)
        cfg2 = copy.deepcopy(cfg)
        cfg2.dataset.repo_id = "local/" + TR["synthetic_dataset"]
        cfg2.dataset.root = str(V31 / "outputs/lerobot" / TR["synthetic_dataset"])
        ds_s = factory.make_dataset(cfg2)
        ds = mx.TwoPart(ds_r, ds_s)
        info["parts"] = {"real": {"root": str(cfg.dataset.root), "frames": len(ds_r), "episodes": ds_r.num_episodes},
                         "synthetic": {"root": cfg2.dataset.root, "frames": len(ds_s), "episodes": ds_s.num_episodes}}
        info["ranges"] = (mx.episode_ranges(ds_r.meta), mx.episode_ranges(ds_s.meta))
        return ds, None

    def make_dl(cfg, dataset, eval_dataset, step, parallel_dims):
        rr, sr = info["ranges"]
        real, syn = mx.frame_pools(a.arm, sel, rr, sr, dataset.nR)
        sampler = mx.StratifiedBatches(a.arm, sel, real, syn, cfg.steps, cfg.batch_size, cfg.seed, TR["sampler_seed"], TR["real_fraction"])
        n_syn = sum(1 for b in [sampler.batch(s) for s in range(min(50, cfg.steps))] for i in b if i >= dataset.nR)
        info["sampler"] = {"arm": a.arm, "n_real_per_batch": sampler.n_real, "real_frames": int(sum(len(v) for v in real.values())), "real_source_weights": [float(x) for x in sampler.w],
                           "synthetic_frames_by_source": {int(E): int(len(v)) for E, v in syn.items()},
                           "synthetic_fraction_first50": n_syn / (min(50, cfg.steps) * cfg.batch_size),
                           "schedule_sha256": mx.schedule_digest(sampler), "selection_keys": sel["keys"].get(a.arm, {})}
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / f"{out.name}_mixture.json").write_text(json.dumps(info | {"ranges": None}, indent=1, default=str))
        dl = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=cfg.num_workers,
                                         pin_memory=parallel_dims.device_type == "cuda", collate_fn=lerobot_collate_fn
                                         if dataset.meta.has_language_columns else None,
                                         prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
                                         persistent_workers=False)
        return dl, None

    T.make_train_eval_datasets = make_ds
    T.make_dataloaders = make_dl
    sys.argv = [sys.argv[0], f"--policy.path={TR['pretrained']}", f"--dataset.repo_id=local/{TR['real_dataset']}",
                f"--dataset.root={V31 / 'outputs/lerobot' / TR['real_dataset']}", f"--steps={a.steps}",
                f"--batch_size={TR['batch_size']}", f"--seed={a.seed}", f"--num_workers={TR['num_workers']}", "--log_freq=50",
                "--save_checkpoint=true", f"--save_freq={a.steps}", f"--output_dir={out}", "--policy.push_to_hub=false",
                "--wandb.enable=false", f"--rename_map={json.dumps(TR['camera_rename'])}"]
    T.train()


if __name__ == "__main__":
    main()
