"""One SD10-L1 fit: arm x seed, exactly 1,000 AdamW updates on the frozen schedule.

Mirrors lerobot_train.update_policy (530b253f) without accelerate: policy(batch) -> backward -> clip_grad_norm_(10)
-> optimizer.step -> zero_grad; float32, no autocast. The registered LR for update i (learning_rates.npy[i]) is set
on the optimizer before update i. Writes a new namespace outputs/train/<arm>_s<seed>/.

    python train_l1.py --arm SD10 --seed 0
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
import warnings

import numpy as np

import l1lib as L

PT = L.PT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(L.ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=L.SEEDS)
    ap.add_argument("--steps", type=int, default=PT.UPDATES, help="smoke tests only; registered runs use 1000")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = L.TRAIN / f"{a.arm}_s{a.seed}" if a.out is None else L.Path(a.out)
    assert not (out / "pretrained_model/model.safetensors").exists(), "refusing to overwrite a finished fit"
    out.mkdir(parents=True, exist_ok=True)
    L.check_seal()
    caught = []
    warnings.simplefilter("always")
    _show = warnings.showwarning
    warnings.showwarning = lambda m, c, f, l, *r: (caught.append(f"{c.__name__}: {m} ({f}:{l})"), _show(m, c, f, l, *r))
    det = L.determinism(a.seed)
    import torch
    from torch.utils.data import DataLoader
    assert torch.cuda.is_available()
    policy, pre, post, cfg = L.make_policy_and_processors(a.arm, "cuda")
    norm_hashes = L.check_normalizers(pre, post, a.arm)
    assert set(cfg.input_features) == {"observation.state", L.CAM_KEY} and cfg.output_features["action"].shape == (16,)
    assert cfg.n_obs_steps == 1 and cfg.chunk_size == 50 and cfg.n_action_steps == 50 and cfg.empty_cameras == 0
    assert cfg.freeze_vision_encoder and cfg.train_expert_only and cfg.train_state_proj and not cfg.adapt_to_pi_aloha
    params = [p for p in policy.parameters() if p.requires_grad]
    tc = L.CFG["optimizer"]
    opt = torch.optim.AdamW(params, lr=tc["lr"], betas=tuple(tc["betas"]), eps=tc["eps"], weight_decay=tc["weight_decay"])
    lrs = np.load(L.L1 / "learning_rates.npy", allow_pickle=False)
    assert np.array_equal(lrs, PT.learning_rates()) and len(lrs) == PT.UPDATES
    # Cross-check against the pinned LeRobot scheduler (33 warmup / 1000 decay, peak 1e-4, floor 2.5e-6).
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    _o = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-4)
    _s = CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=33, num_decay_steps=1000, peak_lr=1e-4, decay_lr=2.5e-6).build(_o, 1000)
    ref = []
    for _ in range(PT.UPDATES):
        ref.append(_o.param_groups[0]["lr"]); _o.step(); _s.step()
    lr_vs_lerobot = float(np.max(np.abs(np.array(ref) - lrs)))
    assert lr_vs_lerobot < 1e-15, lr_vs_lerobot
    sampler = L.FrozenBatches(a.arm, a.seed)
    ds = L.SourceDataset(L.ARMS[a.arm])
    tr = L.CFG["train_config"]
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=tr["num_workers"], prefetch_factor=tr["prefetch_factor"],
                    pin_memory=True, persistent_workers=False)
    n_trainable = sum(p.numel() for p in params)
    dtypes = {}
    for n, p in policy.named_parameters():
        dtypes.setdefault(str(p.dtype), [0, 0])[0 if p.requires_grad else 1] += p.numel()
    pre_info = {"arm": a.arm, "seed": a.seed, "steps": a.steps, "determinism": det, "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "lerobot_file": __import__("lerobot").__file__, "trainable_params": n_trainable,
                "param_numel_by_dtype_[trainable,frozen]": dtypes, "normalizer_tensor_sha256": norm_hashes,
                "schedule_sha256": sampler.digest, "lr_max_abs_diff_vs_pinned_lerobot_scheduler": lr_vs_lerobot,
                "exec_code_sha256": L.exec_code_hashes(), "seal_sha256": L.SEAL_SHA}
    (out / "prelaunch.json").write_text(json.dumps(pre_info, indent=1))
    log, t0 = [], time.time()
    seen_ep, seen_fr = np.zeros_like(sampler.ep), np.zeros_like(sampler.fr)
    policy.train()
    step = 0
    for step, batch in enumerate(dl):
        if step >= a.steps:
            break
        seen_ep[step], seen_fr[step] = batch.pop("episode").numpy(), batch.pop("frame").numpy()
        assert np.array_equal(seen_ep[step], sampler.ep[step]) and np.array_equal(seen_fr[step], sampler.fr[step])
        assert not batch["action_is_pad"].any() and batch["action"].shape == (16, 50, 16)
        b = pre(L.to_policy_batch(batch))
        assert b[L.CAM_KEY].dtype == torch.float32 and "observation.images.body_cam" not in b
        for g in opt.param_groups:
            g["lr"] = float(lrs[step])
        loss, out_d = policy(b)
        assert torch.isfinite(loss), f"nonfinite loss at update {step}"
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), tc["grad_clip_norm"])
        opt.step()
        opt.zero_grad()
        log.append({"update": step, "lr": float(lrs[step]), "loss": float(loss.item()), "grad_norm": float(gn.item())})
        if step % 50 == 0 or step == a.steps - 1:
            print(f"update {step} lr {lrs[step]:.3e} loss {loss.item():.4f} gn {gn.item():.3f} {time.time() - t0:.0f}s", flush=True)
    consumed = len(log)
    assert consumed == a.steps
    adam_steps = sorted({int(s["step"]) for s in opt.state_dict()["state"].values()})
    # save terminal state
    pm = out / "pretrained_model"
    policy.save_pretrained(pm)
    pre.save_pretrained(pm)
    post.save_pretrained(pm)
    ts = out / "training_state"
    ts.mkdir(exist_ok=True)
    torch.save(opt.state_dict(), ts / "optimizer_state.pt")
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
           "torch_cuda": torch.cuda.get_rng_state_all()}
    with open(ts / "rng_state.pkl", "wb") as f:
        pickle.dump(rng, f)
    sched = {"consumed_optimizer_updates": consumed, "next_lr_index": consumed, "last_lr": float(lrs[consumed - 1]),
             "lr_source": "learning_rates.npy (set before each update)", "schedule_sha256": sampler.digest,
             "consumed_schedule_sha256": PT.array_sha(np.stack([seen_ep[:consumed], seen_fr[:consumed]])),
             "consumed_equals_frozen": bool(consumed == PT.UPDATES and PT.array_sha(np.stack([seen_ep, seen_fr])) == sampler.digest),
             "dataloader_fetched_batches": len(sampler), "discarded_prefetched_batches": len(sampler) - consumed,
             "adam_step_counters": adam_steps}
    (ts / "schedule_state.json").write_text(json.dumps(sched, indent=1))
    srcs = {str(e): int((seen_ep[:consumed] == e).sum()) for e in L.ARMS[a.arm]}
    term = {"wall_s": time.time() - t0, "consumed_optimizer_updates": consumed, "adam_step_counters": adam_steps,
            "source_draws_consumed": srcs, "valid_target_timesteps": consumed * 16 * 50,
            "scalar_loss_entries": consumed * 16 * 50 * 16, "final_loss": log[-1]["loss"],
            "nonfinite_loss": False, "max_cuda_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
            "warnings": sorted(set(caught))[:50],
            "model_sha256": L.sha(pm / "model.safetensors"),
            "processor_sha256": {p.name: L.sha(p) for p in sorted(pm.glob("policy_*"))},
            "optimizer_state_sha256": L.sha(ts / "optimizer_state.pt"), "schedule_state": sched}
    (out / "train_log.json").write_text(json.dumps(log))
    (out / "terminal.json").write_text(json.dumps(term, indent=1))
    print(json.dumps({k: term[k] for k in ("wall_s", "consumed_optimizer_updates", "adam_step_counters", "final_loss")}))


if __name__ == "__main__":
    main()
