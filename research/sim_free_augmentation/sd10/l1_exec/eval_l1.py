"""SD10-L1 open-loop predictions on the 834 locked common inputs for each terminal (update-1000) checkpoint.

Original body RGB only (verified against locked hashes), recorded state[t], stored per-input noise, fixed task,
batch 32, ten flow steps, bf16 autocast, each checkpoint's own saved processors; float32 unnormalized (834,50,16).

    python eval_l1.py [--runs SD10_s0 ...] [--train_dir D] [--pred_dir P]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import l1lib as L

PT = L.PT


def load_ckpt(pm, device):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(str(pm), local_files_only=True).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(pm),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def predict(policy, pre, post, z, frames, device, batch=32):
    import torch
    items = z["items"]
    P = np.zeros((len(items), 50, 16), np.float32)
    for b0 in range(0, len(items), batch):
        sl = slice(b0, min(b0 + batch, len(items)))
        b = {"observation.state": torch.from_numpy(z["state"][sl].astype(np.float32)),
             L.CAM_KEY: torch.from_numpy(frames[sl]).permute(0, 3, 1, 2).to(torch.float32) / 255.0,
             "task": [PT.TASK] * (sl.stop - sl.start)}
        noise = torch.from_numpy(z["noise"][sl]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            bb = pre(b)
            policy.reset()
            act = policy.predict_action_chunk(bb, noise=noise)
        with torch.inference_mode():
            act = post(act.float())
        assert act.shape == (sl.stop - sl.start, 50, 16)
        P[sl] = act.float().cpu().numpy()
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[f"{a}_s{s}" for a in L.ARMS for s in L.SEEDS])
    ap.add_argument("--train_dir", default=str(L.TRAIN))
    ap.add_argument("--pred_dir", default=str(L.PRED))
    a = ap.parse_args()
    L.check_seal()
    L.determinism(0)
    pred = Path(a.pred_dir)
    pred.mkdir(parents=True, exist_ok=True)
    z = np.load(L.L1 / "evaluation_arrays.npz", allow_pickle=False)
    man = json.loads((L.L1 / "locked_evaluation_manifest.json").read_text())
    frames = np.load(L.CACHE / "eval_frames.npy", mmap_mode="r")
    assert frames.shape == (834, 480, 640, 3)
    for i in range(len(frames)):          # original body RGB only, exactly the locked frames
        assert PT.array_sha(frames[i]) == man["entries"][i]["native_video"]["rgb_sha256"], i
    frames = np.array(frames)              # load into RAM (writable, contiguous)
    for run in a.runs:
        arm, seed = run.rsplit("_s", 1)
        rd = Path(a.train_dir) / run
        pm = rd / "pretrained_model"
        term = json.loads((rd / "terminal.json").read_text())
        assert term["consumed_optimizer_updates"] == PT.UPDATES and term["adam_step_counters"] == [PT.UPDATES], run
        assert L.sha(pm / "model.safetensors") == term["model_sha256"], run
        t0 = time.time()
        policy, pre, post = load_ckpt(pm, "cuda")
        norm = L.check_normalizers(pre, post, arm)
        assert policy.config.num_steps == 10 and policy.config.chunk_size == 50
        P = predict(policy, pre, post, z, frames, "cuda")
        assert np.isfinite(P).all()
        np.savez_compressed(pred / f"{run}.npz", items=z["items"], prediction=P)
        meta = {"run": run, "arm": arm, "seed": int(seed), "update": PT.UPDATES, "model_sha256": term["model_sha256"],
                "processor_sha256": {p.name: L.sha(p) for p in sorted(pm.glob("policy_*"))},
                "normalizer_tensor_sha256": norm, "prediction_sha256": PT.array_sha(P), "items_sha256": PT.array_sha(z["items"]),
                "dtype": str(P.dtype), "shape": list(P.shape), "wall_s": time.time() - t0,
                "eval_arrays_sha256": L.sha(L.L1 / "evaluation_arrays.npz"), "inference": "bf16 autocast, batch 32, 10 flow steps, stored noise"}
        (pred / f"{run}.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: meta[k] for k in ("run", "prediction_sha256", "wall_s")}), flush=True)
        del policy
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
