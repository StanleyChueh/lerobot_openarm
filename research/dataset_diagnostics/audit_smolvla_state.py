"""Runtime audit of how SmolVLA consumes the 16-D OpenArm observation.state (Phase A evidence).

    ~/Stanley_ws/lerobot_openarm/.venv-v2/bin/python research/dataset_diagnostics/audit_smolvla_state.py \
        --out research/dataset_diagnostics/smolvla_state_audit.json

Read-only: loads cached checkpoints and dataset frames, runs one inference pass and one forward/backward
pass on CPU (no optimizer step, nothing saved except the JSON report). Requires the update_dev_v2 fork env.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CHECKPOINTS = {
    "real_20": ("ethanCSL/openarm_pringles_lerobot_real_smolvla_20_v00", "cd6907a4d40c50d833228ea93821dc1e2b008ca2",
                "ethanCSL/openarm_pringles_lerobot_real_20_v00", "dad0defb3b6e206b3ce128ca07190befc3b6ad4f"),
    "real_10": ("ethanCSL/openarm_pringles_lerobot_real_smolvla_10_v00", "8593bf2539eb663a536aac77c2ab6f33219a54ba",
                "ethanCSL/openarm_pringles_lerobot_real_10_v00", "cebdee98115ab6827a4ec2655335c536b5dd085b"),
    "sim_10": ("ethanCSL/openarm_pringles_lerobot_sim_smolvla_10_v00", "9ce7ce4811de11adcc4f407eb5a93586d94a6aee",
               "ethanCSL/openarm_pringles_lerobot_sim_10_v00", "b6d39dae630b8ce0b5822bee7a40bffc68a28819"),
}
BASE = "lerobot/smolvla_base"


def static_checks(ckpt_dir: Path, ds: LeRobotDataset) -> dict:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    norm = load_file(str(ckpt_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    unnorm = load_file(str(ckpt_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    ds_stats = ds.meta.stats
    out = {
        "config_input_state_shape": cfg["input_features"]["observation.state"]["shape"],
        "config_output_action_shape": cfg["output_features"]["action"]["shape"],
        "max_state_dim": cfg["max_state_dim"], "max_action_dim": cfg["max_action_dim"],
        "dataset_state_shape": list(ds.meta.features["observation.state"]["shape"]),
        "dataset_state_names": ds.meta.features["observation.state"]["names"],
        "dataset_action_names": ds.meta.features["action"]["names"],
        "normalizer_state_mean_shape": list(norm["observation.state.mean"].shape),
        "normalizer_action_mean_shape": list(norm["action.mean"].shape),
        "normalizer_state_count": float(norm["observation.state.count"]),
        "dataset_total_frames": ds.meta.total_frames,
        "normalizer_episode_index_max": float(norm["episode_index.max"]),
        "dataset_total_episodes": ds.meta.total_episodes,
    }
    for key in ("observation.state", "action"):
        for stat in ("mean", "std"):
            a = norm[f"{key}.{stat}"].numpy()
            b = np.asarray(ds_stats[key][stat], dtype=np.float32)
            out[f"normalizer_vs_dataset_{key}_{stat}_maxabs"] = float(np.abs(a - b).max())
        out[f"pre_vs_post_{key}_mean_equal"] = bool(
            torch.equal(norm[f"{key}.mean"], unnorm[f"{key}.mean"])) if f"{key}.mean" in unnorm else None
    return out


def state_proj_weight_drift(ckpt_dir: Path) -> dict:
    """Per-input-column change of state_proj.weight vs smolvla_base: only trained columns should move."""
    key = "model.state_proj.weight"
    tuned = load_file(str(ckpt_dir / "model.safetensors"))[key].float()
    out = {"state_proj_weight_shape": list(tuned.shape), "max_abs_change_per_input_column": {}}
    # The fine-tune pulled smolvla_base's main at train time; every cached base revision is compared.
    base_root = Path(snapshot_download(BASE, local_files_only=True)).parent
    for snap in sorted(base_root.iterdir()):
        base = load_file(str(snap / "model.safetensors"))[key].float()
        diff = (tuned - base).abs().amax(dim=0)  # [max_state_dim]
        out["max_abs_change_per_input_column"][snap.name] = [round(float(x), 6) for x in diff]
    return out


def runtime_checks(ckpt_dir: Path, ds: LeRobotDataset, frame: int) -> dict:
    policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir))
    policy.config.device = "cpu"
    policy.to("cpu")
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    item = ds[frame]
    raw_state = item["observation.state"].clone()
    obs = {k: v for k, v in item.items() if k.startswith("observation.")}
    obs["task"] = item["task"]
    batch = pre(obs)

    captured = {}
    hook = policy.model.state_proj.register_forward_hook(
        lambda m, inp, out: captured.update(inp=inp[0].detach().clone(), out_shape=list(out.shape)))
    stats = pre.steps[-1]._tensor_stats["observation.state"]
    expected_norm = (raw_state - stats["mean"].cpu()) / (stats["std"].cpu() + 1e-8)

    with torch.no_grad():
        policy.reset()
        chunk = policy.predict_action_chunk(dict(batch))
    hook.remove()
    action = post(chunk[:, 0])
    proj_in = captured["inp"]
    res = {
        "frame": frame,
        "raw_state_shape": list(raw_state.shape),
        "preprocessed_state_shape": list(batch["observation.state"].shape),
        "preprocessed_state_matches_manual_mean_std": bool(
            torch.allclose(batch["observation.state"][0].cpu(), expected_norm, atol=1e-5)),
        "state_proj_input_shape": list(proj_in.shape),
        "state_proj_input_first16_equal_normalized_state": bool(
            torch.allclose(proj_in[0, :16], batch["observation.state"][0].float(), atol=1e-6)),
        "state_proj_input_pad_16_32_all_zero": bool(torch.all(proj_in[0, 16:] == 0)),
        "state_proj_output_shape": captured["out_shape"],
        "predicted_chunk_shape": list(chunk.shape),
        "postprocessed_action_shape": list(action.shape),
        "postprocessed_action_first_step": [round(float(x), 4) for x in action[0]],
        "dataset_action_this_frame": [round(float(x), 4) for x in item["action"]],
    }

    # One training forward/backward (no optimizer step) to inspect state_proj gradients.
    policy.train()
    policy.zero_grad(set_to_none=True)
    tb = dict(batch)
    tb["action"] = torch.randn(1, policy.config.chunk_size, 16)
    tb["action_is_pad"] = torch.zeros(1, policy.config.chunk_size, dtype=torch.bool)
    loss, _ = policy.forward(tb)
    loss.backward()
    g = policy.model.state_proj.weight.grad
    res.update({
        "state_proj_requires_grad": policy.model.state_proj.weight.requires_grad,
        "state_proj_grad_is_none": g is None,
        "state_proj_grad_colnorm_0_16": float(g[:, :16].norm()) if g is not None else None,
        "state_proj_grad_colnorm_16_32": float(g[:, 16:].norm()) if g is not None else None,
        "state_proj_grad_nonzero_columns": int((g.abs().sum(0) > 0).sum()) if g is not None else None,
        "num_trainable_params": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "state_proj_in_named_trainable": any(
            n == "model.state_proj.weight" for n, p in policy.named_parameters() if p.requires_grad),
        "vision_encoder_trainable_params": sum(
            p.numel() for n, p in policy.named_parameters() if p.requires_grad and "vision_model" in n),
    })
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runtime", nargs="*", default=["real_20", "sim_10"],
                    help="checkpoints to run the (slow, CPU) forward/backward on")
    ap.add_argument("--frame", type=int, default=120)
    args = ap.parse_args()
    torch.manual_seed(0)
    report = {}
    for name, (ckpt, ckpt_rev, repo, rev) in CHECKPOINTS.items():
        ckpt_dir = Path(snapshot_download(ckpt, revision=ckpt_rev,
                                          allow_patterns=["*.json", "*.safetensors", "tokenizer/*"]))
        ds = LeRobotDataset(repo, revision=rev)
        r = {"checkpoint": ckpt, "checkpoint_rev": ckpt_rev, "dataset": repo, "dataset_rev": rev}
        r.update(static_checks(ckpt_dir, ds))
        r.update(state_proj_weight_drift(ckpt_dir))
        if name in args.runtime:
            r["runtime"] = runtime_checks(ckpt_dir, ds, args.frame)
        report[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "max_abs_change_per_input_column"}, indent=1))
    args.out.write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
