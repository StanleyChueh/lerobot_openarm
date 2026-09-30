#!/usr/bin/env python
"""Open-loop check of SmolVLA checkpoints against recorded demos, split by task phase.

For sampled frames of each dataset, runs the policy exactly like deploy_smolvla_async.py does
(preprocessor -> predict_action_chunk -> postprocessor) and compares the predicted 50-step chunk
with the demo's own next 50 actions. Every (policy, dataset) pair is evaluated, so passing two
checkpoints and their two datasets also gives the cross-dataset numbers.

Phases come from the demo's gripper events (same segmentation as analyze_dataset_quality.py):
  approach   before the right gripper grasps
  carry      right holds, left not yet
  handover   both hold
  retreat    after the right releases

Per phase it reports:
  arm_mae10 / arm_mae50   mean |pred - demo| over arm joints, first 10 / all 50 steps (rad)
  R_close_ok / L_close_ok fraction of steps where the demo holds and the BINARIZED prediction
                          (deploy's Schmitt trigger: close < --close-below, open > --open-above)
                          would also be closed
  R_false_close           fraction of steps where the demo is open but the binarized prediction closes
  R_pred_hold / L_pred_hold median raw gripper prediction while the demo holds
  seed_spread             |pred(seed 0) - pred(seed 1)| on arm joints: how undecided the policy is
  noise_sens              |pred(state + N(0, --state-noise)) - pred| on arm joints, same seed

Usage:
  python offline_policy_eval.py \\
      --policy A=ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz B=ethanCSL/openarm_pringles_v0_smolvla_20k \\
      --dataset A=ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz B=ethanCSL/openarm_pringles_v0
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from analyze_dataset_quality import gripper_segments
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Copied from deploy_smolvla_pickup_jointspace.py, which can't be imported without the robot stack
# (pinocchio). Keep in sync with GRIPPER_CLOSE_BELOW / GRIPPER_OPEN_ABOVE / TASK there.
GRIPPER_CLOSE_BELOW = 0.035
GRIPPER_OPEN_ABOVE = 0.042
TASK = "Pick up the Pringles can with the right arm, hand it to the left arm"

PHASES = ["approach", "carry", "handover", "retreat"]
OPEN_VAL = 0.044


def phase_labels(actions, rg, lg, n):
    """Per-frame phase from the demo's gripper events; None when the episode has no clean handover."""
    rseg, _ = gripper_segments(actions[:, rg], OPEN_VAL, 0.0015, 0.004, 10)
    lseg, _ = gripper_segments(actions[:, lg], OPEN_VAL, 0.0015, 0.004, 10)
    rhold = [g for g in rseg if not g["blip"] and g["min"] >= 0.016]
    if not rhold or not lseg or lseg[-1]["end"] != n:
        return None
    rmain = max(rhold, key=lambda g: g["len"])
    grasp, release, left_grasp = rhold[0]["start"], rmain["end"], lseg[-1]["start"]
    lab = np.empty(n, dtype=object)
    for t in range(n):
        lab[t] = "approach" if t < grasp else "carry" if t < left_grasp else "handover" if t < release else "retreat"
    return lab


def binarize(chunk_g, start_cmd, close_below, open_above):
    """Deploy's Schmitt trigger applied along a predicted gripper sequence; True = closed."""
    out, cmd = np.empty(len(chunk_g), bool), start_cmd
    for i, v in enumerate(chunk_g):
        cmd = True if v < close_below else False if v > open_above else cmd
        out[i] = cmd
    return out


def compat_checkpoint(path, work_dir):
    """Local copy of a checkpoint saved by a newer lerobot, patched so this lerobot can load it:
    config fields SmolVLAConfig doesn't know are dropped, and a tokenizer bundled in the checkpoint
    is referenced by absolute path. Weights and normalisation stats are untouched."""
    import dataclasses
    import json
    import shutil

    from huggingface_hub import snapshot_download

    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    src = path if os.path.isdir(path) else snapshot_download(path)
    dst = os.path.join(work_dir, "_compat", path.replace("/", "__"))
    shutil.copytree(src, dst, dirs_exist_ok=True)
    cfg_path = os.path.join(dst, "config.json")
    cfg = json.load(open(cfg_path))
    known = {f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}
    dropped = sorted(set(cfg) - known)
    json.dump({k: v for k, v in cfg.items() if k in known}, open(cfg_path, "w"), indent=4)
    pre_path = os.path.join(dst, "policy_preprocessor.json")
    pre = json.load(open(pre_path))
    for step in pre["steps"]:
        name = step.get("config", {}).get("tokenizer_name")
        if name and os.path.isdir(os.path.join(dst, name)):
            step["config"]["tokenizer_name"] = os.path.join(dst, name)
            step.pop("artifacts", None)
    json.dump(pre, open(pre_path, "w"), indent=1)
    print(f"  [compat] {path}: dropped config fields {dropped}")
    return dst


def load_policy(path, device, work_dir):
    try:
        model = SmolVLAPolicy.from_pretrained(path)
    except Exception as e:  # draccus DecodingError on fields from a newer lerobot
        if "not valid for SmolVLAConfig" not in str(e):
            raise
        path = compat_checkpoint(path, work_dir)
        model = SmolVLAPolicy.from_pretrained(path)
    model.to(device).eval()
    pre, post = make_pre_post_processors(model.config, path)
    return model, pre, post


@torch.inference_mode()
def predict(model, pre, post, batch, seed, device):
    torch.manual_seed(seed)
    chunk = model.predict_action_chunk(pre(batch))  # (B, 50, action_dim), normalised
    b, h, d = chunk.shape
    out = post(chunk.reshape(b * h, d))
    out = out["action"] if isinstance(out, dict) else out
    return out.reshape(b, h, -1).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", nargs="+", required=True, help="label=checkpoint (HF id or local path)")
    ap.add_argument("--dataset", nargs="+", required=True, help="label=HF dataset repo id")
    ap.add_argument("--episodes", type=int, default=20, help="Episodes per dataset (taken from the first 100, one video file)")
    ap.add_argument("--stride", type=int, default=8, help="Evaluate every Nth frame of each episode")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--state-noise", type=float, default=0.02, help="rad of Gaussian noise on arm joints for noise_sens")
    ap.add_argument("--close-below", type=float, default=GRIPPER_CLOSE_BELOW)
    ap.add_argument("--open-above", type=float, default=GRIPPER_OPEN_ABOVE)
    ap.add_argument("--task", type=str, default=TASK)
    ap.add_argument("--out", type=str, default="outputs/offline_eval")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    datasets = {}
    for spec in args.dataset:
        label, repo = spec.split("=", 1)
        ep_ids = list(range(0, 100, max(1, 100 // args.episodes)))[: args.episodes]
        fps = LeRobotDatasetMetadata(repo).fps
        ds = LeRobotDataset(repo, episodes=ep_ids, delta_timestamps={"action": [i / fps for i in range(50)]})
        names = ds.meta.features["action"]["names"]
        rg, lg = names.index("RJ8.pos"), names.index("LJ8.pos")
        arm = [i for i, n in enumerate(names) if not n.endswith("8.pos")]
        hf = ds.hf_dataset.with_format("numpy")
        ep_col, fr_col = np.asarray(hf["episode_index"]), np.asarray(hf["frame_index"])
        acts = np.stack(hf["action"])
        samples = []
        for e in ep_ids:
            rows = np.where(ep_col == e)[0]
            lab = phase_labels(acts[rows], rg, lg, len(rows))
            if lab is None:
                continue
            for t in range(0, len(rows) - 1, args.stride):
                samples.append((int(rows[t]), lab[t], int(e), int(fr_col[rows[t]])))
        datasets[label] = (ds, samples, rg, lg, arm)
        print(f"[data] {label}: {len(samples)} frames from {len(ep_ids)} episodes")

    rows = []
    for pspec in args.policy:
        plabel, path = pspec.split("=", 1)
        print(f"[policy] loading {plabel} = {path}")
        model, pre, post = load_policy(path, device, args.out)
        for dlabel, (ds, samples, rg, lg, arm) in datasets.items():
            for i0 in range(0, len(samples), args.batch_size):
                chunk = samples[i0 : i0 + args.batch_size]
                items = [ds[s[0]] for s in chunk]
                gt = torch.stack([it["action"] for it in items]).numpy()
                is_pad = torch.stack([it["action_is_pad"] for it in items]).numpy()

                def make_batch(noise=None):
                    b = {k: torch.stack([it[k] for it in items]) for k in items[0] if k.startswith("observation.")}
                    if noise is not None:
                        b["observation.state"] = b["observation.state"].clone()
                        b["observation.state"][:, arm] += noise
                    b["task"] = [args.task] * len(items)
                    return b

                p0 = predict(model, pre, post, make_batch(), 0, device)
                p1 = predict(model, pre, post, make_batch(), 1, device)
                g = torch.Generator().manual_seed(i0)
                pn = predict(model, pre, post, make_batch(torch.randn(len(items), len(arm), generator=g) * args.state_noise), 0, device)
                for j, (_, phase, e, f) in enumerate(chunk):
                    valid = ~is_pad[j]
                    P, G = p0[j][valid], gt[j][valid]
                    state = items[j]["observation.state"].numpy()
                    rec = {"policy": plabel, "dataset": dlabel, "episode": e, "frame": f, "phase": phase,
                           "arm_mae10": np.abs(P[:10, arm] - G[:10, arm]).mean(),
                           "arm_mae50": np.abs(P[:, arm] - G[:, arm]).mean(),
                           "seed_spread": np.abs(p0[j][valid][:, arm] - p1[j][valid][:, arm]).mean(),
                           "noise_sens": np.abs(pn[j][valid][:, arm] - P[:, arm]).mean()}
                    for side, gi in (("R", rg), ("L", lg)):
                        gt_closed = G[:, gi] < OPEN_VAL - 0.004
                        pred_closed = binarize(P[:, gi], state[gi] < OPEN_VAL - 0.004, args.close_below, args.open_above)
                        rec[f"{side}_gt_closed_steps"] = gt_closed.sum()
                        rec[f"{side}_close_ok_steps"] = (gt_closed & pred_closed).sum()
                        rec[f"{side}_gt_open_steps"] = (~gt_closed).sum()
                        rec[f"{side}_false_close_steps"] = (~gt_closed & pred_closed).sum()
                        rec[f"{side}_pred_hold"] = float(np.median(P[gt_closed, gi])) if gt_closed.any() else np.nan
                    rows.append(rec)
            print(f"  {plabel} on {dlabel}: done")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "frames.csv"), index=False)

    def agg(g):
        s = {"n": len(g), "arm_mae10": g.arm_mae10.mean(), "arm_mae50": g.arm_mae50.mean(),
             "seed_spread": g.seed_spread.mean(), "noise_sens": g.noise_sens.mean()}
        for side in "RL":
            c = g[f"{side}_gt_closed_steps"].sum()
            o = g[f"{side}_gt_open_steps"].sum()
            s[f"{side}_close_ok"] = g[f"{side}_close_ok_steps"].sum() / c if c else np.nan
            s[f"{side}_false_close"] = g[f"{side}_false_close_steps"].sum() / o if o else np.nan
            s[f"{side}_pred_hold"] = g[f"{side}_pred_hold"].median() if g[f"{side}_pred_hold"].notna().any() else np.nan
        return pd.Series(s)

    by_phase = df.groupby(["policy", "dataset", "phase"]).apply(agg, include_groups=False).reindex(PHASES, level="phase")
    overall = df.groupby(["policy", "dataset"]).apply(agg, include_groups=False)
    by_phase.to_csv(os.path.join(args.out, "by_phase.csv"))
    overall.to_csv(os.path.join(args.out, "overall.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n=== overall\n", overall)
        print("\n=== by phase\n", by_phase)
    print(f"\n(binarization: close < {args.close_below}, open > {args.open_above})  wrote {args.out}/")


if __name__ == "__main__":
    main()
