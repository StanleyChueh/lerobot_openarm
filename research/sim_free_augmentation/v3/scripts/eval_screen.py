"""Held-out offline evaluation of the training-screen checkpoints on EXTERNAL real episodes (pool 10-19), open loop:
at every 5th frame the policy predicts a 50-step chunk from (body_cam image, state, task) and is compared with the
recorded next 50 actions. No robot, no simulator. Decision rule: configs/train_screen.json (declared before training).

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/eval_screen.py
"""

from __future__ import annotations

import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from sfa3.recon3 import data, rb  # noqa: E402

CFG = json.loads((HERE / "configs/train_screen.json").read_text())
POOL = glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/*"))[0]
ARM = [rb.NAMES.index(f"{s}J{i}.pos") for s in "RL" for i in range(1, 8)]
GR, GL = rb.NAMES.index("RJ8.pos"), rb.NAMES.index("LJ8.pos")
H = CFG["heldout_eval"]


def heldout():
    from tools.dataset_diagnostics.loader import load_dataset
    view = load_dataset(root=POOL, label="pool", episodes=H["episodes"])
    S, A = view.vectors("observation.state"), view.vectors("action")
    out = {}
    for e in H["episodes"]:
        F = data.decode(view, data.CAMS[0], e)
        ev = data.command_events(A[e])
        T = len(S[e])
        ph = np.full(T, "retreat", dtype=object)
        ph[: ev["right_close"]] = "approach"
        ph[ev["right_close"]: ev["left_close"]] = "carry"
        ph[ev["left_close"]: min(T, ev["right_open"] + 10)] = "handover"
        out[e] = {"S": S[e].astype(np.float32), "A": A[e].astype(np.float32), "F": F, "phase": ph, "events": ev}
    return out


def closed(x, side):
    return np.clip(x / rb.GRIP_OPEN_RAW[side], 0, 1) < 0.5


def evaluate(ckpt, episodes, device):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt),
                                         preprocessor_overrides={"device_processor": {"device": device}})
    task = "Pick up the Pringles can with the right arm, hand it to the left arm"
    rows = []
    for e, d in episodes.items():
        T = len(d["S"])
        for t in range(0, T - H["chunk"], H["frame_stride"]):
            batch = {"observation.state": torch.from_numpy(d["S"][t])[None],
                     "observation.images.camera1": torch.from_numpy(d["F"][t]).permute(2, 0, 1)[None].float() / 255.0,
                     "task": [task]}
            torch.manual_seed(H["noise_seed"])
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                b = pre(batch)
                act = policy.predict_action_chunk(b)
            with torch.inference_mode():
                act = post(act.float())
            pred = act[0].float().cpu().numpy()[: H["chunk"]]
            rows.append({"e": e, "t": t, "phase": d["phase"][t], "pred": pred})
    return rows


def score(rows, episodes):
    allA = np.concatenate([d["A"] for d in episodes.values()])
    sd = allA[:, ARM].std(0) + 1e-6
    per_phase = {}
    joint = []
    skill_num, skill_den = [], []
    g_tp = g_fp = g_fn = 0
    timing = []
    for r in rows:
        d = episodes[r["e"]]
        t = r["t"]
        gt = d["A"][t: t + H["chunk"]]
        err = np.abs(r["pred"][:, ARM] - gt[:, ARM]) / sd
        per_phase.setdefault(r["phase"], []).append(err.mean())
        joint.append(err.mean(0))
        hold = np.abs(d["S"][t, ARM][None] - gt[:, ARM]) / sd
        skill_num.append(err.mean())
        skill_den.append(hold.mean())
        for side, i in (("R", GR), ("L", GL)):
            pc, gc = closed(r["pred"][:11, i], side), closed(gt[:11, i], side)
            g_tp += int((pc & gc).sum())
            g_fp += int((pc & ~gc).sum())
            g_fn += int((~pc & gc).sum())
            if gc.any() != gc.all():                       # a ground-truth transition inside the first 11 steps
                k_gt = int(np.argmax(gc != gc[0]))
                k_pr = int(np.argmax(pc != pc[0])) if (pc != pc[0]).any() else 11
                timing.append(abs(k_pr - k_gt))
    phases = ("approach", "carry", "handover", "retreat")
    pb = float(np.mean([np.mean(per_phase[p]) for p in phases if p in per_phase]))
    return {"phase_balanced_err": pb, "per_phase": {p: float(np.mean(per_phase[p])) for p in phases if p in per_phase},
            "per_joint": dict(zip([rb.NAMES[i] for i in ARM], np.mean(joint, 0).round(4).tolist())),
            "skill_vs_persistence": float(1 - np.mean(skill_num) / np.mean(skill_den)),
            "persistence_err": float(np.mean(skill_den)),
            "gripper_precision": g_tp / max(1, g_tp + g_fp), "gripper_recall": g_tp / max(1, g_tp + g_fn),
            "gripper_transition_timing_err_frames": float(np.mean(timing)) if timing else None, "n_transitions": len(timing),
            "n_samples": len(rows)}


def main():
    t0 = time.time()
    eps = heldout()
    t_load = time.time() - t0
    runs = sorted(glob.glob(str(HERE / "outputs/train/*/checkpoints/last/pretrained_model")))
    results = {}

    def one(i, ck):
        name = Path(ck).parents[2].name
        rows = evaluate(ck, eps, f"cuda:{i % 2}")
        return name, score(rows, eps)

    with ThreadPoolExecutor(2) as ex:
        for name, s in ex.map(lambda a: one(*a), enumerate(runs)):
            results[name] = s
            print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k not in ("per_joint", "per_phase")}, flush=True)
    # aggregate per arm and apply the pre-declared rule
    arms = {}
    for name, s in results.items():
        arm = name.rsplit("_s", 1)[0]
        arms.setdefault(arm, []).append(s)
    agg = {a: {"phase_balanced_err_mean": float(np.mean([s["phase_balanced_err"] for s in v])),
               "phase_balanced_err_sd": float(np.std([s["phase_balanced_err"] for s in v])),
               "skill_mean": float(np.mean([s["skill_vs_persistence"] for s in v])),
               "timing_mean": float(np.mean([s["gripper_transition_timing_err_frames"] for s in v if s["gripper_transition_timing_err_frames"] is not None])),
               "gripper_precision_mean": float(np.mean([s["gripper_precision"] for s in v])),
               "gripper_recall_mean": float(np.mean([s["gripper_recall"] for s in v])), "seeds": len(v)} for a, v in arms.items()}
    rule = {}
    if "R" in agg:
        base = agg["R"]
        for a in ("RpGp", "RpC"):
            if a in agg:
                rel = 1 - agg[a]["phase_balanced_err_mean"] / base["phase_balanced_err_mean"]
                rule[a] = {"rel_improvement_vs_R": rel, "timing_delta_frames": agg[a]["timing_mean"] - base["timing_mean"],
                           "augmentation_helps": bool(rel >= 0.10 and agg[a]["timing_mean"] - base["timing_mean"] <= 1.0)}
        if "RpC" in agg and "RpGp" in agg:
            rule["contact_helps"] = bool(rule["RpC"]["augmentation_helps"] and
                                         1 - agg["RpC"]["phase_balanced_err_mean"] / agg["RpGp"]["phase_balanced_err_mean"] >= 0.10)
    out = {"per_run": results, "per_arm": agg, "decision": rule, "heldout_episodes": H["episodes"], "load_s": t_load,
           "total_s": time.time() - t0}
    (HERE / "outputs/train_screen_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"per_arm": agg, "decision": rule}, indent=1))


if __name__ == "__main__":
    main()
