"""Task A: vision-dependence of the existing v3 SmolVLA checkpoints (no retraining), protocol configs/vision_ablation.json.

    cd research/sim_free_augmentation/v3_1
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/vision_ablation.py

Held-out real recordings (pool 10-49), every 5th frame, 50-step chunks; robot state, language, action targets, the
diffusion noise tensor and preprocessing are identical across checkpoints and image conditions. All predictions and
checkpoint hashes are saved.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
V3 = HERE.parent / "v3"
sys.path.insert(0, str(V3))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "research/contact_aware_augmentation/gate1"))
from sfa3.recon3 import data, rb  # noqa: E402

CFG = json.loads((HERE / "configs/vision_ablation.json").read_text())
H = CFG["heldout"]
POOL = Path.home() / f".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/{H['revision']}"
ARM = [rb.NAMES.index(f"{s}J{i}.pos") for s in "RL" for i in range(1, 8)]
GR, GL = rb.NAMES.index("RJ8.pos"), rb.NAMES.index("LJ8.pos")
CONDS = ["original", "blank", "wrong_episode", "object_occluded", "state_only"]
PHASES = ["approach", "grasp", "carry", "handover", "release"]
OUT = HERE / "outputs/vision_ablation"
PRED = HERE / "outputs/predictions_all"


def clicks():
    ann = REPO / "research/dataset_diagnostics/annotations/real_v00.csv"
    return {int(r["episode_index"]): (float(r["u"]), float(r["v"])) for r in csv.DictReader(open(ann))
            if r["dataset"] == "real_v00_pool" and r["verified"] == "1"}


def phase_of(T, ev):
    ph = np.empty(T, dtype=object)
    rc, lc, ro = ev["right_close"], ev["left_close"], ev["right_open"]
    for t in range(T):
        ph[t] = ("approach" if t < rc - 5 else "grasp" if t < rc + 15 else "carry" if t < lc - 5 else
                 "handover" if t < ro + 5 else "release")
    return ph


def occlusion_mask(rgb, lid, area):
    m = np.zeros(rgb.shape[:2], np.uint8)
    if lid is None:
        return m.astype(bool)
    r = int(max(45, 1.4 * np.sqrt(max(area, 1.0) / np.pi)))
    cv2.circle(m, (int(lid[0]), int(lid[1])), r, 1, -1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    near = np.zeros_like(m)
    cv2.circle(near, (int(lid[0]), int(lid[1])), 150, 1, -1)
    m |= red & near
    return cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0


def load_heldout(eps):
    import lid_tracker as lt
    from tools.dataset_diagnostics.loader import load_dataset
    view = load_dataset(root=POOL, label="pool", episodes=eps)
    S, A = view.vectors("observation.state"), view.vectors("action")
    ck = clicks()
    out = {}
    for e in eps:
        F = data.decode(view, data.CAMS[0], e)
        tr = lt.track(F, ck[e])
        T = len(S[e])
        ev = data.command_events(A[e])
        ts = list(range(0, T - CFG["heldout"]["chunk"], H["frame_stride"]))
        last, occ = None, {}
        for t in range(T):
            if np.isfinite(tr.uv[t]).all():
                last = (tr.uv[t], tr.area[t])
            if t in ts:
                lid, area = (last if last is not None else (np.array(ck[e]), 900.0))
                o = F[t].copy()
                o[occlusion_mask(F[t], lid, area)] = 128
                occ[t] = o
        out[e] = {"S": S[e].astype(np.float32), "A": A[e].astype(np.float32), "ts": ts, "F": {t: F[t] for t in ts}, "occ": occ,
                  "phase": phase_of(T, ev), "events": ev, "T": T, "click": ck[e]}
        print(f"  loaded pool ep{e}: {T} frames, {len(ts)} chunks, lid tracked {np.isfinite(tr.uv).all(1).mean():.2f}", flush=True)
    return out


def wrong_partner(eps, ck):
    return {e: max((x for x in eps if x != e), key=lambda x: np.hypot(ck[x][0] - ck[e][0], ck[x][1] - ck[e][1])) for e in eps}


def noise_for(e, t):
    rng = np.random.default_rng(20261105 + 1000 * e + t)
    return rng.standard_normal((CFG["heldout"]["chunk"], 32)).astype(np.float32)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_ckpt(ckpt, D, partner, device, batch=32):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}})
    task = CFG["fixed"]["language"]
    items = [(e, t) for e in D for t in D[e]["ts"]]
    preds = {}
    for cond in CONDS:
        P = np.zeros((len(items), CFG["heldout"]["chunk"], 16), np.float32)
        for b0 in range(0, len(items), batch):
            chunk = items[b0:b0 + batch]
            imgs = []
            for e, t in chunk:
                if cond in ("original", "state_only"):
                    im = D[e]["F"][t]
                elif cond == "blank":
                    im = np.full_like(D[e]["F"][t], 128)
                elif cond == "object_occluded":
                    im = D[e]["occ"][t]
                else:
                    p = partner[e]
                    rel = t / D[e]["T"]
                    tp = min(D[p]["ts"], key=lambda x: abs(x / D[p]["T"] - rel))
                    im = D[p]["F"][tp]
                imgs.append(im)
            b = {"observation.state": torch.from_numpy(np.stack([D[e]["S"][t] for e, t in chunk])),
                 "observation.images.camera1": torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0,
                 "task": [task] * len(chunk)}
            noise = torch.from_numpy(np.stack([noise_for(e, t) for e, t in chunk])).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bb = pre(b)
                if cond == "state_only":
                    bb["observation.images.camera1_padding_mask"] = torch.zeros(len(chunk), dtype=torch.bool, device=device)
                policy.reset()
                act = policy.predict_action_chunk(bb, noise=noise)
            with torch.inference_mode():
                act = post(act.float())
            P[b0:b0 + len(chunk)] = act.float().cpu().numpy()[:, :CFG["heldout"]["chunk"]]
        preds[cond] = P
    return items, preds


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PRED.mkdir(parents=True, exist_ok=True)
    eps = H["primary"] + H["extended"]
    D = load_heldout(eps)
    ck = clicks()
    partner = wrong_partner(eps, ck)
    t_load = time.time() - t0
    ckpts = sorted(glob.glob(str(V3 / "outputs/train/*/checkpoints/last/pretrained_model")))
    hashes = {Path(c).parents[2].name: {"model.safetensors": sha256(Path(c) / "model.safetensors"),
                                        "policy_preprocessor.json": sha256(Path(c) / "policy_preprocessor.json"),
                                        "normalizer": sha256(next(Path(c).glob("policy_preprocessor_step_*_normalizer_processor.safetensors")))}
              for c in ckpts}
    (OUT / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=1))
    res = {}

    def one(i, c):
        name = Path(c).parents[2].name
        t = time.time()
        items, preds = run_ckpt(c, D, partner, f"cuda:{i % 2}")
        np.savez_compressed(PRED / f"{name}.npz", items=np.array(items), **{k: v.astype(np.float16) for k, v in preds.items()})
        print(f"{name}: {len(items)} chunks x {len(CONDS)} conditions in {time.time() - t:.0f} s", flush=True)
        return name, items

    with ThreadPoolExecutor(2) as ex:
        for name, items in ex.map(lambda a: one(*a), enumerate(ckpts)):
            res[name] = items
    meta = {"episodes": {e: {"T": D[e]["T"], "events": D[e]["events"], "click": D[e]["click"], "partner": partner[e]} for e in eps},
            "load_s": t_load, "total_s": time.time() - t0, "checkpoints": list(res)}
    # targets/states/phases for scoring
    np.savez_compressed(OUT / "targets.npz", **{f"A_{e}": D[e]["A"] for e in eps}, **{f"S_{e}": D[e]["S"] for e in eps},
                        **{f"phase_{e}": np.array(D[e]["phase"], dtype="U9") for e in eps})
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
