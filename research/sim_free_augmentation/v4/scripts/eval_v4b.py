"""v4B held-out inference (B3 + B4). Reuses the frozen v3.1 evaluator pieces (v3_1/scripts/vision_ablation.py): same
pool revision, recordings 10-49, frame stride 5, 50-step chunks, the per-(episode, frame) N(0,1) noise tensors, bf16
autocast, each checkpoint's own saved processors, the farthest-initial-lid wrong-episode partner.

Image conditions (states, language, targets, noise fixed):
  original, blank, wrong_episode, object_only, background (equal-area), gripper_only   (masks: sfa4b/vis_masks.py)

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/eval_v4b.py \
        [--train_dir D ...] [--pred_dir P] [--episodes ...] [--viz]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
V31 = HERE.parent / "v3_1"
sys.path.insert(0, str(V31 / "scripts"))
sys.path.insert(0, str(HERE))
import vision_ablation as va  # noqa: E402
from sfa4b import vis_masks as vm  # noqa: E402

CONDS = ["original", "blank", "wrong_episode", "object_only", "background", "gripper_only"]
OUT = HERE / "outputs/v4b"
MASKS = OUT / "masks"


def load(eps):
    import lid_tracker as lt
    from tools.dataset_diagnostics.loader import load_dataset
    view = load_dataset(root=va.POOL, label="pool", episodes=eps)
    S, A = view.vectors("observation.state"), view.vectors("action")
    ck = va.clicks()
    robot, cam = vm.rb.Robot(), vm.rb.Camera()
    MASKS.mkdir(parents=True, exist_ok=True)
    out = {}
    for e in eps:
        F = va.data.decode(view, va.data.CAMS[0], e)
        T = len(S[e])
        ts = list(range(0, T - va.H["chunk"], va.H["frame_stride"]))
        mf = MASKS / f"pool{e}.npz"
        if mf.exists():
            z = np.load(mf)
            M = {k: np.unpackbits(z[k], axis=-1)[..., :vm.W].astype(bool) for k in ("object_only", "gripper_only", "background")}
            how = list(z["how"])
        else:
            tr = lt.track(F, ck[e])
            last, M, how = None, {k: [] for k in ("object_only", "gripper_only", "background")}, []
            for t in range(T):
                if np.isfinite(tr.uv[t]).all():
                    last = (tr.uv[t], tr.area[t])
                if t in ts:
                    lid, area = last if last is not None else (np.array(ck[e]), 900.0)
                    m, h = vm.all_masks(robot, cam, F[t], S[e][t].astype(float), lid, area)
                    for k in M:
                        M[k].append(m[k])
                    how.append(h)
            M = {k: np.stack(v) for k, v in M.items()}
            np.savez_compressed(mf, how=np.array(how), **{k: np.packbits(v, axis=-1) for k, v in M.items()})
        out[e] = {"S": S[e].astype(np.float32), "A": A[e].astype(np.float32), "ts": ts, "F": {t: F[t] for t in ts},
                  "M": {k: {t: M[k][i] for i, t in enumerate(ts)} for k in M}, "how": dict(zip(ts, how)),
                  "phase": va.phase_of(T, va.data.command_events(A[e])), "T": T}
        print(f"  pool ep{e}: {T} frames, {len(ts)} chunks", flush=True)
    return out


def image(D, partner, e, t, cond):
    im = D[e]["F"][t]
    if cond == "original":
        return im
    if cond == "blank":
        return np.full_like(im, 128)
    if cond == "wrong_episode":
        p = partner[e]
        tp = min(D[p]["ts"], key=lambda x: abs(x / D[p]["T"] - t / D[e]["T"]))
        return D[p]["F"][tp]
    o = im.copy()
    o[D[e]["M"][cond][t]] = 128
    return o


def run_ckpt(ckpt, D, partner, device, conds, batch=32):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(str(ckpt)).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt), preprocessor_overrides={"device_processor": {"device": device}})
    items = [(e, t) for e in D for t in D[e]["ts"]]
    preds = {}
    for cond in conds:
        P = np.zeros((len(items), va.H["chunk"], 16), np.float32)
        for b0 in range(0, len(items), batch):
            chunk = items[b0:b0 + batch]
            b = {"observation.state": torch.from_numpy(np.stack([D[e]["S"][t] for e, t in chunk])),
                 "observation.images.camera1": torch.from_numpy(np.stack([image(D, partner, e, t, cond) for e, t in chunk])).permute(0, 3, 1, 2).float() / 255.0,
                 "task": [va.CFG["fixed"]["language"]] * len(chunk)}
            noise = torch.from_numpy(np.stack([va.noise_for(e, t) for e, t in chunk])).to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bb = pre(b)
                policy.reset()
                act = policy.predict_action_chunk(bb, noise=noise)
            with torch.inference_mode():
                act = post(act.float())
            P[b0:b0 + len(chunk)] = act.float().cpu().numpy()[:, :va.H["chunk"]]
        preds[cond] = P
    return items, preds


def viz(D, path):
    tiles = []
    for e in list(D)[:6]:
        for t in D[e]["ts"][::max(1, len(D[e]["ts"]) // 4)][:4]:
            im = D[e]["F"][t].copy()
            for k, col in (("object_only", (255, 0, 255)), ("gripper_only", (0, 255, 255)), ("background", (0, 255, 0))):
                m = D[e]["M"][k][t].astype(np.uint8)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, col, 2)
            cv2.putText(im, f"pool{e} t{t} {D[e]['phase'][t]}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(cv2.resize(im, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", nargs="+", default=[str(OUT / "train")])
    ap.add_argument("--pred_dir", default=str(OUT / "predictions"))
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--conds", nargs="*", default=CONDS)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    pred = Path(a.pred_dir)
    pred.mkdir(parents=True, exist_ok=True)
    eps = a.episodes or (va.H["primary"] + va.H["extended"])
    D = load(eps)
    if a.viz:
        viz(D, OUT / "mask_check.jpg")
        stats = {k: float(np.mean([D[e]["M"][k][t].mean() for e in D for t in D[e]["ts"]])) for k in ("object_only", "gripper_only", "background")}
        how = [h for e in D for h in D[e]["how"].values()]
        print("mean mask area fraction", stats, "background placement", {h: how.count(h) for h in set(x.split("(")[0] for x in how)} if how else {})
        (OUT / "mask_stats.json").write_text(json.dumps({"mean_area_fraction": stats, "n_frames": len(how),
                                                         "background_fallback_disk": sum(h == "disk_fallback" for h in how),
                                                         "empty_object": sum(h == "empty" for h in how)}, indent=1))
        return
    ckpts = sorted(c for d in a.train_dir for c in glob.glob(str(Path(d) / "*/checkpoints/last/pretrained_model")))
    name = lambda c: a.prefix + Path(c).parents[2].name  # noqa: E731
    todo = [c for c in ckpts if not (pred / f"{name(c)}.npz").exists()]
    print(f"{len(ckpts)} checkpoints, {len(todo)} to evaluate on {len(eps)} recordings x {len(a.conds)} conditions", flush=True)
    partner = va.wrong_partner(eps, va.clicks())
    hp = pred / "checkpoint_hashes.json"
    hashes = json.loads(hp.read_text()) if hp.exists() else {}

    def one(i, c):
        t = time.time()
        hashes[name(c)] = {"path": str(c), "model.safetensors": va.sha256(Path(c) / "model.safetensors"),
                           "normalizer": va.sha256(next(Path(c).glob("policy_preprocessor_step_*_normalizer_processor.safetensors")))}
        items, preds = run_ckpt(c, D, partner, f"cuda:{i % 2}", a.conds)
        np.savez_compressed(pred / f"{name(c)}.npz", items=np.array(items), **{k: v.astype(np.float16) for k, v in preds.items()})
        print(f"{name(c)}: {len(items)} chunks x {len(a.conds)} in {time.time() - t:.0f} s", flush=True)

    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda x: one(*x), enumerate(todo)))
    hp.write_text(json.dumps(hashes, indent=1))
    np.savez_compressed(pred / "targets.npz", **{f"A_{e}": D[e]["A"] for e in eps}, **{f"phase_{e}": np.array(D[e]["phase"], dtype="U9") for e in eps})
    (pred / "eval_meta.json").write_text(json.dumps({"episodes": eps, "conditions": a.conds, "partner": partner, "pool_revision": va.H["revision"],
                                                     "evaluator": "v4/scripts/eval_v4b.py (v3_1 vision_ablation noise/processors/partner)",
                                                     "total_s": time.time() - t0}, indent=1, default=int))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
