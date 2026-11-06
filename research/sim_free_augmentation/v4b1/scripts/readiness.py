"""v4B.1 execution-gate checks that need the model/GPU. Writes outputs/readiness.json.

  1. frozen reference checkpoints (v4B R, RpBoth, RpPos x 3 seeds) hash-match the v4B evaluation manifest
  2. model-level image intervention: SmolVLA training loss (fixed noise/time, reduction='none') of a synthetic row with
     padding_mask=False is invariant to its pixels (no leak through masked tokens), while with padding_mask=True the
     loss does depend on pixels (sensitivity sanity); real rows' losses are unchanged by synthetic rows' treatment
  3. loss normalisation after target masking equals sum(valid)/count(valid) recomputed by hand
  4. 20-step smoke training of every v4B.1 arm (model init + intervention path), budget logs read back

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/readiness.py --smoke_dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from sfa4b1 import interventions as iv  # noqa: E402

V4 = HERE.parent / "v4"
V31 = HERE.parent / "v3_1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke_dir", required=True)
    a = ap.parse_args()
    res = {}
    # 1
    man = json.loads((V4 / "outputs/v4b/predictions/checkpoint_hashes.json").read_text())
    ok = {}
    for arm in ("R", "RpBoth", "RpPos"):
        for s in iv.CFG["seeds"]:
            p = V4 / f"outputs/v4b/train/{arm}_s{s}/checkpoints/last/pretrained_model/model.safetensors"
            ok[f"{arm}_s{s}"] = sha(p) == man[f"{arm}_s{s}"]["model.safetensors"]
    res["frozen_reference_hashes_match"] = ok
    # 2 + 3
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.rename_processor import rename_batch_keys
    ck = V4 / "outputs/v4b/train/RpBoth_s0/checkpoints/last/pretrained_model"
    dev = "cuda:0"
    pol = SmolVLAPolicy.from_pretrained(str(ck)).to(dev).eval()
    pre, _ = make_pre_post_processors(pol.config, pretrained_path=str(ck), preprocessor_overrides={"device_processor": {"device": dev}})
    dt = {"action": [k / 30 for k in range(50)]}
    dr = LeRobotDataset("local/r", root=V31 / "outputs/lerobot/sfa_v31_R", delta_timestamps=dt)
    dsn = LeRobotDataset("local/s", root=V31 / "outputs/lerobot/sfa_v31_Gp_matched_evidence_complete", delta_timestamps=dt)
    view = iv.TwoPartV41(dr, dsn, "RpBoth_NoSynthRGB", 0)
    idx = [10, 500, view.nR + 40, view.nR + 9000]
    items = [view[i] for i in idx]
    from torch.utils.data import default_collate
    raw = default_collate(items)

    def prep(batch, pix=None, mask=None):
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        b.pop("sfa_is_synth"), b.pop("sfa_target_mask")
        if pix is not None:
            b["observation.images.body_cam"][2:] = pix
        b = rename_batch_keys(b, {"observation.images.body_cam": iv.CAM})
        out = pre(b)
        if mask is not None:
            out[f"{iv.CAM}_padding_mask"] = mask.to(dev)
        return out

    g = torch.Generator().manual_seed(0)
    noise = torch.randn(4, 50, 32, generator=g).to(dev)
    tm = torch.rand(4, generator=g).to(dev) * 0.9 + 0.05
    pixA = torch.rand(2, 3, 480, 640, generator=g)
    pixB = torch.rand(2, 3, 480, 640, generator=g)
    m_off = torch.tensor([True, True, False, False])
    with torch.no_grad():
        L = {}
        for name, pix, m in (("A_masked", pixA, m_off), ("B_masked", pixB, m_off), ("A_open", pixA, None), ("B_open", pixB, None)):
            l, _ = pol.forward(prep(raw, pix, m), noise=noise.clone(), time=tm.clone(), reduction="none")
            L[name] = l.float().cpu().numpy()
    res["image_intervention"] = {
        "synthetic_rows_masked_loss_invariant_to_pixels_max_abs": float(np.abs(L["A_masked"][2:] - L["B_masked"][2:]).max()),
        "synthetic_rows_open_loss_change_with_pixels_max_abs": float(np.abs(L["A_open"][2:] - L["B_open"][2:]).max()),
        "real_rows_loss_change_from_synthetic_treatment_max_abs": float(np.abs(L["A_masked"][:2] - L["A_open"][:2]).max()),
        "losses": {k: v.tolist() for k, v in L.items()}}
    # 3: loss normalisation with a target mask
    b = prep(raw)
    pad = b["action_is_pad"].clone()
    pad[2, 20:] = True
    pad[3, :] = True
    b["action_is_pad"] = pad
    with torch.no_grad():
        loss, _ = pol.forward(b, noise=noise.clone(), time=tm.clone())
        b2 = dict(b)
        b2["action_is_pad"] = torch.zeros_like(pad)
        per, _ = pol.forward(b2, noise=noise.clone(), time=tm.clone(), reduction="none")
    # recompute: per-element losses are needed; use the model forward directly
    with torch.no_grad():
        images, img_masks = pol.prepare_images(b)
        st = pol.prepare_state(b)
        acts = pol.prepare_action(b)
        el = pol.model.forward(images, img_masks, b["observation.language.tokens"], b["observation.language.attention_mask"], st, acts,
                               noise.clone(), tm.clone())[:, :, :16]
        valid = (~pad).unsqueeze(-1).float()
        manual = float((el * valid).sum() / (valid.sum() * 16 / 16 * 1).clamp_min(1) / 1.0)
        manual = float((el * valid).sum() / ((~pad).sum() * 16))
    res["loss_normalisation"] = {"policy_loss": float(loss), "manual_sum_over_valid_count": manual,
                                 "abs_diff": abs(float(loss) - manual), "valid_steps": int((~pad).sum())}
    del pol
    torch.cuda.empty_cache()
    # 4: smoke training
    smoke = {}
    sd = Path(a.smoke_dir)
    sd.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    for arm in iv.CFG["arms"]:
        out = sd / f"{arm}_s0"
        rc = subprocess.run([str(REPO / ".venv/bin/python"), str(HERE / "scripts/train_v4b1.py"), "--arm", arm, "--seed", "0", "--steps", "20",
                             "--out", str(out)], stdout=open(sd / f"{arm}.log", "w"), stderr=subprocess.STDOUT, env=env, cwd=HERE).returncode
        bud = json.loads((sd / f"{arm}_s0_budget.json").read_text())["summary"] if (sd / f"{arm}_s0_budget.json").exists() else None
        run = json.loads((sd / f"{arm}_s0_run.json").read_text()) if (sd / f"{arm}_s0_run.json").exists() else None
        frozen = json.loads((V4 / f"outputs/v4b/train/{iv.CFG['arms'][arm]['base']}_s0_mixture.json").read_text())["sampler"]["schedule_sha256"]
        smoke[arm] = {"rc": rc, "checkpoint": (out / "checkpoints/last/pretrained_model/model.safetensors").exists(), "budget": bud,
                      "schedule_differs_from_frozen_only_by_length": run is not None and run["sampler"]["schedule_sha256"] != frozen}
    res["smoke_training_20_steps"] = smoke
    (HERE / "outputs/readiness.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "image_intervention"} | {"image_intervention": {k: v for k, v in res["image_intervention"].items() if k != "losses"}}, indent=1, default=float))


if __name__ == "__main__":
    main()
