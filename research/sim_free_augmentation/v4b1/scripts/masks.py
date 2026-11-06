"""Amendment A1: expected supervision masks of the H2 arms under the frozen RpPos schedules (seeds 0-2) and the
loss-budget accounting that the per-row matched control must equalise. Parquet only (no video, no GPU).

Writes outputs/masks/{arm}_s{seed}.npz (index, is_syn, source, valid_packed: the retained target steps actually
supervised) and outputs/matched_budget.json.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/masks.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa4b1 import interventions as iv  # noqa: E402
from sfa4b1 import schedule_masks as sm  # noqa: E402

ARMS = ["RpPos_rerun", "RpPos_PhaseMask", "RpPos_MatchedRandMask", "RpPos_RandMask"]   # rerun = unmasked base; RandMask superseded (reference)


def account(E, base):
    v = E["valid"].sum(-1)                                   # [S,B] retained steps per row
    syn = E["is_syn"]
    den = v.sum(1)                                           # per-batch valid steps (x16 = loss denominator)
    vs = (v * syn).sum(1)
    src_target = np.bincount(E["source"][syn], weights=v[syn], minlength=10)
    src_mass = np.bincount(E["source"][syn], weights=(v / den[:, None])[syn], minlength=10) / len(den)
    return {"retained_synthetic_steps": int(vs.sum()),
            "synthetic_retention_vs_unmasked": float(vs.sum() / (base["valid"].sum(-1) * syn).sum()),
            "fully_masked_synthetic_row_fraction": float(((v == 0) & syn).sum() / syn.sum()),
            "mean_batch_synthetic_share_of_valid_elements": float(np.mean(vs / den)),
            "mean_real_element_multiplier_vs_unmasked": float(np.mean(base["valid"].sum(-1).sum(1) / den)),
            "batch_denominator_min_max": [int(den.min()), int(den.max())],
            "synthetic_direct_target_steps_by_source": src_target.astype(int).tolist(),
            "synthetic_loss_coefficient_mass_by_source": src_mass.tolist()}


def main():
    meta = sm.Meta()
    out = HERE / "outputs/masks"
    out.mkdir(parents=True, exist_ok=True)
    res = {"_doc": "A1 accounting under the frozen v4B RpPos schedules. 'valid' = target steps entering the loss; loss "
                   "denominator = 16 x per-batch valid steps. Coefficient mass = per-batch fraction of valid elements, not gradient norm.",
           "seeds": {}}
    for seed in iv.CFG["seeds"]:
        E = {a: sm.expected(meta, a, seed) for a in ARMS}
        base, ph = E["RpPos_rerun"], E["RpPos_PhaseMask"]
        q = {a: account(E[a], base) for a in ARMS}
        for a in ("RpPos_MatchedRandMask", "RpPos_RandMask"):
            vp, va = ph["valid"].sum(-1), E[a]["valid"].sum(-1)
            q[a]["vs_PhaseMask"] = {
                "rows_with_equal_retained_count": int((vp == va).sum()), "rows": int(vp.size),
                "batches_with_equal_denominator": int((vp.sum(1) == va.sum(1)).sum()), "batches": int(vp.shape[0]),
                "active_row_sets_equal_in_batches": int(((vp > 0) == (va > 0)).all(1).sum()),
                "max_real_weight_ratio": float((va.sum(1) / vp.sum(1)).max()),
                "min_real_weight_ratio": float((va.sum(1) / vp.sum(1)).min()),
                "synthetic_rows_with_identical_positions": int(((ph["valid"] == E[a]["valid"]).all(-1) & ph["is_syn"]).sum()),
                "synthetic_rows_with_0<K<valid": int((ph["is_syn"] & (vp > 0) & (vp < ph["valid_orig"].sum(-1))).sum())}
        for a in ("RpPos_PhaseMask", "RpPos_MatchedRandMask"):
            f = out / f"{a}_s{seed}.npz"
            np.savez_compressed(f, index=E[a]["index"], is_syn=E[a]["is_syn"], source=E[a]["source"],
                                valid_packed=np.packbits(E[a]["valid"], axis=-1))
            q[a]["manifest"] = {"path": str(f.relative_to(HERE)), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        res["seeds"][seed] = q
        m = q["RpPos_MatchedRandMask"]["vs_PhaseMask"]
        print(seed, "matched rows", m["rows_with_equal_retained_count"], "/", m["rows"], "denominators", m["batches_with_equal_denominator"],
              "| old RandMask denominators", q["RpPos_RandMask"]["vs_PhaseMask"]["batches_with_equal_denominator"], flush=True)
    (HERE / "outputs/matched_budget.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
