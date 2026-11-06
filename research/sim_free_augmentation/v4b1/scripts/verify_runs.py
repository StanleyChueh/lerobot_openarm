"""Amendment A1 post-training verification of every registered v4B.1 run (fails closed).

For each arm x seed: optimizer step counter, the batches the trainer ACTUALLY consumed (row indices recorded by the
applier) versus the frozen v4B schedule, the supervision mask actually applied versus the expected mask
(schedule_masks.expected), real padding never changed, the sampler schedule hash, and real-only normaliser identity
with the frozen references. Writes outputs/run_verification.json.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/verify_runs.py [--train_dir D] [--steps N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa4b1 import interventions as iv  # noqa: E402
from sfa4b1 import schedule_masks as sm  # noqa: E402

V4 = HERE.parent / "v4"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default=str(HERE / "outputs/train"))
    ap.add_argument("--steps", type=int, default=iv.CFG["steps"])
    ap.add_argument("--out", default=str(HERE / "outputs/run_verification.json"))
    a = ap.parse_args()
    td = Path(a.train_dir)
    meta = sm.Meta()
    ref_norm = json.loads((V4 / "outputs/v4b/predictions/checkpoint_hashes.json").read_text())["R_s0"]["normalizer"]
    runs = {}
    for arm, spec in iv.CFG["arms"].items():
        for seed in iv.CFG["seeds"]:
            n = f"{arm}_s{seed}"
            r, ck = {}, td / n / "checkpoints/last"
            try:
                r["optimizer_step"] = json.loads((ck / "training_state/training_step.json").read_text())["step"]
                rec = np.load(td / f"{n}_masks.npz")
                bud = json.loads((td / f"{n}_budget.json").read_text())["summary"]
                info = json.loads((td / f"{n}_run.json").read_text())
                E = sm.expected(meta, arm, seed, a.steps)
                got_valid = np.unpackbits(rec["valid_packed"], axis=-1)[..., :iv.CHUNK].astype(bool)
                r["preprocess_calls"] = int(len(rec["index"]))
                r["consumed_batches_equal_frozen_schedule"] = bool(np.array_equal(rec["index"], E["index"]))
                r["is_synthetic_equal"] = bool(np.array_equal(rec["is_syn"], E["is_syn"]))
                r["applied_mask_equal_expected"] = bool(np.array_equal(got_valid, E["valid"]))
                r["real_rows_fully_supervised_as_original"] = bool(np.array_equal(got_valid[~E["is_syn"]], E["valid_orig"][~E["is_syn"]]))
                r["original_padding_preserved"] = bool(not (got_valid & ~E["valid_orig"]).any())
                r["real_pad_ever_changed"] = bud["real_pad_ever_changed"]
                frozen = json.loads((V4 / f"outputs/v4b/train/{spec['base']}_s{seed}_mixture.json").read_text())["sampler"]["schedule_sha256"]
                r["schedule_sha256_equal_frozen"] = info["sampler"]["schedule_sha256"] == frozen
                r["git_head_at_launch"] = info["prelaunch"]["git_head"]
                r["v4b1_dirty_at_launch"] = info["prelaunch"]["v4b1_dirty"] + info["prelaunch"].get("v4b1_untracked_code", [])
                pm = ck / "pretrained_model"
                r["model_sha256"] = sha(pm / "model.safetensors")
                r["normalizer_sha256"] = sha(next(pm.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")))
                r["normalizer_equal_frozen_real_only"] = r["normalizer_sha256"] == ref_norm
                r["pass"] = bool(r["optimizer_step"] == a.steps and r["preprocess_calls"] == a.steps and r["consumed_batches_equal_frozen_schedule"]
                                 and r["is_synthetic_equal"] and r["applied_mask_equal_expected"] and r["real_rows_fully_supervised_as_original"]
                                 and r["original_padding_preserved"] and not r["real_pad_ever_changed"] and r["schedule_sha256_equal_frozen"]
                                 and r["normalizer_equal_frozen_real_only"] and not r["v4b1_dirty_at_launch"])
            except Exception as ex:  # missing artefact -> fail closed
                r["error"], r["pass"] = repr(ex), False
            runs[n] = r
            print(n, "PASS" if r["pass"] else f"FAIL {r.get('error', '')}", flush=True)
    res = {"steps": a.steps, "train_dir": str(td), "runs": runs, "all_pass": all(r["pass"] for r in runs.values()),
           "n_runs": len(runs)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("all_pass", res["all_pass"])
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
