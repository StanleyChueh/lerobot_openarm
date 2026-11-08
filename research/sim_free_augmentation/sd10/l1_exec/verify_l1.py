"""Post-hoc verification of saved SD10-L1 terminal states (reads files only; no training/inference).
Checks every run's saved optimizer state (Adam step == 1000 for every trainable tensor), model/processor/optimizer
hashes, saved normalizer safetensors == frozen arm statistics, schedule position, training log length/finiteness and
LR sequence == learning_rates.npy. Writes outputs/run_verification.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from safetensors.torch import load_file

import l1lib as L


def main():
    lrs = np.load(L.L1 / "learning_rates.npy")
    out, allok = {}, True
    for a in L.ARMS:
        for s in L.SEEDS:
            r = L.TRAIN / f"{a}_s{s}"
            term = json.loads((r / "terminal.json").read_text())
            opt = torch.load(r / "training_state/optimizer_state.pt", map_location="cpu", weights_only=True)
            steps = {float(v["step"]) for v in opt["state"].values()}
            log = json.loads((r / "train_log.json").read_text())
            pm = r / "pretrained_model"
            st = L.NORM["arms"][a]["stats"]
            nz = load_file(next(pm.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")))
            un = load_file(next(pm.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors")))
            norm_ok = all(np.array_equal(nz[f"{k}.{m}"].numpy(), np.array(st[k][m], np.float32)) for k in st for m in ("mean", "std")) \
                and all(np.array_equal(un[f"action.{m}"].numpy(), np.array(st["action"][m], np.float32)) for m in ("mean", "std"))
            ck = {"adam_step_all_1000": steps == {1000.0}, "n_param_tensors_with_state": len(opt["state"]),
                  "lr_in_saved_group": opt["param_groups"][0]["lr"], "lr_last_registered": float(lrs[-1]),
                  "log_updates": len(log), "log_lrs_equal_registered": np.array_equal([x["lr"] for x in log], lrs),
                  "losses_finite": bool(np.isfinite([x["loss"] for x in log]).all()),
                  "final_loss": log[-1]["loss"], "mean_loss_last100": float(np.mean([x["loss"] for x in log[-100:]])),
                  "model_sha256_matches": L.sha(pm / "model.safetensors") == term["model_sha256"],
                  "optimizer_sha256_matches": L.sha(r / "training_state/optimizer_state.pt") == term["optimizer_state_sha256"],
                  "saved_normalizers_equal_frozen": norm_ok,
                  "schedule_consumed_equals_frozen": term["schedule_state"]["consumed_equals_frozen"],
                  "model_sha256": term["model_sha256"]}
            ok = ck["adam_step_all_1000"] and ck["log_updates"] == 1000 and ck["log_lrs_equal_registered"] and ck["losses_finite"] \
                and ck["model_sha256_matches"] and ck["optimizer_sha256_matches"] and norm_ok and ck["schedule_consumed_equals_frozen"] \
                and ck["lr_in_saved_group"] == ck["lr_last_registered"]
            ck["PASS"] = bool(ok)
            allok &= ok
            out[f"{a}_s{s}"] = ck
    preds = {p.stem: {"npz_sha256": L.sha(p), **{k: v for k, v in json.loads(p.with_suffix(".json").read_text()).items()
                                                 if k in ("prediction_sha256", "model_sha256")}} for p in sorted(L.PRED.glob("*.npz"))}
    (L.OUT / "run_verification.json").write_text(json.dumps({"ALL_PASS": bool(allok), "runs": out, "predictions": preds}, indent=1))
    print("ALL_PASS", allok, {k: (v["PASS"], round(v["mean_loss_last100"], 4)) for k, v in out.items()})


if __name__ == "__main__":
    main()
