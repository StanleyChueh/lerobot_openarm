"""Post-training check requested as non-blocking item 3 of the A1 re-audit: inspect the SAVED optimizer/scheduler state
of every registered v4B.1 run (not only training_step.json), the checkpoint layout (no intermediate or resumed
checkpoints), the saved train config (resume=False, identical optimiser/scheduler/batch to the frozen v4B reference
of the same base arm and seed), and that the evaluated checkpoint hashes equal a fresh rehash of the model bytes.
Read-only on frozen v4B artefacts. Writes outputs/optimizer_verification.json; exits 1 on any failure.

    env -u PYTHONPATH -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/verify_optimizer_state.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CFG = json.loads((HERE / "configs/v4b1.json").read_text())
V4T = HERE.parent / "v4/outputs/v4b/train"
SAME = ("steps", "batch_size", "seed", "optimizer", "scheduler", "save_freq", "num_workers")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def state(run_dir):
    from safetensors import safe_open
    ts = run_dir / "checkpoints/last/training_state"
    with safe_open(str(ts / "optimizer_state.safetensors"), "pt") as f:
        steps = [float(f.get_tensor(k)) for k in f.keys() if k.endswith("/step")]
    sch = json.loads((ts / "scheduler_state.json").read_text())
    pg = json.loads((ts / "optimizer_param_groups.json").read_text())
    return {"adamw_param_states": len(steps), "adamw_step_min": min(steps), "adamw_step_max": max(steps),
            "scheduler_last_epoch": sch["last_epoch"], "scheduler_step_count": sch["_step_count"], "final_lr": sch["_last_lr"],
            "param_groups": [{k: v for k, v in g.items() if k != "params"} for g in pg],
            "trainer_step": json.loads((ts / "training_step.json").read_text())["step"]}


def main():
    steps = CFG["steps"]
    ev = json.loads((HERE / "outputs/predictions/checkpoint_hashes.json").read_text())
    runs = {}
    for arm, spec in CFG["arms"].items():
        for seed in CFG["seeds"]:
            n, r = f"{arm}_s{seed}", {}
            try:
                d = HERE / "outputs/train" / n
                ref = V4T / f"{spec['base']}_s{seed}"
                r.update(state(d))
                rs = state(ref)
                r["checkpoint_dirs"] = sorted(p.name for p in (d / "checkpoints").iterdir())
                tc = json.loads((d / "checkpoints/last/pretrained_model/train_config.json").read_text())
                rc = json.loads((ref / "checkpoints/last/pretrained_model/train_config.json").read_text())
                r["resume"] = tc.get("resume")
                r["train_config_diff_vs_frozen_base"] = {k: [tc.get(k), rc.get(k)] for k in SAME if tc.get(k) != rc.get(k)}
                r["param_groups_equal_frozen_base"] = r["param_groups"] == rs["param_groups"]
                r["scheduler_equal_frozen_base"] = (r["scheduler_last_epoch"], r["scheduler_step_count"], r["final_lr"]) == \
                                                   (rs["scheduler_last_epoch"], rs["scheduler_step_count"], rs["final_lr"])
                r["model_sha256_rehash"] = sha(d / "checkpoints/last/pretrained_model/model.safetensors")
                r["evaluated_hash_equal_rehash"] = ev[n]["model.safetensors"] == r["model_sha256_rehash"]
                r["pass"] = bool(r["adamw_step_min"] == r["adamw_step_max"] == steps and r["trainer_step"] == steps
                                 and r["scheduler_last_epoch"] == steps and r["scheduler_step_count"] == steps + 1
                                 and r["adamw_param_states"] == rs["adamw_param_states"] and r["checkpoint_dirs"] == ["001000", "last"]
                                 and r["resume"] is False and not r["train_config_diff_vs_frozen_base"]
                                 and r["param_groups_equal_frozen_base"] and r["scheduler_equal_frozen_base"] and r["evaluated_hash_equal_rehash"])
            except Exception as ex:  # fail closed
                r["error"], r["pass"] = repr(ex), False
            r.pop("param_groups", None)
            runs[n] = r
            print(n, "PASS" if r["pass"] else f"FAIL {r.get('error', '')}", flush=True)
    res = {"steps": steps, "runs": runs, "all_pass": all(r["pass"] for r in runs.values()), "n_runs": len(runs)}
    (HERE / "outputs/optimizer_verification.json").write_text(json.dumps(res, indent=1))
    print("all_pass", res["all_pass"])
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
