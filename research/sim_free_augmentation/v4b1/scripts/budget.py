"""Synthetic-loss budget of the H2 phase mask under the frozen v4B RpPos schedules (seeds 0-2), and the matched
random-mask drop probability. Reads only parquet columns (no video). Writes outputs/budget.json.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/budget.py [--write_p_drop]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sfa4b1 import interventions as iv  # noqa: E402
from sfa.data import command_events  # noqa: E402

mx = iv.mx
V31 = HERE.parent / "v3_1"


def meta(name):
    root = V31 / "outputs/lerobot" / name
    ep = pd.read_parquet(sorted((root / "meta/episodes").rglob("*.parquet"))[0])
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))]).sort_values("index")
    return ep, df


def schedule_stats(arm_base, seeds, p_drop=None, ep_r=None, ep_s=None, df_s=None):
    rr = {int(e): (int(a), int(b)) for e, a, b in zip(ep_r["episode_index"], ep_r["dataset_from_index"], ep_r["dataset_to_index"])}
    sr = {int(e): (int(a), int(b)) for e, a, b in zip(ep_s["episode_index"], ep_s["dataset_from_index"], ep_s["dataset_to_index"])}
    nR = max(b for a, b in rr.values())
    A = np.stack(df_s["action"].to_numpy())
    ge = {e: command_events(A[a:b])["right_close"] + iv.CFG["grasp_end_offset"] for e, (a, b) in sr.items()}
    ep_of = np.empty(len(A), int)
    fr_of = np.empty(len(A), int)
    for e, (a, b) in sr.items():
        ep_of[a:b], fr_of[a:b] = e, np.arange(b - a)
    sel = mx.selection()
    out = {}
    for seed in seeds:
        real, syn = mx.frame_pools(arm_base, sel, rr, sr, nR)
        s = mx.StratifiedBatches(arm_base, sel, real, syn, 1000, 16, seed, mx.CFG["training"]["sampler_seed"], 0.5)
        nonpad = phase_valid = rand_valid = full_masked = n = 0
        for b in s:
            for i in b:
                if i < nR:
                    continue
                j = i - nR
                e, f = ep_of[j], fr_of[j]
                L = sr[e][1] - sr[e][0]
                pad = (f + np.arange(50)) >= L
                pm = iv.phase_mask(f, ge[e])
                nonpad += int((~pad).sum())
                phase_valid += int((~pad & ~pm).sum())
                full_masked += int((pad | pm).all())
                if p_drop is not None:
                    rand_valid += int((~pad & ~iv.random_mask(seed, i, p_drop)).sum())
                n += 1
        out[seed] = {"syn_draws": n, "nonpad_target_steps": nonpad, "phase_valid_steps": phase_valid,
                     "phase_valid_fraction": phase_valid / nonpad, "phase_fully_masked_rows_fraction": full_masked / n,
                     "random_valid_fraction": (rand_valid / nonpad) if p_drop is not None else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_p_drop", action="store_true")
    a = ap.parse_args()
    ep_r, _ = meta("sfa_v31_R")
    ep_s, df_s = meta("sfa_v31_Gp_matched_evidence_complete")
    st = schedule_stats("RpPos", [0, 1, 2], None, ep_r, ep_s, df_s)
    vf = sum(v["phase_valid_steps"] for v in st.values()) / sum(v["nonpad_target_steps"] for v in st.values())
    p_drop = round(1 - vf, 4)
    if a.write_p_drop:
        p = HERE / "configs/v4b1.json"
        c = json.loads(p.read_text())
        c["p_drop"] = p_drop
        p.write_text(json.dumps(c, indent=1))
        iv.CFG["p_drop"] = p_drop
    st2 = schedule_stats("RpPos", [0, 1, 2], p_drop, ep_r, ep_s, df_s)
    res = {"phase_mask": st, "pooled_phase_valid_fraction": vf, "p_drop": p_drop, "random_mask": st2,
           "note": "fractions are over non-padded synthetic target steps; real rows (8 per batch) keep all their targets"}
    (HERE / "outputs").mkdir(exist_ok=True)
    (HERE / "outputs/budget.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
