"""Estimate success counts of the evaluation runs that were NOT recorded, from outputs/eval_logs/*/ticks.csv.

Those runs logged the measured state only on ticks that sent an observation (NaN otherwise); the state is
forward-filled between reads. The gripper-based stable-hand-over detector (eval_analysis.py) is then applied.
Validation: the two recorded runs (state on every tick) are scored twice -- with full state, and with their state
thinned to the SAME per-episode read pattern as an unrecorded run -- and compared with the simulator's labels.
"""
import glob
import json

import numpy as np
import pandas as pd

from common import DATA, DATASETS, NAMES, REPO
from eval_analysis import HOLD_STEPS, holding

LOGS = {d.split("/")[-1]: d for d in sorted(glob.glob(str(REPO / "outputs/eval_logs/*pringles*sim*")))}


def load_ticks(d):
    t = pd.read_csv(d + "/ticks.csv")
    return t[t.sent == 1].reset_index(drop=True)


def detect(t, read_mask=None):
    A = t[[f"a_{n}" for n in NAMES]].to_numpy(float)
    S = t[[f"s_{n}" for n in NAMES]].copy()
    if read_mask is not None:
        S[~read_mask] = np.nan
    S = S.groupby(t.episode).ffill().bfill().to_numpy(float)
    res = []
    for ep, idx in t.groupby("episode").indices.items():
        a, s = A[idx], S[idx]
        hR, hL = holding(s, a, "R"), holding(s, a, "L")
        tR = np.flatnonzero(hR)
        ok, cur = False, 0
        for k, p in enumerate(hL & ~hR):
            cur = cur + 1 if p else 0
            if cur >= HOLD_STEPS and len(tR) and k > tR[0]:
                ok = True
                break
        res.append(ok)
    return res


out = {}
sparse_ref = None
for name, d in LOGS.items():
    t = load_ticks(d)
    present = t["s_RJ1.pos"].notna().to_numpy()
    det = detect(t)
    out[name] = {"episodes": int(t.episode.nunique()), "state_read_frac": round(float(present.mean()), 3),
                 "detected_successes": int(sum(det)), "per_episode": [int(x) for x in det]}
    if present.mean() < 0.9 and sparse_ref is None:
        sparse_ref = (name, t.groupby("episode")["s_RJ1.pos"].apply(lambda s: s.notna().to_numpy()).tolist())
    print(name, out[name]["episodes"], "eps | state read on", out[name]["state_read_frac"], "of ticks | detected", out[name]["detected_successes"])

for name, ds in [("20261025_122100_async_rtc_openarm_pringles_lerobot_sim_smolvla_10_v00", "eval_manual"),
                 ("20261024_234225_async_rtc_openarm_pringles_lerobot_sim_10_generated_500_smolvla_v00", "eval_generated")]:
    lab = [int(e["success"]) for e in json.load(open(DATASETS[ds] / "meta/openarm_eval.json"))["episodes"]]
    t = load_ticks(LOGS[name])
    # thin to the sparse run's per-episode read pattern (cycled), i.e. the same read cadence
    masks = []
    for k, (ep, idx) in enumerate(t.groupby("episode").indices.items()):
        ref = sparse_ref[1][k % len(sparse_ref[1])]
        masks.append(np.resize(ref, len(idx)))
    thin = detect(t, np.concatenate(masks))
    full = out[name]["per_episode"]
    out[name]["sim_label_successes"] = sum(lab)
    out[name]["agreement_full_state"] = f"{sum(a == b for a, b in zip(lab, full))}/100"
    out[name]["agreement_thinned_state"] = f"{sum(a == b for a, b in zip(lab, thin))}/100 (detected {sum(thin)})"
    print(name, "sim", sum(lab), "| full-state agreement", out[name]["agreement_full_state"], "| thinned", out[name]["agreement_thinned_state"])
json.dump(out, open(DATA / "eval_runs_from_ticks.json", "w"), indent=1)
