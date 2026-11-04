"""Reconcile the historical 49 vs 47 / 100 evaluation with the matched 53 vs 34 / 100 evaluation (read-only, CPU).

    python3 research/contact_aware_augmentation/gate1/reconcile_eval.py --out research/contact_aware_augmentation/outputs/gate1/eval_reconciliation.json

Inputs are the committed records only (no re-run of any policy): research/generated_data_diagnosis/
matched_evaluation/comparison.json (matched per-pose outcomes, historical region counts), EVALUATION_COMPARISON.md
and MODEL_COMPARISON.md (protocol details, repeat-run estimates, latency).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import chi2, fisher_exact

REPO = Path(__file__).resolve().parents[3]
CMP = REPO / "research/generated_data_diagnosis/matched_evaluation/comparison.json"
REGIONS = ("x<0.203", "0.203-0.27", "x>=0.27")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def cmh(tables):
    """Mantel-Haenszel chi-square (1 df, with continuity correction) and common odds ratio over 2x2 strata
    [[a, b], [c, d]] = [[hist success, hist fail], [matched success, matched fail]]."""
    num = den = var = 0.0
    orn = ord_ = 0.0
    for (a, b), (c, d) in tables:
        n = a + b + c + d
        if n < 2:
            continue
        e = (a + b) * (a + c) / n
        v = (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
        num += a - e
        var += v
        orn += a * d / n
        ord_ += b * c / n
    stat = (abs(num) - 0.5) ** 2 / var if var > 0 else float("nan")
    return {"chi2": round(stat, 3), "p": round(float(chi2.sf(stat, 1)), 4), "common_odds_ratio_hist_vs_matched": round(orn / ord_, 3) if ord_ else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    c = json.loads(CMP.read_text())
    hist, reg = c["historical"], c["regions"]
    out = {"sources": [str(CMP.relative_to(REPO)), "research/generated_data_diagnosis/EVALUATION_COMPARISON.md",
                       "research/generated_data_diagnosis/matched_evaluation/MODEL_COMPARISON.md"],
           "checkpoints": {"manual-10": "ethanCSL/openarm_pringles_lerobot_sim_smolvla_10_v00 @ 9ce7ce48 (matched run_meta); historical run did not record a revision",
                           "generated-500": "ethanCSL/openarm_pringles_lerobot_sim_10_generated_500_smolvla_v00 @ 733c6856 (matched run_meta); historical run did not record a revision"},
           "models": {}}
    n_matched = {r: reg[r]["success_any"]["n"] for r in REGIONS}
    for key, name in (("a", "manual-10"), ("b", "generated-500")):
        hs = {r: hist[r][key] for r in REGIONS}               # [successes, n] on ESTIMATED historical starts
        ms = {r: [reg[r]["success_any"][key], reg[r]["success_any"]["n"]] for r in REGIONS}
        H = (hist["success_any"][key], hist["success_any"]["n"])
        Mt = (c["metrics"]["success_any"][key] if "success_any" in c["metrics"] else sum(v[0] for v in ms.values()), 100)
        # historical per-region rates re-weighted to the matched pose mix (what the old protocol "would" score on the
        # matched poses if only the pose mix differed); variance by the delta method
        exp = sum(hs[r][0] / hs[r][1] * n_matched[r] for r in REGIONS)
        var = sum((hs[r][0] / hs[r][1]) * (1 - hs[r][0] / hs[r][1]) / hs[r][1] * n_matched[r] ** 2 for r in REGIONS)
        tables = [((hs[r][0], hs[r][1] - hs[r][0]), (ms[r][0], ms[r][1] - ms[r][0])) for r in REGIONS]
        out["models"][name] = {
            "historical": {"success": f"{H[0]}/{H[1]}", "wilson95": wilson(*H), "by_region": {r: f"{v[0]}/{v[1]}" for r, v in hs.items()}},
            "matched": {"success": f"{Mt[0]}/{Mt[1]}", "wilson95": wilson(*Mt), "by_region": {r: f"{v[0]}/{v[1]}" for r, v in ms.items()}},
            "unpaired_fisher_hist_vs_matched_p": round(float(fisher_exact([[H[0], H[1] - H[0]], [Mt[0], Mt[1] - Mt[0]]])[1]), 4),
            "historical_standardized_to_matched_pose_mix": {"expected": round(exp, 1), "sd": round(math.sqrt(var), 1),
                                                            "observed_matched": Mt[0], "z": round((Mt[0] - exp) / math.sqrt(var + Mt[0] * (1 - Mt[0] / 100)), 2)},
            "per_region_fisher_p": {r: round(float(fisher_exact([[hs[r][0], hs[r][1] - hs[r][0]], [ms[r][0], ms[r][1] - ms[r][0]]])[1]), 4) for r in REGIONS},
            "stratified_mantel_haenszel_hist_vs_matched": cmh(tables),
        }
    a, b = c["metrics"]["success_any"]["a"], c["metrics"]["success_any"]["b"]
    out["matched_unpaired_fisher_a_vs_b_p"] = round(float(fisher_exact([[a, 100 - a], [b, 100 - b]])[1]), 4)
    out["historical_unpaired_fisher_a_vs_b_p"] = round(float(fisher_exact([[49, 51], [47, 53]])[1]), 4)
    out["protocol_differences"] = [
        {"item": "can poses", "historical": "random truncated Gaussian, unseeded, not logged (estimated afterwards from body_cam, LOO 0.6 mm in sim)", "matched": "fixed 100-pose file (sha256 2b95ecf0...), identical order for both models, placed at the task rest height and verified <= 1.66 mm"},
        {"item": "pairing", "historical": "unpaired (different poses per model)", "matched": "paired by pose (McNemar)"},
        {"item": "horizon", "historical": "15 s wall clock = 270-321 sim steps, fewer in successful episodes", "matched": "exactly 330 sim steps"},
        {"item": "success window", "historical": "OR over policy steps AND the unrecorded return home", "matched": "policy steps only (return contamination measured: 0/200)"},
        {"item": "inference latency / RTC", "historical": "mean 209 ms, delay 1.77 steps", "matched": "mean 134 ms (A) / 130 ms (B)"},
        {"item": "viewer", "historical": "rerun display on", "matched": "display off"},
        {"item": "stage attribution", "historical": "gripper-only detector: both fail at the pass (51/51, 51/53)", "matched": "simulator ground truth: main failure is right grasp/lift (can never > 2 cm: 32 vs 50); pass failures 5 vs 11"},
        {"item": "repeat runs (historical protocol, tick-log estimates)", "historical": "manual 52/100 (vs 49); generated 33/73 = 45 % (vs 47); generated_400 61/100", "matched": "one run per checkpoint"},
    ]
    m = out["models"]
    out["conclusion"] = {
        "manual-10": (f"49 -> 53 is consistent with noise and pose mix (standardized expectation "
                      f"{m['manual-10']['historical_standardized_to_matched_pose_mix']['expected']} +- "
                      f"{m['manual-10']['historical_standardized_to_matched_pose_mix']['sd']}; Fisher p "
                      f"{m['manual-10']['unpaired_fisher_hist_vs_matched_p']})."),
        "generated-500": (f"47 -> 34 is NOT explained by the pose mix (standardized expectation "
                          f"{m['generated-500']['historical_standardized_to_matched_pose_mix']['expected']} +- "
                          f"{m['generated-500']['historical_standardized_to_matched_pose_mix']['sd']}, stratified MH p "
                          f"{m['generated-500']['stratified_mantel_haenszel_hist_vs_matched']['p']}). The drop is spread over the "
                          "near-base and middle bands. Remaining candidate causes, none isolated: single unlogged historical run "
                          "(its own repeat estimate was 45 %), inference-latency/RTC timing change, exact placement at the rest "
                          "height, horizon and success-window definition. UNRESOLVED; it would need repeated runs of the same "
                          "checkpoint under both protocols."),
        "direction": ("Both protocols agree that the 500 generated demos do not improve on 10 manual demos (tie vs worse) "
                      "and that generated-data failures concentrate where generation never started (x < 0.203). They "
                      "DISAGREE on the failure stage: the historical gripper-only detector put it at the pass, the matched "
                      "simulator ground truth puts it at the right-hand grasp/lift (the historical attribution was a detector "
                      "artifact, MODEL_COMPARISON §3). The historical tie and the 'pass is the bottleneck' motivation cited in "
                      "Gate 0 are superseded by the matched result. Caveat: historical regions use estimated starts, and one "
                      "checkpoint per arm limits both results."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
