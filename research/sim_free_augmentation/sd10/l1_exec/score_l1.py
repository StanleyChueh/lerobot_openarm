"""SD10-L1 frozen scorer and decision hierarchy (SD10_LEARNING_PROTOCOL.md section 7). Uses the reference metric /
CI definitions in learning_v1/protocol_tools.py unchanged. Writes outputs/results.json, outputs/per_record_scores.csv,
outputs/per_joint_errors.json and outputs/tables.md. Fails closed: any completeness defect -> INCOMPLETE.
"""
from __future__ import annotations

import csv
import json

import numpy as np

import l1lib as L

PT = L.PT
PH = PT.PHASES
RUNS = [(a, s) for a in L.ARMS for s in L.SEEDS]
TH = L.CFG["statistics"]


def completeness(z):
    """Criterion 1. Returns (ok, report, predictions)."""
    rep, P = {}, {}
    for a, s in RUNS:
        r = f"{a}_s{s}"
        try:
            term = json.loads((L.TRAIN / r / "terminal.json").read_text())
            pre = json.loads((L.TRAIN / r / "prelaunch.json").read_text())
            meta = json.loads((L.PRED / f"{r}.json").read_text())
            f = np.load(L.PRED / f"{r}.npz", allow_pickle=False)
            p = f["prediction"]
            ok = {"updates_1000": term["consumed_optimizer_updates"] == PT.UPDATES,
                  "adam_steps_1000": term["adam_step_counters"] == [PT.UPDATES],
                  "schedule_consumed_equals_frozen": term["schedule_state"]["consumed_equals_frozen"],
                  "no_discarded_batches": term["schedule_state"]["discarded_prefetched_batches"] == 0,
                  "source_draws_1600_each": all(v == 1600 for v in term["source_draws_consumed"].values())
                  and sorted(map(int, term["source_draws_consumed"])) == L.ARMS[a],
                  "model_hash_matches": L.sha(L.TRAIN / r / "pretrained_model/model.safetensors") == term["model_sha256"] == meta["model_sha256"],
                  "normalizers_match_frozen": pre["normalizer_tensor_sha256"] == meta["normalizer_tensor_sha256"],
                  "seal_at_launch": pre["seal_sha256"] == L.SEAL_SHA,
                  "items_exact": np.array_equal(f["items"], z["items"]),
                  "float32": p.dtype == np.float32, "shape": p.shape == (834, 50, 16), "finite": bool(np.isfinite(p).all()),
                  "prediction_hash": PT.array_sha(p) == meta["prediction_sha256"], "nonfinite_loss_absent": term["nonfinite_loss"] is False}
            P[(a, s)] = p
        except Exception as ex:  # noqa: BLE001
            ok = {"error": repr(ex)}
        rep[r] = ok
    return all(all(v is True for v in d.values()) for d in rep.values()), rep, P


def per_joint(pred, items, targets, phase_ids, lengths):
    """Channel-resolved version of PT.recording_scores (same target-time averaging of absolute errors)."""
    err = np.abs(pred.astype(np.float64) - targets)                     # [N,50,16]
    err[..., 14] /= 1.298
    err[..., 15] /= 1.2121
    out = {}
    for ep in PT.EVAL:
        n = lengths[ep]
        sums, counts, ph = np.zeros((n, 16)), np.zeros(n), np.full(n, -1)
        for i in np.flatnonzero(items[:, 0] == ep):
            t = int(items[i, 1]); rows = np.arange(t, t + 50)
            sums[rows] += err[i]; counts[rows] += 1; ph[rows] = phase_ids[i]
        v = sums / counts[:, None]
        out[ep] = {p: v[ph == j].mean(0) for j, p in enumerate(PH)} | {"overall_time_weighted": v.mean(0)}
    return out


def boot_mean(x, idx):
    """x: [3, R] -> seed-averaged recording mean and recording-bootstrap 95% CI."""
    xx = x.mean(0)
    return {"mean": float(x.mean()), "ci95": np.quantile(xx[idx].mean(1), [.025, .975]).tolist()}


def boot_diff(x, ref, idx):
    d = x.mean(0) - ref.mean(0)
    return {"abs_diff": float(x.mean() - ref.mean()), "ci95": np.quantile(d[idx].mean(1), [.025, .975]).tolist(),
            "per_seed": (x.mean(1) - ref.mean(1)).tolist()}


def compare(S, arm, ref, groups, bi, sd):
    """All registered contrast statistics for arm vs ref."""
    def X(a, metric, g):
        return np.array([[S[(a, s)][e][metric] for e in groups[g]] for s in L.SEEDS])
    c = {}
    for name, metric, g in [("far_grasp", "grasp", "far"), ("near_grasp", "grasp", "near"),
                            ("handover_all21", "handover", "all21"), ("phase_balanced_all21", "phase_balanced", "all21"),
                            ("overall_time_weighted_all21", "overall_time_weighted", "all21")] + \
                           [(f"{p}_all21", p, "all21") for p in PH if p not in ("handover",)] + \
                           [("far_handover", "handover", "far"), ("near_handover", "handover", "near"),
                            ("far_overall", "overall_time_weighted", "far"), ("near_overall", "overall_time_weighted", "near")]:
        x, r = X(arm, metric, g), X(ref, metric, g)
        c[name] = {"arm_error": boot_mean(x, bi[g + "_indices"]), "ref_error": boot_mean(r, bi[g + "_indices"]),
                   "abs_diff": boot_diff(x, r, bi[g + "_indices"]),
                   "relative": PT.paired_relative(x, r, bi[g + "_indices"]),
                   "relative_crossed": PT.paired_relative(x, r, bi[g + "_indices"], sd)}
    for side, j in (("right", 0), ("left", 1)):
        x = np.array([[S[(arm, s)][e]["handover_gripper_openness_mae"][j] for e in groups["all21"]] for s in L.SEEDS])
        r = np.array([[S[(ref, s)][e]["handover_gripper_openness_mae"][j] for e in groups["all21"]] for s in L.SEEDS])
        c[f"handover_gripper_{side}_openness_mae"] = {"arm": boot_mean(x, bi["all21_indices"]), "ref": boot_mean(r, bi["all21_indices"]),
                                                      "increase": boot_diff(x, r, bi["all21_indices"])}
    return c


def screen(c, min_rel):
    p = c["far_grasp"]["relative"]
    crit = {"far_grasp_relative_ge_min": p["relative_improvement"] >= min_rel,
            "far_grasp_ci_lower_gt_0": p["ci95"][0] > 0,
            "far_grasp_all_seeds_nonnegative": all(v >= 0 for v in p["per_seed"]),
            "handover_harm_le_2pct": -c["handover_all21"]["relative"]["relative_improvement"] <= TH["max_handover_relative_harm"],
            "phase_balanced_harm_le_2pct": -c["phase_balanced_all21"]["relative"]["relative_improvement"] <= TH["max_phase_balanced_relative_harm"]}
    grip_flag = {s: c[f"handover_gripper_{s}_openness_mae"]["increase"]["abs_diff"] > 0.02 for s in ("right", "left")}
    passed = all(crit.values())
    verdict = ("PASS" if passed and not any(grip_flag.values()) else "MODIFY (gripper handover flag)" if passed else "FAIL")
    lo, hi = p["ci95"]
    interp = ("intended practical effect (>=%.0f%%) unsupported: CI upper bound below threshold" % (100 * min_rel) if hi < min_rel
              else "inconclusive: CI spans no benefit and material benefit" if lo <= 0 < hi or (lo <= 0 and hi >= min_rel)
              else "CI excludes zero")
    ho = c["handover_all21"]["relative"]
    util = {"handover_harm_point": -ho["relative_improvement"], "handover_harm_ci95_upper": -ho["ci95"][0],
            "utility_confidently_no_more_than_2pct_harm": -ho["ci95"][0] <= 0.02,
            "label": "utility confidently within 2% harm" if -ho["ci95"][0] <= 0.02 else "utility uncertain (handover CI allows >2% harm)",
            "robust_to_seed_uncertainty_crossed_ci_lower_gt_0": c["far_grasp"]["relative_crossed"]["ci95"][0] > 0}
    return {"criteria": crit, "gripper_modify_flags": grip_flag, "verdict": verdict, "primary_interval_reading": interp, "utility": util}


def main():
    L.check_seal()
    z = np.load(L.L1 / "evaluation_arrays.npz", allow_pickle=False)
    man = json.loads((L.L1 / "locked_evaluation_manifest.json").read_text())
    bi = np.load(L.L1 / "bootstrap_indices.npz", allow_pickle=False)
    groups = man["recording_groups"]
    for g in groups:
        assert np.array_equal(bi[g + "_episodes"], groups[g])
    lengths = {int(e): r["length"] for e, r in man["recordings"].items()}
    ok, comp, P = completeness(z)
    res = {"study": "SD10-L1", "seal_sha256": L.SEAL_SHA, "completeness": comp, "complete": ok}
    if not ok:
        res["decision"] = "INCOMPLETE"
        (L.OUT / "results.json").write_text(json.dumps(res, indent=1))
        print("INCOMPLETE", json.dumps(comp, indent=1))
        return
    S = {k: PT.recording_scores(p, z["items"], z["targets"], z["target_phase_ids"], lengths) for k, p in P.items()}
    J = {k: per_joint(p, z["items"], z["targets"], z["target_phase_ids"], lengths) for k, p in P.items()}
    for k in S:   # channel-resolved scorer must reproduce the reference arm metric
        for e in PT.EVAL:
            for p in PH:
                assert abs(J[k][e][p][:14].mean() - S[k][e][p]) < 1e-9
    sd = bi["seed_indices"]
    absolute = {}
    for a in L.ARMS:
        absolute[a] = {}
        for g in groups:
            for m in PH + ["overall_time_weighted", "phase_balanced"]:
                x = np.array([[S[(a, s)][e][m] for e in groups[g]] for s in L.SEEDS])
                absolute[a][f"{g}/{m}"] = boot_mean(x, bi[g + "_indices"]) | {"per_seed": x.mean(1).tolist()}
            for j, side in enumerate(("right", "left")):
                x = np.array([[S[(a, s)][e]["handover_gripper_openness_mae"][j] for e in groups[g]] for s in L.SEEDS])
                absolute[a][f"{g}/handover_gripper_{side}"] = boot_mean(x, bi[g + "_indices"]) | {"per_seed": x.mean(1).tolist()}
    C = {"SD10_vs_Original10": compare(S, "SD10", "Original10", groups, bi, sd),
         "SD10_vs_Random10": compare(S, "SD10", "Random10", groups, bi, sd),
         "Random10_vs_Original10": compare(S, "Random10", "Original10", groups, bi, sd)}
    primary = screen(C["SD10_vs_Original10"], TH["primary_min_relative_improvement"])
    secondary = (screen(C["SD10_vs_Random10"], TH["random_min_relative_improvement"]) if primary["verdict"] == "PASS" else
                 {"tested": False, "reason": "hierarchical: primary screen did not pass",
                  "descriptive_far_grasp_relative": C["SD10_vs_Random10"]["far_grasp"]["relative"],
                  "statement": "advantage over this random draw not established"})
    if primary["verdict"] == "PASS" and secondary.get("verdict") == "PASS":
        rec = "SD10 improves over both controls in this screen -> propose a separate source-set x augmentation experiment"
    elif primary["verdict"] == "PASS":
        rec = "SD10 beats Original10 but advantage over the fixed Random10 not established -> do not attribute to curation; no source-set x augmentation launch yet"
    elif primary["verdict"].startswith("MODIFY"):
        rec = "MODIFY: arm-error screen passed but gripper handover guard flagged"
    else:
        rec = "STOP SD10 and RETURN TO v4A (no meaningful SD10 learning advantage under the registered screen)"
    res.update(absolute_errors=absolute, contrasts=C, primary_screen=primary, secondary_random_screen=secondary,
               decision=primary["verdict"], recommendation=rec,
               per_record={f"{a}_s{s}": {str(e): S[(a, s)][e] for e in PT.EVAL} for a, s in RUNS})
    (L.OUT / "results.json").write_text(json.dumps(res, indent=1))
    (L.OUT / "per_joint_errors.json").write_text(json.dumps(
        {f"{a}_s{s}": {str(e): {m: v.tolist() for m, v in J[(a, s)][e].items()} for e in PT.EVAL} for a, s in RUNS}))
    with open(L.OUT / "per_record_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["arm", "seed", "episode", "bin", *PH, "overall_time_weighted", "phase_balanced", "handover_grip_R", "handover_grip_L"])
        for a, s in RUNS:
            for e in PT.EVAL:
                r = S[(a, s)][e]
                w.writerow([a, s, e, "far" if e in groups["far"] else "near", *[f"{r[p]:.6g}" for p in PH],
                            f"{r['overall_time_weighted']:.6g}", f"{r['phase_balanced']:.6g}",
                            *[f"{v:.6g}" for v in r["handover_gripper_openness_mae"]]])
    print(json.dumps({"decision": res["decision"], "recommendation": rec, "primary": primary,
                      "far_grasp_SD10_vs_O10": C["SD10_vs_Original10"]["far_grasp"]["relative"]}, indent=1))


if __name__ == "__main__":
    main()
