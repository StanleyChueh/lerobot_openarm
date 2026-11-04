"""Assemble GATE1_METRICS.json from outputs/gate1/{metrics_full,eval_reconciliation,annotation_eval}.json (no recomputation).

    python3 research/contact_aware_augmentation/gate1/make_gate1_metrics.py
"""

import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
O = D / "outputs" / "gate1"
m = json.loads((O / "metrics_full.json").read_text())
rec = json.loads((O / "eval_reconciliation.json").read_text())
ann = json.loads((O / "annotation_eval.json").read_text())
PRE, POST = ("A", "Bp", "B", "C", "E"), ("B_rel", "C_rel", "C_rel_take")
DESC = {"A": "video only (object track plausibility)", "Bp": "generic robot-object proximity (baseline)",
        "B": "robot geometry, phase-free (two-hand rigid mixture + absolute attachment)",
        "C": "phase-aware geometry (giver/receiver rigid fits, absolute attachment, take-over continuity, release, phase logic)",
        "E": "event/time only (motion onset vs close command, phase logic)",
        "B_rel": "POST-HOC: B without absolute attachment", "C_rel": "POST-HOC: C without absolute attachment and take-over continuity",
        "C_rel_take": "POST-HOC: C_rel plus take-over continuity"}


def method(k):
    v = m["nominal"][k]
    return {"description": DESC[k], "preregistered": k in PRE, "false_positives_on_real": v["false_positives"],
            "false_positive_episodes": v["false_positive_episodes"], "auc_all": round(v["auc_all"], 3),
            "detected_raw": v["detected_all"], "detected_beyond_paired_original": v["detected_beyond_original"],
            "per_family": {f: {"detected_beyond_original": d["detected_beyond_original"], "auc": round(d["auc"], 3),
                               **({"geometry_only_beyond_original": d["geometry_only_beyond_original"]}
                                  if "geometry_only_beyond_original" in d else {})} for f, d in v["per_family"].items()},
            "per_variant_beyond_original": v["per_variant_beyond_original"]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


scripts = sorted([p for p in (D / "gate1").rglob("*.py")])
cal = {k: {mode: {"C": v[mode]["C"]["false_positives"], "C_rel": v[mode]["C_rel"]["false_positives"],
                  "C_rel_detected_beyond_original": v[mode]["C_rel"]["detected_beyond_original"]} for mode in v}
       for k, v in m["calibration_sensitivity"].items()}
doc = {
    "schema": "contact_aware_augmentation/gate1_metrics/1",
    "verdict": "PARTIAL",
    "verdict_basis": [
        "pre-registered phase-aware verifier C false-rejects 2/3 real episodes (absolute attachment + take-over continuity)",
        "post-hoc C_rel: 0/3 false rejections; temporal 18/18, cross-episode 6/6, phase 9/9 (geometry only), slip 8/9, "
        "world displacement 10/18; rigid hand-frame displacement 0/18 and constant 20 px offset 0/3 not detectable; "
        "exploratory until confirmed on the sealed episodes",
        "human ground truth absent: robot-keypoint reprojection (8 px) and tracker accuracy criteria PENDING",
    ],
    "generated_from": ["outputs/gate1/metrics_full.json", "outputs/gate1/eval_reconciliation.json", "outputs/gate1/annotation_eval.json"],
    "code_base_commit_at_run": m["repo"],
    "script_sha256": {str(p.relative_to(D)): sha(p) for p in scripts},
    "dataset": m["dataset"],
    "split": {"pilot_development_loeo": m["pilot_episodes"], "sealed_not_used": m["sealed_episodes_not_used"],
              "protocol": m["evaluation"]},
    "ground_truth": m["ground_truth"],
    "human_annotation": {"status": ann["status"], "G1_robot_reprojection_le_8px": ann.get("G1"), "G2_tracker_le_6px": ann.get("G2"),
                         "manifest": "outputs/gate1/annotation/annotation_manifest.csv"},
    "tracker": {e: {k: v[k] for k in ("frames", "ok", "uncertain", "lost", "flags")} for e, v in m["tracker"]["per_episode"].items()},
    "video_to_state_lag_frames": {str(f["test"]): f["lag"] for f in m["folds"]},
    "reference_attachment_mm": {str(f["test"]): f["refC_mm"] for f in m["folds"]},
    "methods": {k: method(k) for k in PRE + POST},
    "artifact_check": m["artifact_check"],
    "m2_counterexamples": m["m2_counterexamples"],
    "projection_reconciliation": m["projection_reconciliation"],
    "calibration_sensitivity_false_positives": cal,
    "evaluation_reconciliation": {k: rec[k] for k in ("models", "matched_unpaired_fisher_a_vs_b_p", "historical_unpaired_fisher_a_vs_b_p", "conclusion")},
    "not_claimed": ["force or contact-force validation", "physical contact onset", "calibration validity of the body_cam extrinsic",
                    "generalisation beyond 3 episodes of one rig and session", "any generated-video verification (no generator was run)"],
}
(D / "GATE1_METRICS.json").write_text(json.dumps(doc, indent=1))
print("wrote", D / "GATE1_METRICS.json")
