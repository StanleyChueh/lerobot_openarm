"""Assemble full50_episode_features.csv: gates Q1-Q6, position/interaction features, quality score q.

Inputs: outputs/data_verification.json, outputs/features_raw.json, outputs/lid_centers.json,
        annotations/manual_transfer_review.json, annotations/manual_lid_qa.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sd10 import common as C  # noqa: E402


def load(p):
    return json.loads(Path(p).read_text())


def rank01(x):
    x = np.asarray(x, float)
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return r / max(len(x) - 1, 1)


def main():
    V = load(C.OUT / "data_verification.json")
    F = {int(k): v for k, v in load(C.OUT / "features_raw.json").items()}
    L = {int(k): v for k, v in load(C.OUT / "lid_centers.json").items()}
    T5 = {int(k): v for k, v in load(HERE / "annotations/manual_transfer_review.json")["episodes"].items()}
    Q6 = {int(k): v for k, v in load(HERE / "annotations/manual_lid_qa.json")["episodes"].items()}
    rows = []
    for e in range(50):
        f, lid = F[e], L[e]
        locked = e in C.V4A_LOCKED
        tab = V["tabular"]["pool"]["per_episode"][str(e)]
        pk = {c: V["packets"]["pool"][c][str(e)] for c in C.CAMS}
        px = {c: V["pixels"]["pool"][c][str(e)] for c in C.CAMS}
        frozen = sum(px[c]["frozen_consecutive"] for c in C.CAMS if px[c].get("decoded") is not None)
        longest = max(px[c]["longest_frozen_run"] for c in C.CAMS if px[c].get("decoded") is not None)
        dec_ok = all(px[c].get("decoded") == tab["length"] for c in C.CAMS if not (locked and c != C.BODY))
        bw = sum(px[c]["black_frames"] + px[c]["white_frames"] for c in C.CAMS if px[c].get("decoded") is not None)
        q1 = bool(tab["ok"] and all(pk[c]["count_ok"] and pk[c]["pts_regular"] for c in C.CAMS) and dec_ok and bw == 0
                  and longest <= 3)
        u, v = f["click_uv"]
        mask_touch = f["sam_choice"] is not None and f["sam_candidates"][f["sam_choice"]]["contains_click"]
        q6 = bool(lid is not None and lid["lid_fill_ratio"] >= 0.80 and Q6[e]["accept"] and mask_touch)
        r = {"episode": e, "in_original_ten": e < 10, "v4a_locked": locked, "length": f["T"],
             "Q1_integrity": q1, "Q2_command_structure": f["q2_command_structure"],
             "Q3_measured_right_grasp": f["q3_measured_right_grasp"], "Q4_measured_left_hold": f["q4_measured_left_hold"],
             "Q5_visual_transfer": T5[e]["label"], "Q5_evidence_scope": T5[e]["confidence"], "Q6_lid_localised": q6,
             "lid_u": lid["lid_u"], "lid_v": lid["lid_v"], "lid_major_px": max(lid["lid_axes_px"]),
             "lid_minor_px": min(lid["lid_axes_px"]), "lid_fill_ratio": lid["lid_fill_ratio"],
             "click_u": u, "click_v": v, "click_to_lid_px": float(np.hypot(u - lid["lid_u"], v - lid["lid_v"])),
             "can_mask_area_px": f["can_area_px"], "can_mask_centroid_u": f["can_u"], "can_mask_centroid_v": f["can_v"],
             "rc": f["rc"], "lc": f["lc"], "ro": f["ro"], "handover_dt_s": f["handover_dt_s"],
             "cmd_overlap_frames": f["cmd_overlap_frames"], "meas_dual_hold_frames": f.get("meas_dual_hold_frames"),
             "meas_right_grasp_open": f["meas_right_grasp_median"], "meas_left_hold_open": f["meas_left_hold_median"],
             "meas_left_hold_drift": f["meas_left_hold_drift_p95_p5"]}
        for k, nm in (("right_tcp_rc_xyz", "right_grasp"), ("left_tcp_lc_xyz", "left_receive"), ("right_tcp_lc_xyz", "right_at_lc")):
            for i, ax in enumerate("xyz"):
                r[f"{nm}_{ax}_m"] = f[k][i]
        r.update(right_grasp_yaw_deg=f["right_tcp_rc_yaw_deg"], left_receive_yaw_deg=f["left_tcp_lc_yaw_deg"],
                 tcp_separation_lc_m=f["tcp_separation_lc_m"], can_crop_sharpness=f["can_crop_sharpness"],
                 right_wrist_sharpness_rc=f.get("right_wrist_sharpness_rc"),
                 left_wrist_sharpness_lc=f.get("left_wrist_sharpness_lc"), body_sharpness_median=f["body_sharpness_median"],
                 frozen_frames_total=frozen, longest_frozen_run=longest, track_visible_frac=f["track_visible_frac"],
                 track_displacement_last_px=f["track_displacement_last_px"],
                 notes="; ".join(T5[e]["notes"] + Q6[e]["notes"]))
        r["passes_quality"] = bool(q1 and r["Q2_command_structure"] and r["Q3_measured_right_grasp"]
                                   and r["Q4_measured_left_hold"] and r["Q5_visual_transfer"] == "SUCCESS" and q6)
        r["eligible"] = bool(r["passes_quality"] and not locked)
        r["exclusion_reason"] = ("v4A locked evaluator record" if locked else "") + \
            ("" if r["passes_quality"] else (" quality gate failed" if not locked else "; quality gate failed"))
        rows.append(r)
    el = [r for r in rows if r["eligible"]]
    q = np.mean([rank01([r["can_crop_sharpness"] for r in el]), rank01([r["right_wrist_sharpness_rc"] for r in el]),
                 rank01([r["left_wrist_sharpness_lc"] for r in el]), rank01([-r["frozen_frames_total"] for r in el])], 0)
    for r, qq in zip(el, q):
        r["quality_score_q"] = float(qq)
    for r in rows:
        r.setdefault("quality_score_q", None)
    out = HERE / "full50_episode_features.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()})
    print("rows", len(rows), "passes_quality", sum(r["passes_quality"] for r in rows), "eligible", len(el))
    print("failing gates:", [(r["episode"], [k for k in r if k.startswith("Q") and r[k] is False]) for r in rows if not r["passes_quality"]])


if __name__ == "__main__":
    main()
