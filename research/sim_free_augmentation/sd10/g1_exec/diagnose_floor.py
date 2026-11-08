"""Descriptive decomposition of the G1-A floor from saved records only (no new generation, no threshold changes).
Writes outputs/floor_diagnostics.json."""
from __future__ import annotations

import json

import numpy as np

import g1lib as G


def main():
    rows = [json.loads(l) for l in open(G.OUT / "geometry_attempts.jsonl")]
    prim = [r for r in rows if r["kind"] == "primary" and r.get("status") == "complete_plan"]
    ident = [r for r in rows if r["kind"] == "identity" and r.get("status") != "source_preprocessing_failed"]
    out = {"identity": {"n_prepared_sources": len(ident),
                        "gate_pass_counts": {k: int(sum(r["gates"][k]["ok"] for r in ident)) for k in ident[0]["gates"]},
                        "table_to_giver_switch_mm": {str(r["source_episode"]): 1000 * r["gates"]["8_object_continuity"]["switches"]["table_to_giver"]["step_m"] for r in ident},
                        "receiver_capture_mm": {str(r["source_episode"]): 1000 * r["gates"]["7_receiver_capture"].get("discrepancy_m", np.nan) for r in ident},
                        "can_nonfinger_pen_mm": {str(r["source_episode"]): 1000 * r["gates"]["5_clearance_penetration"]["can_nonfinger_pen_m"] for r in ident},
                        "source_axis_tilt_deg": {str(r["source_episode"]): r["gates"]["2_initial_candidate"]["axis_err_deg"] for r in ident},
                        "source_lid_z_m": {str(r["source_episode"]): G.load_light(r["source_episode"]).table[2, 3] for r in ident}},
           "retargeted": {}}
    for c in G.CONDS:
        rs = [r for r in prim if r["condition"] == c]
        g = np.array([([w for w in r["gates"]["3_ik_residual"]["waypoints"] if w["side"] == "R" and w["kind"] == "contact"][0]["requested_mm"],
                       [w for w in r["gates"]["3_ik_residual"]["waypoints"] if w["side"] == "R" and w["kind"] == "contact"][0]["err_mm"],
                       1000 * r["gates"]["8_object_continuity"]["switches"]["table_to_giver"]["step_m"],
                       1000 * r["transform"]["translation_m"]) for r in rs])
        out["retargeted"][c] = {"n_complete_plans": len(rs), "grasp_requested_mm_median": float(np.median(g[:, 0])),
                                "grasp_ik_err_mm_median": float(np.median(g[:, 1])), "grasp_ik_err_mm_min": float(g[:, 1].min()),
                                "attempts_grasp_within_2mm": int((g[:, 1] <= 2).sum()),
                                "corr_requested_vs_err": float(np.corrcoef(g[:, 0], g[:, 1])[0, 1]),
                                "table_to_giver_switch_mm_min": float(g[:, 2].min()), "se3_translation_mm_median": float(np.median(g[:, 3]))}
    z = np.array(list(out["identity"]["source_lid_z_m"].values()))
    out["height_prior"] = {"registered_candidate_lid_z_m": G.TGT["candidate_lid_height_m"], "source_lid_z_m_min": float(z.min()),
                           "source_lid_z_m_median": float(np.median(z)), "source_lid_z_m_max": float(z.max())}
    (G.OUT / "floor_diagnostics.json").write_text(json.dumps(G.jsonable(out), indent=1))
    print(json.dumps(G.jsonable({"height_prior": out["height_prior"], "identity_gate_pass": out["identity"]["gate_pass_counts"],
                                 "retargeted": out["retargeted"]}), indent=1))


if __name__ == "__main__":
    main()
