"""SD10-G1 frozen G1-A scorer (committed before the first attempt). Uses generation_v1/paired_stats.py unchanged.

Decision (protocol + metrics, applied mechanically):
  INCOMPLETE  - any primary row missing/duplicated or failed fixed-source-6 invariance (arrays differ >1e-6 or Y differs).
  INCONCLUSIVE- floor/ceiling: max condition yield < 0.10 (no arm reaches ten complete targets, so a ten-point gain
                cannot be separated from an all-fail generator) or min condition yield > 0.90; or neither CONTINUE nor
                STOP holds (interval spans meaningful benefit and no benefit).   [floor/ceiling numbers declared in
                README before launch; the protocol names floor/ceiling but gives no number]
  CONTINUE    - meaningful_yield PASS versus BOTH Original10 and Random10.
  STOP        - valid, non-floor comparison whose paired CI upper bound < 0.10 versus at least one control.
"""
from __future__ import annotations

import importlib.util
import json
from collections import Counter

import numpy as np

import g1lib as G

spec = importlib.util.spec_from_file_location("paired_stats", G.GEN / "paired_stats.py")
PS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PS)
GATES = ["1_arrays_support", "2_initial_candidate", "3_ik_residual", "4_kinematic_limits", "5_clearance_penetration",
         "6_steady_holds", "7_receiver_capture", "8_object_continuity", "9_labels"]


def load():
    rows = [json.loads(l) for l in open(G.OUT / "geometry_attempts.jsonl")]
    return ([r for r in rows if r["kind"] == "primary"], [r for r in rows if r["kind"] == "control_source6"],
            [r for r in rows if r["kind"] == "identity"])


def summ(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None,
            "p95": float(np.percentile(x, 95)) if len(x) else None, "max": float(x.max()) if len(x) else None}


def hull_area(P):
    from scipy.spatial import ConvexHull
    P = np.asarray(P)
    return float(ConvexHull(P).volume) if len(P) >= 3 and np.linalg.matrix_rank(P - P.mean(0)) == 2 else 0.0


def invariance(ctl):
    out, ok = {}, True
    by = {}
    for r in ctl:
        by.setdefault(r["target_id"], {})[r["condition"]] = r
    for tid, d in by.items():
        ys = {c: d[c].get("Y_A") for c in d}
        arrs = {c: np.load(G.STORE / "attempts" / d[c]["arrays_file"]) if d[c].get("arrays_file") else None for c in d}
        dev = 0.0
        ref = arrs[G.CONDS[0]]
        for c in G.CONDS[1:]:
            if ref is None or arrs[c] is None:
                dev = np.inf
                continue
            for k in ("state", "action", "can", "qR_full", "qL_full"):
                dev = max(dev, float(np.abs(ref[k].astype(float) - arrs[c][k].astype(float)).max()))
        same = len(set(ys.values())) == 1 and dev <= 1e-6
        ok &= same
        out[tid] = {"Y_A": ys, "max_array_dev": dev, "identical": same}
    return ok and len(by) == 10 and all(len(d) == 3 for d in by.values()), out


def main():
    G.check_seals()
    prim, ctl, ident = load()
    T = G.TGT["targets"]
    tid = [t["target_id"] for t in T]
    keyed = {(r["target_id"], r["condition"]): r for r in prim}
    complete = len(prim) == 300 and len(keyed) == 300 and all((t, c) in keyed for t in tid for c in G.CONDS)
    inv_ok, inv = invariance(ctl)
    res = {"study": "SD10-G1", "stage": "G1-A geometry", "complete_rows": complete, "control_invariance_ok": inv_ok,
           "control_invariance": inv}
    Y = {c: np.array([int(bool(keyed[(t, c)].get("Y_A"))) for t in tid]) for c in G.CONDS}
    yields = {c: float(Y[c].mean()) for c in G.CONDS}
    nf = np.array([t["near_far"] for t in T])
    rg = np.array([t["region"] for t in T])
    res["yield"] = {c: {"all100": yields[c], "count": int(Y[c].sum()),
                        "near54": float(Y[c][nf == "near"].mean()), "far46": float(Y[c][nf == "far"].mean()),
                        **{f"{r}": float(Y[c][rg == r].mean()) for r in ("central", "intermediate", "edge")},
                        "counts_by_region": {r: int(Y[c][rg == r].sum()) for r in ("central", "intermediate", "edge")},
                        "counts_near_far": {k: int(Y[c][nf == k].sum()) for k in ("near", "far")}} for c in G.CONDS}
    res["paired"] = {"SD10_vs_Original10": PS.meaningful_yield(Y["SD10"], Y["Original10"]),
                     "SD10_vs_Random10": PS.meaningful_yield(Y["SD10"], Y["Random10"]),
                     "Random10_vs_Original10_descriptive": PS.paired_effect(Y["Random10"], Y["Original10"])}
    for g, m in (("far", nf == "far"), ("near", nf == "near")):
        idx = np.flatnonzero(m)
        res["paired"][f"SD10_vs_Original10_{g}_descriptive"] = PS.paired_effect(Y["SD10"][m], Y["Original10"][m], idx)
        res["paired"][f"SD10_vs_Random10_{g}_descriptive"] = PS.paired_effect(Y["SD10"][m], Y["Random10"][m], idx)
    # failure categories (non-exclusive per gate; plus first failing gate and exception/timeout)
    fc = {}
    for c in G.CONDS:
        rs = [keyed[(t, c)] for t in tid]
        cnt = Counter(g for r in rs for g in r.get("failed_gates", []))
        first = Counter((r.get("failed_gates") or ["none"])[0] for r in rs)
        only = Counter(r["failed_gates"][0] for r in rs if len(r.get("failed_gates", [])) == 1)
        fc[c] = {"any_fail_by_gate": {g: cnt.get(g, 0) for g in GATES + ["exception", "timeout_120s"]},
                 "first_failing_gate": dict(first), "sole_failing_gate": dict(only),
                 "n_failed_gates_distribution": dict(Counter(len(r.get("failed_gates", [])) for r in rs))}
    res["failure_categories"] = fc
    # displacement, residuals, relationships, runtime, utilization
    def gv(r, g, k):
        try:
            return r["gates"][g][k]
        except (KeyError, TypeError):
            return None
    det = {}
    for c in G.CONDS:
        rs = [keyed[(t, c)] for t in tid]
        det[c] = {"selection_distance_px": summ([r["selection_distance_px"] for r in rs]),
                  "se3_translation_m": summ([r.get("transform", {}).get("translation_m") for r in rs]),
                  "se3_rotation_deg": summ([r.get("transform", {}).get("rotation_deg") for r in rs]),
                  "ik_max_pos_m": summ([gv(r, "3_ik_residual", "max_pos_m_all_frames") for r in rs]),
                  "ik_max_rot_deg": summ([gv(r, "3_ik_residual", "max_rot_deg_all_frames") for r in rs]),
                  "contact_waypoint_max_mm": summ([gv(r, "3_ik_residual", "contact_waypoint_max_mm") for r in rs]),
                  "ik_fail_rate": float(np.mean(["3_ik_residual" in r.get("failed_gates", []) for r in rs])),
                  "relationship_violation_rate_gates_6_7_8": float(np.mean([any(g in r.get("failed_gates", []) for g in
                                                                               ("6_steady_holds", "7_receiver_capture", "8_object_continuity")) for r in rs])),
                  "table_to_giver_switch_m": summ([(gv(r, "8_object_continuity", "switches") or {}).get("table_to_giver", {}).get("step_m") for r in rs]),
                  "giver_to_receiver_switch_m": summ([(gv(r, "8_object_continuity", "switches") or {}).get("giver_to_receiver", {}).get("step_m") for r in rs]),
                  "can_nonfinger_pen_m": summ([gv(r, "5_clearance_penetration", "can_nonfinger_pen_m") for r in rs]),
                  "receiver_capture_m": summ([gv(r, "7_receiver_capture", "discrepancy_m") for r in rs]),
                  "max_accel": summ([max((gv(r, "4_kinematic_limits", "max_accel_rad_s2") or {"x": None}).values(), default=None) for r in rs]),
                  "unavailable_rows": int(sum(r.get("status") != "complete_plan" for r in rs)),
                  "attempt_wall_s_sum": float(sum(r.get("wall_s", 0) for r in rs)),
                  "attempt_wall_s_median": float(np.median([r.get("wall_s", 0) for r in rs])),
                  "output_frames_sum": int(sum(r.get("source_T", 9) - 9 for r in rs)),
                  "source_utilization": dict(Counter(str(r["source_episode"]) for r in rs)),
                  "per_source_yield": {str(e): [int(sum(bool(r.get("Y_A")) for r in rs if r["source_episode"] == e)),
                                                int(sum(r["source_episode"] == e for r in rs))] for e in G.CFG["conditions"][c]}}
    res["details"] = det
    # coverage of accepted targets on the SAME 100 targets
    xyz = np.array([t["candidate_lid_xyz_robot_base_m"][:2] for t in T])
    rc = np.array([[t["grid_row"], t["grid_col"]] for t in T])
    cov = {"all100_hull_m2": hull_area(xyz)}
    for c in G.CONDS:
        acc = np.flatnonzero(Y[c] == 1)
        if len(acc):
            d = np.min(np.linalg.norm(rc[:, None, :] - rc[acc][None], axis=2), axis=1)
            nearest = {"mean_grid_cells": float(d.mean()), "max_grid_cells": float(d.max())}
        else:
            nearest = {"mean_grid_cells": None, "max_grid_cells": None}
        cov[c] = {"accepted_hull_m2": hull_area(xyz[acc]) if len(acc) >= 3 else 0.0, "nearest_accepted_target": nearest,
                  "accepted_target_ids": [tid[i] for i in acc]}
    res["coverage"] = cov
    # leave-one-assigned-source-out (descriptive; saved results only)
    loo = {}
    for ctrl in ("Original10", "Random10"):
        diffs = {}
        for e in G.CFG["conditions"]["SD10"]:
            m = np.array([keyed[(t, "SD10")]["source_episode"] != e for t in tid])
            if m.sum():
                diffs[str(e)] = float(Y["SD10"][m].mean() - Y[ctrl][m].mean())
        loo[f"SD10_vs_{ctrl}"] = {"by_excluded_SD10_source": diffs,
                                  "range": [min(diffs.values()), max(diffs.values())] if diffs else None}
    res["leave_one_assigned_source_out"] = loo
    # pace / recording-quality association (descriptive)
    pace = {}
    for c in G.CONDS:
        rs = [keyed[(t, c)] for t in tid]
        dur = np.array([G.CFG["source_metadata"][str(r["source_episode"])]["duration_s"] for r in rs])
        g2r = np.array([G.CFG["source_metadata"][str(r["source_episode"])]["grasp_to_receive_s"] for r in rs])
        pace[c] = {"mean_assigned_source_duration_s": float(dur.mean()), "mean_assigned_grasp_to_receive_s": float(g2r.mean())}
    res["pace"] = pace
    res["identity_diagnostics"] = {str(r["source_episode"]): {"Y_A_diagnostic": r.get("Y_A_diagnostic"), "failed_gates": r.get("failed_gates"),
                                                              "table_to_giver_switch_m": (gv(r, "8_object_continuity", "switches") or {}).get("table_to_giver", {}).get("step_m"),
                                                              "can_nonfinger_pen_m": gv(r, "5_clearance_penetration", "can_nonfinger_pen_m"),
                                                              "axis_tilt_deg": gv(r, "2_initial_candidate", "axis_err_deg")} for r in ident}
    # decision
    p1, p2 = res["paired"]["SD10_vs_Original10"], res["paired"]["SD10_vs_Random10"]
    floor = max(yields.values()) < 0.10
    ceiling = min(yields.values()) > 0.90
    if not complete or not inv_ok:
        dec = "INCOMPLETE"
    elif floor or ceiling:
        dec = "INCONCLUSIVE (geometry " + ("floor" if floor else "ceiling") + ")"
    elif p1["PASS"] and p2["PASS"]:
        dec = "CONTINUE SD10 (geometric source utility only; H2 separate)"
    elif p1["paired_ci95"][1] < 0.10 or p2["paired_ci95"][1] < 0.10:
        dec = "STOP SD10 generation-utility branch"
    else:
        dec = "INCONCLUSIVE (interval spans meaningful benefit and no benefit)"
    res["floor"], res["ceiling"], res["G1A_decision"] = floor, ceiling, dec
    res["rgb_stage_justified"] = dec.startswith("CONTINUE")
    (G.OUT / "results.json").write_text(json.dumps(G.jsonable(res), indent=1))
    print(json.dumps(G.jsonable({"decision": dec, "yield": {c: res["yield"][c]["all100"] for c in G.CONDS},
                                 "SD10_vs_O10": p1, "SD10_vs_R10": p2, "invariance": inv_ok, "complete": complete}), indent=1))


if __name__ == "__main__":
    main()
