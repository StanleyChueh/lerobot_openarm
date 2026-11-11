"""Which replay checks failed, per candidate (same thresholds as checks.py)."""
from .checks import DEFAULTS


def reasons(ck, cfg_checks, env_min_z):
    c = {**DEFAULTS, **(cfg_checks or {})}; r = []
    if ck.get("action_reference_error_max",0)>2e-6:r.append("action_alignment")
    if not ck.get("finite_arrays",True):r.append("nonfinite")
    if "error" in ck:
        return [ck["error"]]
    if ck["ik_pos_err_mm_max"] > c["ik_pos_mm"] or ck["ik_rot_err_deg_max"] > c["ik_rot_deg"]: r.append("ik")
    if ck.get("ik_pos_err_mm_max_blend", 0) > c["ik_blend_pos_mm"] or ck.get("ik_rot_err_deg_max_blend", 0) > c["ik_blend_rot_deg"]: r.append("ik_blend")
    if ck["limit_violations"]: r.append("limits")
    if ck["speed_ratio_to_real_max"] > c["speed_ratio"]: r.append("speed")
    if ck["accel_ratio_to_real_max"] > c["accel_ratio"]: r.append("accel")
    if ck["min_mesh_distance_mm"] < min(ck["source_min_mesh_distance_same_frames_mm"], 1000 * c["clearance_floor_m"]) - 1: r.append("clearance")
    if ck.get("min_hand_distance_mm", 1e9) < ck.get("source_min_hand_distance_same_frames_mm", 0) - 1000 * c["hand_tol_m"]: r.append("hands")
    if ck["lowest_arm_z_m"] < env_min_z - c["table_tol_m"]: r.append("table")
    if ck["start_state_err_rad"] > c["start_tol_rad"]: r.append("start")
    if not ck["untouched_bit_identical"]: r.append("untouched")
    if not ck["identical_after_rejoin"]: r.append("after_rejoin")
    if ck["object_lowest_point_vs_source_mm"] < -5: r.append("object_low")
    p = ck.get("placement")
    if p and not p["ok"]:
        r += ["place:" + k for k in ("in_view", "on_table") if not p[k]] + (["place:robot_near"] if p["robot_points_near_object_extra_vs_source"] else []) + \
             (["place:gap"] if p["gap_to_other_objects_m"] < c["object_gap_m"] else [])
    return r
