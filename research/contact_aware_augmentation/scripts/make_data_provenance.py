"""Assemble DATA_PROVENANCE.json from outputs/summary.json, outputs/provenance_local.json and outputs/hub_metadata.json.

    python3 research/contact_aware_augmentation/scripts/make_data_provenance.py
"""

import json
from pathlib import Path

D = Path(__file__).resolve().parents[1]
O = D / "outputs"
s = json.loads((O / "summary.json").read_text())
p = json.loads((O / "provenance_local.json").read_text())
h = json.loads((O / "hub_metadata.json").read_text())

REPO_IDS = {"pringles_real_10_v00": "ethanCSL/openarm_pringles_lerobot_real_10_v00",
            "insertbattery_real_v02": "ethanCSL/openarm_insertbattery_lerobot_real_v02",
            "pringles_real_v00_pool": "ethanCSL/openarm_pringles_lerobot_real_v00"}


def ds(label):
    v, hub = p[label], h[REPO_IDS[label]]
    t = s["timing"].get(label, {})
    out = {
        "repo_id": REPO_IDS[label],
        "hub_revision_sha": hub["sha"], "hub_last_modified": hub["last_modified"],
        "hub_refs": {"branches": hub["branches"], "tags": hub["tags"]},
        "local_copy": v["root"],
        "local_vs_hub": v.get("local_vs_hub"),
        "codebase_version": v["codebase_version"], "robot_type": v["robot_type"], "fps_nominal": v["fps"],
        "tasks": v["tasks"],
        "episodes": v["total_episodes_info"], "frames": v["total_frames_info"],
        "episodes_in_metadata": v["episodes_in_metadata"], "frames_in_data": v["frames_in_data"],
        "features": {k: {kk: vv for kk, vv in f.items() if kk != "names"} | ({"names": f["names"]} if k in ("action", "observation.state") else {})
                     for k, f in v["features"].items()},
        "file_sha256": {k: f["sha256"] for k, f in v["files"].items()},
        "file_bytes": {k: f["bytes"] for k, f in v["files"].items()},
    }
    if t:
        out["timing"] = {k: t[k] for k in ("timestamp_minus_frame_index_over_fps_max_abs_s", "frame_index_contiguous_from_0",
                                           "global_index_contiguous", "metadata_length_equals_rows", "episode_length_frames",
                                           "episode_length_s_nominal", "measured_loop_rate")}
        out["timing"]["cameras"] = {k: {kk: c.get(kk) for kk in ("video_span_minus_length_over_fps_max_abs_s", "video_packets_total",
                                                                  "video_packets_equal_frames", "codec", "width_height", "files_local")}
                                    for k, c in t["cameras"].items()}
    if label in s["signals"]:
        out["signals"] = {"columns": s["signals"][label]["columns"], "verdict": s["signals"][label]["verdict"]}
        out["action_semantics"] = {"action_names_equal_state_names": s["action_semantics"][label]["action_names_equal_state_names"],
                                   "action_leads_state_frames": s["action_semantics"][label]["lag"]["k_best"],
                                   "lag_error_rad": s["action_semantics"][label]["lag"]["mean_abs_err_rad"],
                                   "grippers": s["action_semantics"][label]["grippers"]}
    return out


doc = {
    "schema": "contact_aware_augmentation/data_provenance/1",
    "generated_at": s["generated_at"],
    "generated_by": [s["script"], "research/contact_aware_augmentation/scripts/hub_metadata.py",
                     "research/contact_aware_augmentation/scripts/make_data_provenance.py"],
    "hub_queried_at": h["queried_at"],
    "code_repos": s["repos"],
    "urdf": {"path": s["urdf"], "sha256": s["urdf_sha256"]},
    "body_cam_model_used_for_projection": s["body_cam"],
    "datasets": {k: ds(k) for k in REPO_IDS},
    "relations": {
        "real_10_v00_equals_real_v00_pool_episodes_0_9": s.get("real_10_vs_pool", {}).get("all"),
        "insertbattery_other_versions_on_hub": {r: h[r].get("sha") for r in h if "insertbattery" in r and r.endswith(("v00", "v01"))},
        "local_insertbattery_v01_cache_snapshot": "b20b699e7fb2ef70986ab4ca9b24841fe777a23e (older than Hub main 0b3509dd; not used)",
    },
    "not_present_in_any_dataset": ["motor torque / current", "joint velocity", "motor temperature", "force / wrench",
                                   "telemetry sidecar (openarm_telemetry/1)", "depth", "camera timestamps / frame age",
                                   "controller gains or clamped command (q*)", "recording code SHA"],
    "preservation": "datasets were read only; v02 metadata/data and one body_cam video were downloaded into a session "
                    "scratch folder (not the shared HF cache); no file under any dataset root was written; nothing pushed",
}
(D / "DATA_PROVENANCE.json").write_text(json.dumps(doc, indent=1))
print("wrote", D / "DATA_PROVENANCE.json")
