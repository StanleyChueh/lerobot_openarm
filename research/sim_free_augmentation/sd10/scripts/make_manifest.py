"""Source provenance and hash manifest for SD10 -> source_provenance_manifest.json (read-only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sd10 import common as C  # noqa: E402

V4A = C.REPO / "research/sim_free_augmentation/v4a"
SAM = Path.home() / ".cache/huggingface/hub/models--facebook--sam2.1-hiera-small"


def git(*a):
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True).stdout.strip()


def v4a_check():
    r = subprocess.run(["sha256sum", "-c", "--quiet", "locked_manifest.sha256"], cwd=V4A, capture_output=True, text=True)
    return {"locked_manifest_sha256": C.sha256(V4A / "locked_manifest.sha256"), "expected_seal": "91eea62d751c5fd44aa39a452ba875016eb470f31542c02594388b98ed088e17",
            "sha256sum_c_returncode": r.returncode, "all_files_match": r.returncode == 0,
            "note": "read-only check; v4A files are untracked local files and were not modified by SD10"}


def main():
    V = json.loads((C.OUT / "data_verification.json").read_text())
    dup = json.loads((C.OUT / "duplication.json").read_text())
    sel = json.loads((HERE / "selected_10_episodes.json").read_text())
    pool = C.pool()
    eps_meta = {}
    for e in range(50):
        r = pool.ep.loc[e]
        t = V["tabular"]["pool"]["per_episode"][str(e)]
        eps_meta[e] = {"length": t["length"], "dataset_index": [int(r["dataset_from_index"]), int(r["dataset_to_index"])],
                       "state_sha256": t["state_sha256"], "action_sha256": t["action_sha256"],
                       "video_window_s": {c: [float(r[f"videos/{c}/from_timestamp"]), float(r[f"videos/{c}/to_timestamp"])] for c in C.CAMS},
                       "packet_stream_sha256": {c: V["packets"]["pool"][c][str(e)]["packet_stream_sha256"] for c in C.CAMS},
                       "role": ("selected" if e in sel["selected_episodes"] else "") + (" original_ten" if e < 10 else "")
                       + (" v4a_locked" if e in C.V4A_LOCKED else "")}
    files = ["ten_source_selection_protocol.md", "full50_episode_features.csv", "selected_10_episodes.json",
             "FULL50_EPISODE_AUDIT.md", "annotations/manual_transfer_review.json", "annotations/manual_lid_qa.json",
             "outputs/data_verification.json", "outputs/duplication.json", "outputs/features_raw.json",
             "outputs/lid_centers.json", "outputs/selection_details.json", "figures/source_diversity_comparison.png",
             "figures/contact_sheet_full50.jpg", "figures/contact_sheet_original_ten.jpg", "figures/contact_sheet_sd10.jpg"]
    files += sorted(str(p.relative_to(HERE)) for p in (HERE / "scripts").glob("*.py")) + ["sd10/common.py"]
    snap = next((SAM / "snapshots").iterdir())
    man = {"study": "SFA-SD10 (proposed source-diversity curation)",
           "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD"), "base_head_at_manifest": git("rev-parse", "HEAD"),
                   "protocol_freeze_commit": git("log", "-1", "--format=%H", "--", "research/sim_free_augmentation/sd10/ten_source_selection_protocol.md")},
           "datasets": {k: {"repo": v["repo"], "pinned_revision": v["pinned_revision"], "hub_main_sha_at_check": v.get("hub_main_sha_now"),
                            "main_equals_pinned": v.get("main_equals_pinned"),
                            "files": [{x: f.get(x) for x in ("path", "hub_lfs_sha256", "hub_blob_id", "sha256", "bytes", "match", "local")} for f in v["files"]]}
                        for k, v in V["hub"].items()},
           "original_ten_mapping": {"ten_episode_to_pool_episode": dup["mapping_ten_to_pool"],
                                    "state_action": "bit-identical, unique match among 50",
                                    "video": "separately encoded (no identical packets/frames); zero-shift alignment best for all 30 pairs",
                                    "min_ratio_shift1_over_shift0": dup["alignment"]["min_ratio_shift1_over_shift0"]},
           "episodes": eps_meta,
           "annotations_inputs": {"human_clicks": {"path": "research/dataset_diagnostics/annotations/real_v00.csv",
                                                   "sha256": C.sha256(C.REPO / "research/dataset_diagnostics/annotations/real_v00.csv"),
                                                   "last_commit": git("log", "-1", "--format=%H", "--", "research/dataset_diagnostics/annotations/real_v00.csv")}},
           "models": {"sam2.1-hiera-small": {"snapshot": snap.name, "safetensors_sha256": C.sha256(snap / "model.safetensors")},
                      "urdf": {"path": "model/openarm_description.urdf", "sha256": C.sha256(C.REPO / "model/openarm_description.urdf")}},
           "sd10_files_sha256": {f: C.sha256(HERE / f) for f in files if (HERE / f).exists()},
           "v4a_lock_check": v4a_check(),
           "not_committed_regenerable": ["outputs/keyframes/*.npz", "outputs/review/*.jpg"]}
    C.dump(man, HERE / "source_provenance_manifest.json")
    print("v4A lock ok:", man["v4a_lock_check"]["all_files_match"], "files hashed:", len(man["sd10_files_sha256"]))


if __name__ == "__main__":
    main()
