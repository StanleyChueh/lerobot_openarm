"""Publish the sfagen (single-pipeline) revision of the four OpenArm datasets. Per repo: tag the current main (stays
loadable), upload dataset + card + method docs + sfagen code/configs + comparison videos + coverage figures, delete
dataset files the new revision no longer has, move `v3.0` to the new commit. DRY_RUN=1 prints the plan only."""
import glob, os, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
S = Path("/home/stanley/Stanley_ws/sfa_hf_export/sfagen"); api = HfApi(); T = dict(repo_type="dataset"); DRY = os.environ.get("DRY_RUN") == "1"
code = [p for p in sorted(S.glob("sfagen/*.py"))] + [p for p in sorted(S.glob("configs/*.yaml"))] + [S / "README.md"]
JOBS = []
for task, cfg in (("pringles", "openarm_pringles_handover"), ("battery", "openarm_battery_insertion")):
    W = S / f"configs/work/{cfg}"
    syn = {"pringles": "ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00", "battery": "ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00"}[task]
    mix = {"pringles": "ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00", "battery": "ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00"}[task]
    figs = sorted(glob.glob(str(S / f"logs/coverage_{task}_*.png")))
    vids = {f"comparison_videos/{task}_unified_highlights.mp4": W / "video_highlights.mp4", f"comparison_videos/{task}_unified_all_2x.mp4": W / "video_all_2x.mp4"}
    prev = {"pringles": ("v04-65ep-two-pipelines", "v00-75ep-two-pipelines"), "battery": ("v01-65ep-two-pipelines", "v00-75ep-two-pipelines")}[task]
    JOBS.append((syn, W / "dataset", W / "README_synthetic.md", prev[0], figs, vids))
    JOBS.append((mix, W / "dataset_mixed", W / "README_mixed.md", prev[1], figs, {}))
for R, root, readme, old_tag, figs, vids in JOBS:
    old = api.dataset_info(R).sha
    new = {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in new.items()]
    ops += [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme)),
            CommitOperationAdd(path_in_repo="PIPELINE.md", path_or_fileobj=str(S / "PIPELINE.md")),
            CommitOperationAdd(path_in_repo="NOVELTY.md", path_or_fileobj=str(S / "NOVELTY.md"))]
    ops += [CommitOperationAdd(path_in_repo=f"sfagen/{p.relative_to(S)}", path_or_fileobj=str(p)) for p in code]
    ops += [CommitOperationAdd(path_in_repo=f"assets/{Path(f).name}", path_or_fileobj=f) for f in figs]
    ops += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in vids.items() if Path(v).exists()]
    stale = [f for f in api.list_repo_files(R, **T) if f.split("/")[0] in ("data", "meta", "videos") and f not in new]
    ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
    print(R, "| main", old[:8], "-> tag", old_tag, "| upload", sum(isinstance(o, CommitOperationAdd) for o in ops), "files | delete stale", stale, flush=True)
    if DRY:
        continue
    api.create_tag(R, tag=old_tag, revision=old, tag_message="previous revision (task-specific pipelines)", exist_ok=False, **T)
    c = api.create_commit(R, operations=ops, parent_commit=old, commit_message="sfagen revision: one simulation-free pipeline for both OpenArm tasks (SAM2-prompted robot masks, photometric time offset, grasp-constrained object fit, contact-exact IK); replay-checked; verified", **T)
    api.delete_tag(R, tag="v3.0", **T); api.create_tag(R, tag="v3.0", revision=c.oid, **T)
    print("   new main", c.oid, flush=True)
