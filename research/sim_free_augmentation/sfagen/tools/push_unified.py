"""Publish the sfagen (single-pipeline) revision of the four OpenArm datasets. Per repo: tag the current main (stays
loadable), upload dataset + card + method docs + sfagen code/configs + provenance.json + comparison videos + coverage
figures, delete dataset files the new revision no longer has, move `v3.0` to the new commit. DRY_RUN=1 prints the plan only.
Every export is checked before the first write to any repo; an empty or incomplete export never reaches the Hub."""
import glob, json, os, sys, tempfile
from pathlib import Path
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from _provenance import HERE as S, upload_provenance
T = dict(repo_type="dataset")
MESSAGE = ("sfagen revision: one simulation-free pipeline for both OpenArm tasks (SAM2-prompted robot masks, photometric time offset, "
           "grasp-constrained object fit, contact-exact IK); replay-checked; verified")


def export_files(root):
    """{path_in_repo: local file} of a LeRobot export; refuses an empty or incomplete one (it would delete the Hub data)."""
    new = {str(p.relative_to(root)): p for p in sorted(Path(root).rglob("*")) if p.is_file()}
    missing = [f for f in ("meta/info.json",) if f not in new] + [d for d in ("data/", "videos/") if not any(k.startswith(d) for k in new)]
    if missing:
        raise RuntimeError(f"export {root} is empty or incomplete (missing {missing}); nothing was uploaded")
    return new


def publish(api, R, new, extra_ops, old_tag, dry=False):
    old = api.dataset_info(R).sha
    tags = {t.name: t.target_commit for t in api.list_repo_refs(R, **T).tags}
    if old_tag in tags and tags[old_tag] != old:
        raise RuntimeError(f"{R}: tag {old_tag} already points at {tags[old_tag][:8]}, not at main {old[:8]}; a previous run "
                           "probably committed already. Check the repo by hand; nothing was changed.")
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in new.items()] + list(extra_ops)
    stale = [f for f in api.list_repo_files(R, **T) if f.split("/")[0] in ("data", "meta", "videos") and f not in new]
    ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
    print(R, "| main", old[:8], "-> tag", old_tag, "| upload", sum(isinstance(o, CommitOperationAdd) for o in ops), "files | delete stale", stale, flush=True)
    if dry:
        return None
    if old_tag not in tags:  # a rerun after a failed commit finds the tag already at `old`
        api.create_tag(R, tag=old_tag, revision=old, tag_message="previous revision (task-specific pipelines)", **T)
    c = api.create_commit(R, operations=ops, parent_commit=old, commit_message=MESSAGE, **T)
    if "v3.0" in tags:
        api.delete_tag(R, tag="v3.0", **T)
    api.create_tag(R, tag="v3.0", revision=c.oid, **T)
    now = {t.name: t.target_commit for t in api.list_repo_refs(R, **T).tags}
    if now.get("v3.0") != c.oid:
        raise RuntimeError(f"{R}: v3.0 is {now.get('v3.0')}, expected {c.oid}; LeRobot loads v3.0, recreate it at {c.oid}")
    print("   new main", c.oid, flush=True)
    return c.oid


def jobs():
    code = sorted(S.glob("sfagen/*.py")) + sorted(S.glob("configs/*.yaml")) + [S / "README.md"]
    prov = Path(tempfile.mkdtemp()) / "provenance.json"; prov.write_text(json.dumps(upload_provenance(code), indent=1))
    out = []
    for task, cfg in (("pringles", "openarm_pringles_handover"), ("battery", "openarm_battery_insertion")):
        W = S / f"configs/work/{cfg}"
        syn = {"pringles": "ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00", "battery": "ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00"}[task]
        mix = {"pringles": "ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00", "battery": "ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00"}[task]
        figs = sorted(glob.glob(str(S / f"logs/coverage_{task}_*.png")))
        vids = {f"comparison_videos/{task}_unified_highlights.mp4": W / "video_highlights.mp4", f"comparison_videos/{task}_unified_all_2x.mp4": W / "video_all_2x.mp4"}
        prev = {"pringles": ("v04-65ep-two-pipelines", "v00-75ep-two-pipelines"), "battery": ("v01-65ep-two-pipelines", "v00-75ep-two-pipelines")}[task]
        for R, root, readme, old_tag, v in ((syn, W / "dataset", W / "README_synthetic.md", prev[0], vids), (mix, W / "dataset_mixed", W / "README_mixed.md", prev[1], {})):
            extra = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme)),
                     CommitOperationAdd(path_in_repo="PIPELINE.md", path_or_fileobj=str(S / "PIPELINE.md")),
                     CommitOperationAdd(path_in_repo="NOVELTY.md", path_or_fileobj=str(S / "NOVELTY.md")),
                     CommitOperationAdd(path_in_repo="provenance.json", path_or_fileobj=str(prov))]
            extra += [CommitOperationAdd(path_in_repo=f"sfagen/{p.relative_to(S)}", path_or_fileobj=str(p)) for p in code]
            extra += [CommitOperationAdd(path_in_repo=f"assets/{Path(f).name}", path_or_fileobj=f) for f in figs]
            extra += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(x)) for k, x in v.items() if Path(x).exists()]
            out.append((R, root, extra, old_tag))
    return out


if __name__ == "__main__":
    from huggingface_hub import HfApi
    api = HfApi(); J = jobs()
    checked = [(R, export_files(root), extra, old_tag) for R, root, extra, old_tag in J]  # all exports, before any write
    for R, new, extra, old_tag in checked:
        publish(api, R, new, extra, old_tag, dry=os.environ.get("DRY_RUN") == "1")
