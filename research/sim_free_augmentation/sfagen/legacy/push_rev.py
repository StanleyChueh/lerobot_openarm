"""Publish a new revision of a LeRobot dataset repo: tag the current main (old revision stays loadable by tag), upload
every file of the local dataset + README + assets, delete dataset files that the new revision no longer has, and move the
'v3.0' tag (the tag LeRobot loads) to the new commit.
usage: push_rev.py REPO DATASET_DIR README ASSETS_GLOB EXPECTED_OLD_SHA OLD_TAG "OLD_TAG_MSG" "COMMIT_MSG" """
import glob, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
R, root, readme, assets, old, old_tag, old_msg, msg = sys.argv[1:9]
extra = sys.argv[9].split(",") if len(sys.argv) > 9 else []
api = HfApi(); T = dict(repo_type="dataset"); root = Path(root)
import os
DRY = os.environ.get("DRY_RUN") == "1"
assert api.dataset_info(R).sha == old, (api.dataset_info(R).sha, old)
if not DRY:
    api.create_tag(R, tag=old_tag, revision=old, tag_message=old_msg, **T)
new = {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}
ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in new.items()]
ops += [CommitOperationAdd(path_in_repo=f"assets/{Path(p).name}", path_or_fileobj=p) for p in sorted(glob.glob(assets))]
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme))
ops += [CommitOperationAdd(path_in_repo=Path(p).name, path_or_fileobj=p) for p in extra]
stale = [f for f in api.list_repo_files(R, **T) if f.split("/")[0] in ("data", "meta", "videos") and f not in new]
ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
if DRY:
    print("DRY RUN", R, "| current main", old, "-> will be tagged", old_tag)
    print("  upload", len(ops) - len(stale), "files:", sorted(o.path_in_repo for o in ops if isinstance(o, CommitOperationAdd)))
    print("  delete stale:", stale); print("  then move tag v3.0 to the new commit"); sys.exit(0)
c = api.create_commit(R, operations=ops, parent_commit=old, commit_message=msg, **T)
api.delete_tag(R, tag="v3.0", **T); api.create_tag(R, tag="v3.0", revision=c.oid, **T)
print("new main", c.oid, "stale removed", stale)
