from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
HERE = Path(__file__).resolve().parent
api = HfApi(); R = "ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00"; T = dict(repo_type="dataset")
OLD = "f9692b379c51b2e4ebe091255ebe14aa0d9dd5fe"
assert api.dataset_info(R).sha == OLD
api.create_tag(R, tag="v02-20ep-v3render", revision=OLD, tag_message="20 synthetic episodes, v3 homography renderer", **T)
root = HERE / "dataset"
ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(root)), path_or_fileobj=str(p)) for p in sorted(root.rglob("*")) if p.is_file()]
ops += [CommitOperationAdd(path_in_repo=f"assets/{p.name}", path_or_fileobj=str(p)) for p in sorted((HERE / "dataset_assets").glob("*.png"))]
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(HERE / "README.md")))
c = api.create_commit(R, operations=ops, parent_commit=OLD, commit_message=(
    "v3: 55 replay-checked synthetic episodes (46 new can positions, 9 timing); body_cam re-rendered with mesh-guided "
    "warping (held-out +0.98 dB PSNR vs previous renderer); can-position heatmaps"), **T)
api.delete_tag(R, tag="v3.0", **T); api.create_tag(R, tag="v3.0", revision=c.oid, **T)
print("new main", c.oid)
