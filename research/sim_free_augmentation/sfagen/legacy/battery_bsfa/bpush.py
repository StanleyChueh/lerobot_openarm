from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
HERE = Path(__file__).resolve().parent
api = HfApi(); R = "ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00"; T = dict(repo_type="dataset")
api.create_repo(R, private=True, exist_ok=False, **T)
root = HERE / "dataset"
ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(root)), path_or_fileobj=str(p)) for p in sorted(root.rglob("*")) if p.is_file()]
ops += [CommitOperationAdd(path_in_repo=f"assets/{p.name}", path_or_fileobj=str(p)) for p in sorted((HERE / "analysis").glob("battery_heatmap_*.png"))]
ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(HERE / "README.md")))
c = api.create_commit(R, operations=ops, commit_message="Synthetic-only battery-insertion dataset: battery-position variants of the 10 real demos (mesh-guided rendering, replay-checked)", **T)
api.create_tag(R, tag="v3.0", revision=c.oid, **T)
print("main", c.oid)
