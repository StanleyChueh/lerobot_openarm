import hashlib
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
api = HfApi(); W = Path("/home/stanley/Stanley_ws/sfa_hf_export/sfagen/configs/work")
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for R, loc in (("ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00", W / "openarm_pringles_handover/dataset"),
               ("ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00", W / "openarm_pringles_handover/dataset_mixed"),
               ("ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00", W / "openarm_battery_insertion/dataset"),
               ("ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00", W / "openarm_battery_insertion/dataset_mixed")):
    snap = Path(snapshot_download(R, repo_type="dataset", revision="v3.0", local_dir=f"hubcheck/{R.split('/')[1]}", allow_patterns=["data/*", "meta/*", "videos/*", "README.md"]))
    files = [p for p in loc.rglob("*") if p.is_file()]; mism = [str(p.relative_to(loc)) for p in files if h(p) != h(snap / p.relative_to(loc))]
    ds = LeRobotDataset(R, root=snap); x = ds[len(ds) // 2]; i = api.dataset_info(R)
    print(R.split("/")[1], "| public", not i.private, "| main", i.sha[:8], "| files", len(files), "mismatch", mism, "| episodes", ds.meta.total_episodes, "frames", ds.meta.total_frames, "| tags", [t.name for t in api.list_repo_refs(R, repo_type="dataset").tags], flush=True)
