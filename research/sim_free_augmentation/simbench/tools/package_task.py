"""Package one RoboTwin task into three local LeRobot v3.0 datasets (R03):
  source/     the 10 source demos (copy of the robotwin_pilot subset, unchanged)
  synthetic/  all episode groups' `<group>_export/dataset` merged (synthetic only)
  mixed/      source + synthetic, built the way sfagen's build_mixed does it (is_synthetic flag; real frames get
              wrist_view_valid = 1 and zero wrist pose offset; heterogeneous codecs: containers copied unchanged)
Provenance (meta/sfa_provenance_merged.json) lists every episode with its source demo, family and shift.
Usage: package_task.py TASK RUN_DIR SOURCE_ROOT OUT_ROOT     (refuses to overwrite OUT_ROOT/TASK)"""
import json, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import add_features, merge_datasets
from lerobot.datasets.io_utils import write_info
from sfagen.build import mixed_video_features
from sfagen.verify import video_span_report

task, run, src_root, out_root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out = out_root / task; assert not out.exists(), f"{out} exists"; out.mkdir(parents=True)
groups = sorted(p.parents[2].name.removesuffix("_export") for p in (run / task).glob("*_export/dataset/meta/info.json"))
syn_parts = [LeRobotDataset(f"local/{task}_{g}", root=run / task / f"{g}_export" / "dataset") for g in groups]
syn_parts = [d for d in syn_parts if d.meta.total_episodes > 0]
# synthetic only
if len(syn_parts) > 1:
    syn = merge_datasets(syn_parts, output_repo_id=f"local/robotwin_{task}_sfagen_synthetic", output_dir=out / "synthetic")
else:
    shutil.copytree(syn_parts[0].root, out / "synthetic"); syn = LeRobotDataset(f"local/robotwin_{task}_sfagen_synthetic", root=out / "synthetic")
prov = []; off = 0
for g, d in zip(groups, syn_parts):
    for e in json.loads((d.root / "meta/sfa_provenance.json").read_text())["episodes"]:
        prov.append({**e, "group": g, "episode_index": off + e["export_episode_index"]})
    off += d.meta.total_episodes
(out / "synthetic/meta/sfa_provenance_merged.json").write_text(json.dumps({"task": task, "groups": groups, "episodes": prov}, indent=1, default=float))
# source only
shutil.copytree(src_root / task, out / "source"); real = LeRobotDataset(f"local/robotwin_{task}_source", root=out / "source")
# mixed = source + synthetic (sfagen build_mixed recipe)
wr = [k for k in syn.meta.features if k.startswith("observation.images.") and "wrist" in k]; cams = [k for k in syn.meta.features if k.startswith("observation.images.")]
nr, ns = real.meta.total_frames, syn.meta.total_frames; f1 = {"dtype": "float32", "shape": [1], "names": ["is_synthetic"]}
tmp = out / "_mixed_tmp"; tmp.mkdir()
r2 = add_features(real, {"wrist_view_valid": (np.ones((nr, len(wr)), np.float32), syn.meta.features["wrist_view_valid"]),
                         "wrist_view_pose_offset": (np.zeros((nr, 2 * len(wr)), np.float32), syn.meta.features["wrist_view_pose_offset"]),
                         "is_synthetic": (np.zeros((nr, 1), np.float32), f1)}, output_dir=tmp / "real", repo_id="tmp/real")
s2 = add_features(syn, {"is_synthetic": (np.ones((ns, 1), np.float32), f1)}, output_dir=tmp / "syn", repo_id="tmp/syn")
def harmonize(fs, fps):
    """metadata-only: per-feature `fps` = dataset fps on both; video `info` fields that differ or exist on one side -> None
    (same principle as sfagen.build.mixed_video_features; video containers are copied unchanged)."""
    for k in fs[0]:
        for f in fs:
            if "fps" in f[k] or any("fps" in g[k] for g in fs):
                f[k]["fps"] = fps
        if any("info" in f[k] for f in fs):
            keys = set().union(*(f[k].get("info", {}) for f in fs))
            merged = {kk: (fs[0][k].get("info", {}).get(kk) if all(f[k].get("info", {}).get(kk) == fs[0][k].get("info", {}).get(kk) for f in fs) else None) for kk in keys}
            for f in fs:
                f[k]["info"] = dict(merged)
    return fs


for d, feats in zip((r2, s2), harmonize(mixed_video_features([r2.meta.features, s2.meta.features]), int(syn.meta.fps))):
    d.meta.info.features = feats; write_info(d.meta.info, d.root)
mix = merge_datasets([r2, s2], output_repo_id=f"local/robotwin_{task}_sfagen_mixed", output_dir=out / "mixed", concatenate_videos=False)
nre = real.meta.total_episodes; src_ids = sorted(int(e) for e in np.unique(real.hf_dataset["episode_index"]))
(out / "mixed/meta/sfa_provenance_merged.json").write_text(json.dumps({"task": task, "real_episodes": list(range(nre)), "per_frame_flag": "is_synthetic",
    "episodes": [{"episode_index": i, "synthetic": False, "source_real_episode": e} for i, e in enumerate(src_ids)] +
                [{**e, "episode_index": nre + e["episode_index"]} for e in prov]}, indent=1, default=float))
shutil.rmtree(tmp)
spans = video_span_report(out / "mixed", cams, int(syn.meta.fps)); (out / "mixed/meta/video_span_verification.json").write_text(json.dumps(spans, indent=1))
assert spans["pass"], "mixed video spans failed"
print(json.dumps({"task": task, "groups": groups, "source_episodes": nre, "synthetic_episodes": syn.meta.total_episodes,
                  "mixed_episodes": mix.meta.total_episodes, "frames": {"source": nr, "synthetic": ns, "mixed": mix.meta.total_frames}}))
