"""Writes ../ARTIFACT_PROVENANCE.json: Git state, dataset/model revisions, library versions, every existing artifact read
(path, size, SHA-256), every generated file (SHA-256, committed or local-only) and a summary of each diagnostic run."""
import glob, json, os, subprocess, time
from pathlib import Path
from _common import GUIDE, REPO, SFAGEN, TASKS, sha256

def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()

def f(p):
    p = Path(p); return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} if p.exists() else {"path": str(p), "missing": True}

import importlib.metadata as md
libs = {}
for k in ("torch", "torchvision", "transformers", "pin", "opencv-python", "av", "lerobot", "numpy", "matplotlib", "huggingface_hub"):
    try:
        libs[k] = md.version(k)
    except Exception:
        libs[k] = "UNKNOWN"
existing = {}
for t, T_ in TASKS.items():
    W = T_["work"]; E, key = T_["source_episode"], T_["variation"]
    existing[t] = {"config": f(T_["config"]), "calibration": f(W / "calibration.json"), "table_height": f(W / "table_height.json"),
                   "recon_cache": f(W / f"cache/recon_ep{E}.npz"), "sam2_cache": f(W / f"cache/sam2_ep{E}.npz"),
                   "candidate": f(W / f"candidates/ep{E}/{key}.npz"), "checks": f(W / f"candidates/ep{E}/checks.json"),
                   "body_render": f(W / f"body/{E}_{key}.npy"), "body_render_log": f(W / f"body/{E}_{key}.json"),
                   "verification": f(W / "verification.json"), "profile_production": f(W / "profile.jsonl"),
                   "profile_clean_one_episode": f(SFAGEN / f"logs/prof_{W.name}/profile.jsonl"),
                   "benchmark": f(SFAGEN / f"logs/xbench_{t}_final.json"),
                   "dataset_provenance": f(W / "dataset/meta/sfa_provenance.json"),
                   "comparison_videos_local": [f(W / "video_highlights.mp4"), f(W / "video_all_2x.mp4")]}
existing["historical"] = {"v1_track_py": f(REPO / "research/sim_free_augmentation/sfa/track.py"), "v1_results_ep0": f(REPO / "research/sim_free_augmentation/outputs/ep0/results.json"),
                          "gate1_annotation_eval": f(REPO / "research/contact_aware_augmentation/outputs/gate1/annotation_eval.json"),
                          "gate1r_annotation_eval": f(REPO / "research/contact_aware_augmentation/outputs/gate1r/annotation_eval.json")}
ignored = set(subprocess.run(["git", "-C", str(GUIDE), "ls-files", "--others", "--ignored", "--exclude-standard", "."], capture_output=True, text=True).stdout.split())
gen = []
for p in sorted(GUIDE.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts and p.name != "ARTIFACT_PROVENANCE.json":
        rel = str(p.relative_to(GUIDE)); gen.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p),
                                                     "in_git": rel not in ignored and not p.suffix in (".mp4", ".npz", ".npy", ".gif"),
                                                     "local_location": str(p)})
runs = {}
for r in sorted(glob.glob(str(GUIDE / "diagnostics/run_*.json"))):
    d = json.loads(open(r).read()); runs[Path(r).stem] = {k: d[k] for k in ("command", "git_sha", "started", "wall_time_s", "gpu", "inputs", "error")}
    runs[Path(r).stem]["outputs_at_run_time"] = d["outputs"]
out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
       "purpose": "diagnostic-only visual walkthrough of the existing sfagen pipeline; no dataset, model, Hub or frozen-study modification",
       "git": {"repo": str(REPO), "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
               "pipeline_commit": "50832b3a6342e5e342cbc91df39fd3d417c261a3 (research/sfagen-unified-pipeline)",
               "sfagen_working_copy": str(SFAGEN / "sfagen") + " (diff -rq vs research/sim_free_augmentation/sfagen/sfagen: identical)",
               "historical": {"v1": "e8abc26", "v2": ["b9b7dfe", "03ba314"]}},
       "datasets": {t: {"published": T_["repo"], "published_revision_v3.0": T_["repo_rev"], "source": T_["source_repo"], "source_revision_used": T_["source_rev"],
                        "representative_pair": {"source_episode": T_["source_episode"], "published_episode": T_["export_episode"], "variation": T_["variation"]}} for t, T_ in TASKS.items()},
       "notes_on_revisions": "battery source v3.0 tag points at 92476e0e, sfagen read main a1c29d77; mixed datasets: pringles 9f5ed124, battery 2de75e9f (not re-read here)",
       "models": {"sam2": {"id": "facebook/sam2.1-hiera-small", "hf_revision": "ee5bba1d82bb8749febdf90f45e84b687142ba03", "dtype": "bfloat16", "package": "transformers Sam2VideoModel"},
                  "raft_historical": {"id": "torchvision raft_large", "weights": "Raft_Large_Weights.C_T_SKHT_V2", "num_flow_updates": 12}},
       "libraries": libs, "hardware": "2x NVIDIA GeForce RTX 4090 24 GB (one per diagnostic run)",
       "existing_artifacts_read": existing,
       "new_diagnostic_runs": runs,
       "post_processing": {"videos": "all stage MP4s transcoded from OpenCV mp4v to H.264 (libx264 crf 26, yuv420p, faststart) for browser playback; hashes below are of the H.264 files, hashes at run time are in new_diagnostic_runs",
                           "images": "PNG figures > 250 kB converted to JPEG q88; original PNG SHA-256 in diagnostics/png_to_jpg_conversions.json"},
       "generated_files": gen,
       "not_done": ["no dataset regeneration", "no Hub writes", "no VLA training", "no robot/CAN", "no simulator", "no frozen study modified",
                    "no v4A withheld wrist streams (pool records 10/17/24/31/38/45) read"]}
(GUIDE / "ARTIFACT_PROVENANCE.json").write_text(json.dumps(out, indent=1, default=str)); print(len(gen), "files;", len(runs), "runs")
