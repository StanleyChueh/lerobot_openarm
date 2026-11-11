"""Read-only check of the 4 published sfagen datasets (local exports, confirmed against the Hub v3.0 file hashes).

(a) labels: every synthetic episode's exported action must equal  source action + (exported state - source state)(t + lead),
    the rule a' = a + dq(t + lead) the pipeline claims; a clipped action breaks it.
(b) cameras: per episode and camera, the metadata timestamp span must equal the episode length, and each video file's
    packet count must equal the frames the metadata assigns to it.
Writes published_check.json next to this script. Changes nothing."""
import hashlib, json, sys
from pathlib import Path
import av, numpy as np, pandas as pd, yaml

REPO = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation/sfagen")
sys.path.insert(0, str(REPO))  # the committed package (== the code uploaded with these datasets, 50832b3)
from sfagen.data import Source

WORK = Path("/home/stanley/Stanley_ws/sfa_hf_export/sfagen/configs/work")
TASKS = {"openarm_pringles_handover": ("ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00", "ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00"),
         "openarm_battery_insertion": ("ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00", "ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00")}
TOL = 2e-6  # float32 storage of values up to ~3 rad


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def hub_match(repo, root):
    from huggingface_hub import HfApi
    api = HfApi(); files = [f for f in api.list_repo_files(repo, repo_type="dataset", revision="v3.0") if f.split("/")[0] in ("data", "meta", "videos")]
    info = {i.path: i for i in api.get_paths_info(repo, files, revision="v3.0", repo_type="dataset")}
    bad = []
    for f in files:
        p = root / f
        if not p.is_file():
            bad.append((f, "missing locally")); continue
        i = info[f]
        want = i.lfs.sha256 if i.lfs else None
        if want is None:  # small non-LFS file: compare git blob sha1
            b = p.read_bytes(); got = hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest(); want = i.blob_id
        else:
            got = sha(p)
        if got != want:
            bad.append((f, "hash differs"))
    local = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and str(p.relative_to(root)).split("/")[0] in ("data", "meta", "videos")}
    return {"hub_files": len(files), "mismatches": bad, "local_only": sorted(local - set(files))}


def label_check(cfg, root, prov):
    exp = Source({**cfg["source"], "root": str(root)}); src = Source(cfg["source"])
    rows = []
    for e_mixed, p in enumerate(prov["episodes"]):  # list position == episode index in this export
        if not p.get("synthetic"):
            continue
        e_out = p.get("export_episode_index", p.get("episode_index")); e_src = p["source_real_episode"]
        S, A = exp.arrays(e_mixed); S0, A0 = src.arrays(e_src)
        lead = json.loads((root.parent / f"candidates/ep{e_src}/checks.json").read_text())[p["variation"]]["lead"]
        fut = np.minimum(np.arange(len(S)) + lead, len(S) - 1)
        err = np.abs(A - (A0 + (S[fut] - S0[fut])))
        rows.append({"episode": e_mixed, "synthetic_index": e_out, "source": e_src, "variation": p["variation"], "family": p["family"],
                     "max_err_rad": float(err.max()), "frames_over_tol": int((err.max(1) > TOL).sum()), "n_frames": len(S)})
    bad = [r for r in rows if r["max_err_rad"] > TOL]
    return {"synthetic_episodes": len(rows), "label_mismatch_episodes": len(bad), "max_err_rad": max((r["max_err_rad"] for r in rows), default=0.0), "mismatches": bad}


def span_check(root):
    info = json.loads((root / "meta/info.json").read_text()); fps = info["fps"]
    eps = pd.read_parquet(sorted((root / "meta/episodes").rglob("*.parquet")))
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    out = {}
    for c in cams:
        span = np.round((eps[f"videos/{c}/to_timestamp"] - eps[f"videos/{c}/from_timestamp"]) * fps).astype(int)
        bad_eps = eps.loc[span != eps["length"], "episode_index"].tolist()
        files = {}
        for (ci, fi), g in eps.groupby([f"videos/{c}/chunk_index", f"videos/{c}/file_index"]):
            f = root / info["video_path"].format(video_key=c, chunk_index=ci, file_index=fi)
            with av.open(str(f)) as cont:
                n = sum(1 for pk in cont.demux(video=0) if pk.size)
            files[str(f.relative_to(root))] = {"packets": n, "expected": int(g["length"].sum())}
        out[c] = {"episodes_span_ne_length": bad_eps, "files_with_frame_deficit": {k: v for k, v in files.items() if v["packets"] != v["expected"]}, "files": len(files)}
    return out


res = {}
for stem, repos in TASKS.items():
    cfg = yaml.safe_load((WORK.parent / f"{stem}.yaml").read_text())
    for kind, repo in zip(("dataset", "dataset_mixed"), repos):
        root = WORK / stem / kind; prov = json.loads((root / "meta/sfa_provenance.json").read_text())
        r = {"repo": repo, "local": str(root), "hub": hub_match(repo, root), "labels": label_check(cfg, root, prov), "cameras": span_check(root)}
        res[repo] = r
        print(repo, "| hub mismatches", len(r["hub"]["mismatches"]), "| label-mismatch eps", r["labels"]["label_mismatch_episodes"], "/", r["labels"]["synthetic_episodes"],
              "max", f'{r["labels"]["max_err_rad"]:.3g}', "| span", {c: (v["episodes_span_ne_length"], list(v["files_with_frame_deficit"])) for c, v in r["cameras"].items()}, flush=True)
(Path(__file__).parent / "published_check.json").write_text(json.dumps(res, indent=1))
