"""Synthetic miniature LeRobot datasets (v3.0 and v2.1 layouts) for the analyzer's tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NAMES16 = [f"{p}J{i}.pos" for i in range(1, 9) for p in ("R", "L")]
CAM = "observation.images.body_cam"


def handover_episode(n: int, rng: np.random.Generator, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """16-D OpenArm-like episode: right grasps at 30%, left at 55%, right releases at 65%; binary gripper commands."""
    t = np.linspace(0, 1, n)
    arm = np.stack([0.2 * np.sin(2 * np.pi * (t + 0.05 * j + offset)) for j in range(14)], 1)
    act = np.zeros((n, 16))
    act[:, :14] = arm
    r_close, l_close, r_open = int(0.30 * n), int(0.55 * n), int(0.65 * n)
    act[:, 14] = -1.3
    act[r_close:r_open, 14] = 0.0
    act[:, 15] = 1.21
    act[l_close:, 15] = 0.0
    st = np.zeros_like(act)
    st[1:, :14] = act[:-1, :14]  # state lags the command by one frame
    st[0, :14] = act[0, :14]
    st[:, 14] = np.where(act[:, 14] == 0, -0.52, -1.24)  # right stalls on the can
    st[:, 15] = np.where(act[:, 15] == 0, 0.41, 1.16)
    st += rng.normal(0, 1e-4, st.shape)
    return act.astype(np.float32), st.astype(np.float32)


def _stats(arr: np.ndarray) -> dict:
    return {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(), "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(), "count": [int(len(arr))]}


def write_v30(root: Path, lengths=(60, 60, 60), *, fps: int = 30, names=NAMES16, cameras=(CAM,), seed: int = 0,
              make_videos: bool = True, mutate=None) -> Path:
    """`mutate(df, episodes_df, info, stats)` may edit the tables before they are written."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    rows, eps, idx = [], [], 0
    dim = len(names)
    for e, n in enumerate(lengths):
        if dim == 16:
            a, s = handover_episode(n, rng, offset=0.1 * e) if n else (np.zeros((0, 16)), np.zeros((0, 16)))
        else:
            a = rng.normal(0, 0.1, (n, dim)).astype(np.float32)
            s = a + rng.normal(0, 0.01, (n, dim)).astype(np.float32)
        for k in range(n):
            rows.append({"action": a[k], "observation.state": s[k], "timestamp": np.float32(k / fps), "frame_index": k,
                         "episode_index": e, "index": idx, "task_index": 0})
            idx += 1
        ep = {"episode_index": e, "length": n, "dataset_from_index": idx - n, "dataset_to_index": idx,
              "tasks": ["pick"], "data/chunk_index": 0, "data/file_index": 0}
        t0 = sum(lengths[:e]) / fps
        for cam in cameras:
            ep.update({f"videos/{cam}/chunk_index": 0, f"videos/{cam}/file_index": 0,
                       f"videos/{cam}/from_timestamp": t0, f"videos/{cam}/to_timestamp": t0 + n / fps})
        eps.append(ep)
    df = pd.DataFrame(rows)
    eps_df = pd.DataFrame(eps)
    features = {"action": {"dtype": "float32", "shape": [dim], "names": list(names)},
                "observation.state": {"dtype": "float32", "shape": [dim], "names": list(names)},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None}}
    for cam in cameras:
        features[cam] = {"dtype": "video", "shape": [48, 64, 3], "names": ["height", "width", "channels"],
                         "info": {"video.fps": fps, "video.codec": "av1"}}
    info = {"codebase_version": "v3.0", "robot_type": "synthetic", "fps": fps, "total_episodes": len(lengths),
            "total_frames": int(sum(lengths)), "total_tasks": 1, "chunks_size": 1000,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features, "splits": {"train": f"0:{len(lengths)}"}}
    stats = {}
    if len(df):
        for key in ("action", "observation.state"):
            stats[key] = _stats(np.stack(df[key].values).astype(np.float64))
    if mutate:
        df, eps_df, info, stats = mutate(df, eps_df, info, stats)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick"], name="task")).to_parquet(root / "meta" / "tasks.parquet")
    eps_df.to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    df.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    if make_videos:
        for cam in cameras:
            p = root / "videos" / cam / "chunk-000"
            p.mkdir(parents=True, exist_ok=True)
            (p / "file-000.mp4").write_bytes(b"\x00")  # existence only; frames are never decoded in tests
    return root


def write_v21(root: Path, lengths=(40, 40), fps: int = 30, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    idx = 0
    ep_lines, st_lines = [], []
    for e, n in enumerate(lengths):
        a, s = handover_episode(n, rng)
        df = pd.DataFrame({"action": list(a), "observation.state": list(s), "timestamp": np.arange(n, dtype=np.float32) / fps,
                           "frame_index": np.arange(n), "episode_index": e, "index": np.arange(idx, idx + n),
                           "task_index": 0})
        idx += n
        df.to_parquet(root / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
        ep_lines.append(json.dumps({"episode_index": e, "tasks": ["pick"], "length": n}))
        st_lines.append(json.dumps({"episode_index": e, "stats": {
            "action": _stats(a.astype(np.float64)), "observation.state": _stats(s.astype(np.float64))}}))
    info = {"codebase_version": "v2.1", "robot_type": "synthetic", "fps": fps, "total_episodes": len(lengths),
            "total_frames": int(sum(lengths)), "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {"action": {"dtype": "float32", "shape": [16], "names": NAMES16},
                         "observation.state": {"dtype": "float32", "shape": [16], "names": NAMES16},
                         "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
                         "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
                         "task_index": {"dtype": "int64", "shape": [1]}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines) + "\n")
    (root / "meta" / "episodes_stats.jsonl").write_text("\n".join(st_lines) + "\n")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick"}) + "\n")
    return root


def write_checkpoint(root: Path, *, state_decl: int = 6, state_norm: int | None = 16, action: int = 16,
                     max_state_dim: int = 32, cameras=("observation.images.camera3",), rename=None) -> Path:
    from safetensors.numpy import save_file

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = {"type": "smolvla", "max_state_dim": max_state_dim, "max_action_dim": 32, "chunk_size": 50,
           "resize_imgs_with_padding": [512, 512], "empty_cameras": 0,
           "input_features": {"observation.state": {"type": "STATE", "shape": [state_decl]},
                              **{c: {"type": "VISUAL", "shape": [3, 256, 256]} for c in cameras}},
           "output_features": {"action": {"type": "ACTION", "shape": [action]}}}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "train_config.json").write_text(json.dumps({
        "dataset": {"repo_id": "synthetic/ds", "episodes": None}, "rename_map": rename or {CAM: cameras[0]} if cameras
        else {}, "steps": 10, "batch_size": 2, "policy": {"pretrained_path": "lerobot/smolvla_base"}}))
    tensors = {"action.mean": np.zeros(action, np.float32), "action.std": np.ones(action, np.float32),
               "action.count": np.array([1.0], np.float32)}
    if state_norm:
        tensors.update({"observation.state.mean": np.zeros(state_norm, np.float32),
                        "observation.state.std": np.ones(state_norm, np.float32),
                        "observation.state.count": np.array([1.0], np.float32)})
    save_file(tensors, str(root / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    return root


def tree_digest(root: Path) -> dict:
    import hashlib

    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}
