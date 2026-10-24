"""Read-only access to a LeRobot dataset (v2.1 or v3.x), local or on the Hugging Face Hub.

Nothing here writes into a dataset directory. Hub datasets are fetched into the Hugging Face cache
(meta/ and data/ only; single video files on demand), never into the dataset's own root.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_MAJOR = {"v2", "v3"}
VECTOR_KEYS = ("observation.state", "action")


def read_parquet(path: str | Path) -> pd.DataFrame:
    # fastparquet chokes on LeRobot's dotted column names; insist on pyarrow (as analyze_dataset_quality does).
    return pd.read_parquet(path, engine="pyarrow")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _unflatten(flat: dict) -> dict:
    out: dict = {}
    for key, value in flat.items():
        *parents, leaf = key.split("/")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out


@dataclass
class HubProvenance:
    repo_id: str
    requested_revision: str | None
    resolved_revision: str | None = None
    main_sha: str | None = None
    codebase_tag_sha: str | None = None
    error: str | None = None


@dataclass
class DatasetView:
    """Everything the analyzer needs, loaded once. Image/video columns are never loaded here."""

    label: str
    root: Path
    info: dict
    data: pd.DataFrame  # non-video columns, sorted by (episode_index, frame_index)
    episodes: pd.DataFrame  # one row per episode in metadata, indexed by episode_index
    stats: dict | None
    version: str
    repo_id: str | None = None
    hub: HubProvenance | None = None
    loader_notes: list[str] = field(default_factory=list)
    episode_filter: list[int] | None = None

    @property
    def fps(self) -> float:
        return float(self.info.get("fps") or 0)

    @property
    def features(self) -> dict:
        return self.info.get("features", {})

    @property
    def camera_keys(self) -> list[str]:
        return [k for k, v in self.features.items() if v.get("dtype") in ("video", "image")]

    @property
    def major(self) -> str:
        return self.version.split(".")[0]

    def names(self, key: str) -> list[str]:
        names = self.features.get(key, {}).get("names")
        if isinstance(names, dict):  # some v2 datasets store {"motors": [...]}
            names = next(iter(names.values()))
        return list(names) if names else [f"{key}[{i}]" for i in range(self.dim(key))]

    def dim(self, key: str) -> int:
        shape = self.features.get(key, {}).get("shape") or [0]
        return int(np.prod(shape))

    def vectors(self, key: str) -> dict[int, np.ndarray]:
        """episode_index -> (T, D) float64 array for a vector feature."""
        if key not in self.data.columns:
            return {}
        return {int(e): np.stack(g[key].values).astype(np.float64) if len(g) else np.zeros((0, self.dim(key)))
                for e, g in self.data.groupby("episode_index")}

    # ---------------------------------------------------------------- video access (on demand)
    def video_path(self, camera_key: str, episode: int) -> tuple[str, float]:
        """(relative video path, start timestamp of this episode inside that file)."""
        if self.major == "v3":
            m = self.episodes.loc[episode]
            rel = self.info["video_path"].format(
                video_key=camera_key,
                chunk_index=int(m[f"videos/{camera_key}/chunk_index"]),
                file_index=int(m[f"videos/{camera_key}/file_index"]),
            )
            return rel, float(m[f"videos/{camera_key}/from_timestamp"])
        chunk = episode // int(self.info.get("chunks_size", 1000))
        rel = self.info["video_path"].format(episode_chunk=chunk, video_key=camera_key, episode_index=episode)
        return rel, 0.0

    def local_video(self, rel: str) -> Path | None:
        p = self.root / rel
        if p.exists():
            return p
        if self.repo_id and self.hub is not None:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(self.repo_id, rel, repo_type="dataset", revision=self.hub.resolved_revision))
        return None

    def read_frame(self, camera_key: str, episode: int, frame: int = 0) -> np.ndarray:
        """One RGB uint8 frame (H, W, 3) of an episode, decoded by frame time."""
        rel, t0 = self.video_path(camera_key, episode)
        path = self.local_video(rel)
        if path is None:
            raise FileNotFoundError(f"video for {camera_key} episode {episode} not found: {rel}")
        ts = t0 + frame / self.fps
        try:
            from lerobot.datasets.video_utils import decode_video_frames

            frames = decode_video_frames(path, [ts], tolerance_s=0.5 / self.fps, backend="pyav")
            return (frames[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        except ImportError:
            pass
        import av  # PyAV ships with LeRobot

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(int(max(ts - 1.0, 0) / stream.time_base), stream=stream)
            best = None
            for fr in container.decode(stream):
                if fr.time is None:
                    continue
                if best is None or abs(fr.time - ts) < abs(best.time - ts):
                    best = fr
                if fr.time > ts + 0.5 / self.fps:
                    break
            if best is None:
                raise RuntimeError(f"no frame decoded at t={ts:.3f}s in {path}")
            return best.to_ndarray(format="rgb24")


def _resolve_hub(repo_id: str, revision: str | None, offline: bool) -> tuple[Path, HubProvenance]:
    from huggingface_hub import snapshot_download

    prov = HubProvenance(repo_id=repo_id, requested_revision=revision)
    rev = revision
    if not offline:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            refs = api.list_repo_refs(repo_id, repo_type="dataset")
            prov.main_sha = next((b.target_commit for b in refs.branches if b.name == "main"), None)
            tags = {t.name: t.target_commit for t in refs.tags}
            # lerobot-train pulls the codebase-version tag, not main (memory: "Hub v3.0 tag pitfall").
            codebase_tag = max((t for t in tags if t.startswith("v3")), default=None) or max(
                (t for t in tags if t.startswith("v2")), default=None)
            prov.codebase_tag_sha = tags.get(codebase_tag) if codebase_tag else None
            if rev is None:
                rev = codebase_tag or "main"
            rev = tags.get(rev) or (prov.main_sha if rev == "main" else rev)  # pin to a commit sha
        except Exception as exc:  # network trouble must not stop a local-cache analysis
            prov.error = f"ref lookup failed: {exc}"
    # Reuse a complete copy in LeRobot's own snapshot cache ($HF_LEROBOT_HOME/hub, current LeRobot layout) before
    # downloading anything; otherwise fetch meta/ and data/ into the ordinary Hugging Face cache.
    cached = _lerobot_cache_snapshot(repo_id, rev)
    if cached is not None:
        prov.resolved_revision = cached.name
        return cached, prov
    root = Path(snapshot_download(repo_id, repo_type="dataset", revision=rev,
                                  allow_patterns=["meta/*", "meta/**", "data/**", "README.md"],
                                  local_files_only=offline))
    prov.resolved_revision = root.name
    return root, prov


def _lerobot_cache_snapshot(repo_id: str, rev: str | None) -> Path | None:
    home = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))
    repo_dir = home / "hub" / f"datasets--{repo_id.replace('/', '--')}"
    if not repo_dir.is_dir():
        return None
    sha = rev
    for ref in ([rev] if rev else []) + ["v3.0", "main"]:
        ref_file = repo_dir / "refs" / ref
        if ref_file.exists():
            sha = ref_file.read_text().strip()
            break
    def complete(s: Path) -> bool:
        return s.is_dir() and (s / "meta" / "info.json").exists() and any((s / "data").rglob("*.parquet"))

    if sha:
        snap = repo_dir / "snapshots" / sha
        return snap if complete(snap) else None
    snaps = [s for s in (repo_dir / "snapshots").iterdir() if complete(s)] if (repo_dir / "snapshots").is_dir() else []
    return snaps[0] if len(snaps) == 1 else None  # offline and unpinned: only an unambiguous copy is used


def load_dataset(*, repo_id: str | None = None, root: str | Path | None = None, revision: str | None = None,
                 label: str | None = None, offline: bool = False, episodes: list[int] | None = None) -> DatasetView:
    notes: list[str] = []
    hub = None
    if root is None:
        if repo_id is None:
            raise ValueError("give repo_id or root")
        root, hub = _resolve_hub(repo_id, revision, offline)
    root = Path(root).expanduser()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} missing: not a LeRobot dataset root")
    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", "unknown"))
    major = version.split(".")[0]
    if major not in SUPPORTED_MAJOR:
        raise ValueError(f"unsupported codebase_version {version!r}")

    if major == "v3":
        data_files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
        ep_files = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
        eps = (pd.concat([read_parquet(f) for f in ep_files], ignore_index=True) if ep_files
               else pd.DataFrame(columns=["episode_index", "length"]))
        stats_path = root / "meta" / "stats.json"
        stats = _load_stats_json(stats_path) if stats_path.exists() else None
        try:  # cross-check against LeRobot's own reader when it is installed
            try:
                from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
            except ImportError:
                from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            meta = LeRobotDatasetMetadata(repo_id or f"local/{root.name}", root=root)
            if int(meta.total_episodes) != int(info.get("total_episodes", -1)):
                notes.append("LeRobotDatasetMetadata total_episodes disagrees with info.json")
            notes.append(f"v3 metadata readable by LeRobotDatasetMetadata ({type(meta).__module__})")
        except Exception as exc:
            notes.append(f"LeRobotDatasetMetadata could not read this root: {type(exc).__name__}: {exc}")
    else:
        data_files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
        ep_path = root / "meta" / "episodes.jsonl"
        eps = pd.DataFrame(_read_jsonl(ep_path)) if ep_path.exists() else pd.DataFrame(columns=["episode_index", "length"])
        stats = None
        if (root / "meta" / "stats.json").exists():
            stats = _load_stats_json(root / "meta" / "stats.json")
        elif (root / "meta" / "episodes_stats.jsonl").exists():
            stats = _aggregate_v21_stats(_read_jsonl(root / "meta" / "episodes_stats.jsonl"))
            notes.append("stats aggregated from meta/episodes_stats.jsonl (v2.1)")
        notes.append(f"{version}: current LeRobot (v3 API) refuses this format; convert with "
                     "lerobot.datasets.v30.convert_dataset_v21_to_v30 before training. Parsed read-only here.")

    if not data_files and int(info.get("total_episodes") or 0) > 0:
        raise FileNotFoundError(f"{root}: info.json lists {info.get('total_episodes')} episodes but there are no "
                                "data parquet files (incomplete snapshot; rerun without --offline or pass --root)")
    video_cols = {k for k, v in info.get("features", {}).items() if v.get("dtype") in ("video", "image")}
    frames = []
    for f in data_files:
        cols = [c for c in _parquet_columns(f) if c not in video_cols]
        frames.append(pd.read_parquet(f, engine="pyarrow", columns=cols))
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["episode_index", "frame_index"])
    if len(data):
        data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    if "episode_index" in eps.columns:
        eps = eps.set_index("episode_index", drop=False).sort_index()
    if episodes is not None:
        data = data[data["episode_index"].isin(episodes)].reset_index(drop=True)
        eps = eps[eps.index.isin(episodes)]
        notes.append(f"restricted to episodes {episodes}")
    return DatasetView(label=label or (repo_id or root.name).split("/")[-1], root=root, info=info, data=data,
                       episodes=eps, stats=stats, version=version, repo_id=repo_id, hub=hub, loader_notes=notes,
                       episode_filter=episodes)


def _load_stats_json(path: Path) -> dict:
    raw = json.loads(path.read_text())
    return _unflatten(raw) if any("/" in k for k in raw) else raw


def _parquet_columns(path: str) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def _aggregate_v21_stats(rows: list[dict]) -> dict:
    """Pool per-episode mean/std/min/max/count (v2.1 episodes_stats.jsonl) into dataset stats."""
    pooled: dict = {}
    keys = set().union(*(r.get("stats", {}).keys() for r in rows)) if rows else set()
    for key in keys:
        parts = [r["stats"][key] for r in rows if key in r.get("stats", {})]
        try:
            n = np.array([np.asarray(p["count"]).reshape(-1)[0] for p in parts], dtype=float)
            mean = np.stack([np.asarray(p["mean"], dtype=float) for p in parts])
            std = np.stack([np.asarray(p["std"], dtype=float) for p in parts])
        except (KeyError, ValueError):
            continue
        w = (n / n.sum()).reshape((-1,) + (1,) * (mean.ndim - 1))
        mu = (w * mean).sum(0)
        var = (w * (std**2 + (mean - mu) ** 2)).sum(0)
        pooled[key] = {"mean": mu.tolist(), "std": np.sqrt(var).tolist(), "count": [float(n.sum())],
                       "min": np.min(np.stack([np.asarray(p["min"], float) for p in parts]), 0).tolist(),
                       "max": np.max(np.stack([np.asarray(p["max"], float) for p in parts]), 0).tolist()}
    return pooled


def dataset_fingerprint(view: DatasetView) -> dict:
    """Small provenance record: where the data came from and what it contains."""
    return {
        "label": view.label,
        "repo_id": view.repo_id,
        "root": str(view.root),
        "codebase_version": view.version,
        "requested_revision": view.hub.requested_revision if view.hub else None,
        "resolved_revision": view.hub.resolved_revision if view.hub else None,
        "hub_main_sha": view.hub.main_sha if view.hub else None,
        "hub_codebase_tag_sha": view.hub.codebase_tag_sha if view.hub else None,
        "robot_type": view.info.get("robot_type"),
        "fps": view.fps,
        "total_episodes_info": view.info.get("total_episodes"),
        "total_frames_info": view.info.get("total_frames"),
        "episodes_loaded": int(view.data["episode_index"].nunique()) if len(view.data) else 0,
        "frames_loaded": int(len(view.data)),
        "episode_filter": view.episode_filter,
        "loader_notes": view.loader_notes,
        "user": os.environ.get("USER"),
    }
