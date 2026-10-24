"""TECHNICAL READINESS: schema, dimensions, timing, finiteness, cameras and normalization statistics."""

from __future__ import annotations

import numpy as np

from .checks import FAIL, PASS, UNKNOWN, WARN, Check
from .loader import VECTOR_KEYS, DatasetView

REQUIRED_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def run_technical_checks(view: DatasetView, *, timestamp_tol: float = 1e-3, stats_rel_tol: float = 1e-3,
                         min_episode_frames: int = 1, check_videos: bool = True) -> list[Check]:
    checks: list[Check] = []
    info, data = view.info, view.data

    checks.append(Check("codebase_version", PASS if view.major == "v3" else WARN,
                        f"{view.version}" + ("" if view.major == "v3" else
                                             " (legacy: current LeRobot needs conversion to v3.0 first)"),
                        {"version": view.version}))
    checks.append(Check("fps", PASS if view.fps > 0 else FAIL, f"info.fps = {view.fps:g}", {"fps": view.fps}))

    if view.hub is not None:
        h = view.hub
        if h.main_sha and h.codebase_tag_sha and h.main_sha != h.codebase_tag_sha:
            checks.append(Check("hub_tag_matches_main", WARN,
                                "the codebase-version tag lerobot-train downloads differs from main: training sees "
                                "an older snapshot unless --dataset.revision is pinned",
                                {"main": h.main_sha, "tag": h.codebase_tag_sha}))
        elif h.main_sha and h.codebase_tag_sha:
            checks.append(Check("hub_tag_matches_main", PASS, "codebase-version tag == main",
                                {"sha": h.main_sha}))
        else:
            checks.append(Check("hub_tag_matches_main", UNKNOWN, h.error or "refs not available (offline?)"))

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    checks.append(Check("required_columns", FAIL if missing_cols else PASS,
                        f"missing {missing_cols}" if missing_cols else "all index/timestamp columns present",
                        {"missing": missing_cols}))

    # ---------------------------------------------------------------- feature schema
    schema_problems = []
    for key, ft in view.features.items():
        if ft.get("dtype") in ("video", "image"):
            continue
        if key not in data.columns:
            schema_problems.append(f"{key}: declared in info.json, absent from parquet")
            continue
        shape = ft.get("shape") or [1]
        expect = int(np.prod(shape))
        sample = data[key].iloc[0] if len(data) else None
        got = int(np.size(sample)) if sample is not None else expect
        if got != expect:
            schema_problems.append(f"{key}: info shape {shape} but parquet rows have {got} values")
        names = ft.get("names")
        if isinstance(names, list) and len(names) and expect > 1 and len(names) != expect:
            schema_problems.append(f"{key}: {len(names)} names for {expect} values")
    for key in VECTOR_KEYS:
        if key not in view.features:
            schema_problems.append(f"{key}: not a feature of this dataset")
    checks.append(Check("feature_schema", FAIL if schema_problems else PASS,
                        "; ".join(schema_problems) or "info.json features match parquet columns and shapes",
                        {"problems": schema_problems,
                         "dims": {k: view.dim(k) for k in VECTOR_KEYS if k in view.features}}))

    # ---------------------------------------------------------------- finite values
    nonfinite = {}
    for key in VECTOR_KEYS:
        if key in data.columns and len(data):
            arr = np.stack(data[key].values).astype(np.float64)
            bad = ~np.isfinite(arr)
            if bad.any():
                rows = np.where(bad.any(1))[0]
                nonfinite[key] = {"values": int(bad.sum()),
                                  "episodes": sorted(map(int, data["episode_index"].iloc[rows].unique()))}
    checks.append(Check("finite_state_action", FAIL if nonfinite else PASS,
                        f"non-finite values: {nonfinite}" if nonfinite else "no NaN/inf in state or action",
                        nonfinite))

    # ---------------------------------------------------------------- episodes, frames, timing
    if len(view.episodes) and "length" in view.episodes.columns:
        meta_len = view.episodes["length"].astype(int).to_dict()
    else:
        meta_len = {}
    data_len = data.groupby("episode_index").size().astype(int).to_dict() if len(data) else {}
    all_eps = sorted(set(meta_len) | set(data_len))
    empty = [e for e in all_eps if data_len.get(e, 0) < min_episode_frames or meta_len.get(e, 1) == 0]
    mismatch = {e: (meta_len.get(e), data_len.get(e)) for e in all_eps
                if e in meta_len and meta_len[e] != data_len.get(e, 0)}
    checks.append(Check("empty_episodes", FAIL if empty else PASS,
                        f"episodes with < {min_episode_frames} frames: {empty}" if empty else "no empty episodes",
                        {"episodes": empty}))
    checks.append(Check("episode_lengths_match_metadata", FAIL if mismatch else (PASS if meta_len else UNKNOWN),
                        f"metadata vs parquet length differs: {mismatch}" if mismatch else
                        ("episode metadata lengths match parquet" if meta_len else "no per-episode metadata"),
                        {"mismatch": {int(k): v for k, v in mismatch.items()}}))
    if info.get("total_episodes") is not None and data_len:
        n_info, n_data = int(info["total_episodes"]), len(data_len)
        if view.episode_filter is None:
            checks.append(Check("total_episodes", PASS if n_info == n_data else FAIL,
                                f"info.json {n_info}, parquet {n_data}", {"info": n_info, "data": n_data}))

    gaps, ts_bad, nonzero_start = {}, {}, {}
    if len(data) and view.fps > 0:
        dt_expect = 1.0 / view.fps
        for e, g in data.groupby("episode_index"):
            fi = g["frame_index"].to_numpy()
            if len(fi) and not np.array_equal(fi, np.arange(fi[0], fi[0] + len(fi))):
                gaps[int(e)] = int(len(np.setdiff1d(np.arange(fi.min(), fi.max() + 1), fi)) or -1)
            if len(fi) and fi[0] != 0:
                nonzero_start[int(e)] = int(fi[0])
            ts = g["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in g else None
            if ts is not None and len(ts) > 1:
                d = np.diff(ts)
                err = float(np.max(np.abs(d - dt_expect)))
                if err > timestamp_tol or (d <= 0).any():
                    ts_bad[int(e)] = {"max_dt_error_s": round(err, 6), "median_dt": float(np.median(d)),
                                      "non_monotonic": bool((d <= 0).any())}
    checks.append(Check("missing_frames", FAIL if gaps or nonzero_start else PASS,
                        (f"frame_index gaps in {sorted(gaps)}; non-zero start in {sorted(nonzero_start)}"
                         if gaps or nonzero_start else "frame_index contiguous from 0 in every episode"),
                        {"gaps": gaps, "nonzero_start": nonzero_start}))
    checks.append(Check("timestamps", FAIL if ts_bad else PASS,
                        f"timestamps off the 1/fps grid (> {timestamp_tol}s) in {sorted(ts_bad)}" if ts_bad else
                        f"timestamps on a 1/{view.fps:g} s grid, monotonic",
                        {"episodes": ts_bad,
                         "note": "timestamps are written as frame_index/fps by lerobot-record, so a wrong control "
                                 "rate is not visible here; see the real/sim joint-speed comparison"}))
    if data_len and view.fps:
        lengths = np.array(list(data_len.values()), dtype=float)
        checks.append(Check("episode_durations", PASS,
                            f"{len(lengths)} episodes, {lengths.min() / view.fps:.1f}-{lengths.max() / view.fps:.1f} s "
                            f"(mean {lengths.mean() / view.fps:.1f} s)",
                            {"min_frames": int(lengths.min()), "max_frames": int(lengths.max()),
                             "mean_frames": float(lengths.mean())}))

    # ---------------------------------------------------------------- cameras
    checks.extend(_camera_checks(view, data_len, check_videos))

    # ---------------------------------------------------------------- normalization statistics
    checks.extend(_stats_checks(view, stats_rel_tol))
    return checks


def _camera_checks(view: DatasetView, data_len: dict, check_videos: bool) -> list[Check]:
    out = []
    cams = view.camera_keys
    if not cams:
        return [Check("camera_streams", WARN, "dataset declares no camera features", {})]
    problems, fps_mismatch = [], {}
    for cam in cams:
        ft = view.features[cam]
        vfps = (ft.get("info") or {}).get("video.fps")
        if vfps is not None and abs(float(vfps) - view.fps) > 1e-6:
            fps_mismatch[cam] = vfps
        if ft.get("dtype") != "video":
            continue
        if view.major == "v3":
            cols = [f"videos/{cam}/chunk_index", f"videos/{cam}/file_index", f"videos/{cam}/from_timestamp",
                    f"videos/{cam}/to_timestamp"]
            missing_cols = [c for c in cols if c not in view.episodes.columns]
            if missing_cols:
                problems.append(f"{cam}: episode metadata lacks {missing_cols}")
                continue
            for e, row in view.episodes.iterrows():
                n_vid = (float(row[cols[3]]) - float(row[cols[2]])) * view.fps
                if e in data_len and abs(n_vid - data_len[e]) > 1.5:
                    problems.append(f"{cam} ep{e}: video span {n_vid:.1f} frames vs {data_len[e]} data frames")
        if check_videos:
            missing_files = set()
            for e in data_len:
                try:
                    rel, _ = view.video_path(cam, e)
                except (KeyError, IndexError, ValueError):
                    missing_files.add(f"{cam} ep{e}: no video location")
                    continue
                if not (view.root / rel).exists() and view.repo_id is None:
                    missing_files.add(rel)
            if missing_files:
                problems.append(f"{cam}: missing video files {sorted(missing_files)[:5]}"
                                + (" ..." if len(missing_files) > 5 else ""))
    out.append(Check("camera_streams", FAIL if problems else PASS,
                     "; ".join(problems[:8]) or f"{len(cams)} camera streams with complete episode coverage",
                     {"cameras": cams, "problems": problems,
                      "shapes": {c: view.features[c].get("shape") for c in cams},
                      "remote_videos_not_checked": view.repo_id is not None}))
    out.append(Check("camera_fps", WARN if fps_mismatch else PASS,
                     f"video fps differs from dataset fps: {fps_mismatch}" if fps_mismatch else "video fps == fps",
                     fps_mismatch))
    return out


def _stats_checks(view: DatasetView, rel_tol: float) -> list[Check]:
    out = []
    if not view.stats:
        return [Check("normalization_stats", FAIL, "no meta/stats.json (or v2.1 episodes_stats) to normalize with", {})]
    for key in VECTOR_KEYS:
        if key not in view.data.columns:
            continue
        st = view.stats.get(key)
        if st is None or "mean" not in st or "std" not in st:
            out.append(Check(f"stats_{key}", FAIL, f"stats.json has no mean/std for {key}", {}))
            continue
        mean, std = np.asarray(st["mean"], float).ravel(), np.asarray(st["std"], float).ravel()
        if mean.shape[0] != view.dim(key):
            out.append(Check(f"stats_{key}", FAIL, f"stats dim {mean.shape[0]} != feature dim {view.dim(key)}", {}))
            continue
        arr = np.stack(view.data[key].values).astype(np.float64)
        arr = arr[np.isfinite(arr).all(1)]
        if not len(arr):
            out.append(Check(f"stats_{key}", UNKNOWN, "no finite rows to recompute statistics from", {}))
            continue
        m2, s2 = arr.mean(0), arr.std(0)
        scale = np.maximum(s2, 1e-6)
        dmean = np.abs(m2 - mean) / scale
        dstd = np.abs(s2 - std) / scale
        stale = (dmean > rel_tol * 10) | (dstd > max(rel_tol * 10, 0.01))
        constant = np.where(s2 < 1e-6)[0]
        names = view.names(key)
        ev = {"max_mean_diff_in_std": float(dmean.max()), "max_std_rel_diff": float(dstd.max()),
              "constant_dims": [names[i] for i in constant],
              "count_in_stats": (np.asarray(st.get("count", [np.nan]), float).ravel()[0] if "count" in st else None),
              "frames": int(len(arr))}
        if view.episode_filter is not None:
            out.append(Check(f"stats_{key}", UNKNOWN, "episode subset loaded; stats.json describes the full dataset",
                             ev))
        elif stale.any():
            out.append(Check(f"stats_{key}", FAIL,
                             f"stats.json disagrees with the data in {[names[i] for i in np.where(stale)[0]]} "
                             "(stale stats: the normalizer would use the wrong mean/std)", ev))
        elif len(constant):
            out.append(Check(f"stats_{key}", WARN, f"constant dimensions {ev['constant_dims']}: std~0, any "
                                                    "deployment deviation is amplified by 1/eps", ev))
        else:
            out.append(Check(f"stats_{key}", PASS, "stats.json mean/std match a recomputation from parquet", ev))
    return out
