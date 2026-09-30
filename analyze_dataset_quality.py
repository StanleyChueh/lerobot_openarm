#!/usr/bin/env python
"""Per-episode quality audit for OpenArm LeRobot (v3.0) handover datasets.

Reads only meta/ and data/ parquet files (no video unless --frames is given), so a
400-episode dataset is analysed in seconds. Pass several datasets to compare them
side by side (e.g. a policy that works vs one that doesn't).

What it checks, per episode:
  Gripper behaviour (open value is auto-detected, ~0.044; closing on the can stalls ~0.036)
    Hold level varies a lot between episodes (~0.018-0.038), so a close only counts as an
    air grab when it goes below --empty-thresh (the gripper nearly shuts on nothing).
    The right "main hold" is its longest real hold; the left one is the hold that lasts to the end.
    - right_pre_grasp_close    right gripper closed (usually on air) before the main grasp
    - right_regrasp            right gripper grasped the can more than once (lost it / re-tried)
    - right_post_release_close right gripper closed again (even briefly) after handing the can over
    - left_early_close         left gripper closed before its handover grasp
    - left_abnormal            left gripper does not end the episode holding the can
    - gripper_twitch           partial dips / short blips that never became a real close
    - hold_slip                gripper value keeps drifting after the grasp settled (slip / squeeze)
  Handover
    - handover_overlap   frames where BOTH grippers hold the can (right release - left grasp)
    - drop_risk          right released before / right as left closed (overlap < --min-overlap)
    - no_handover        left never ends up holding, or right never releases
  Timing
    - grasp_frame, handover_frame (absolute and normalised to episode length)
    - timing_outlier     grasp or handover time is a robust (MAD) outlier
  Motion
    - action_spike       largest per-frame arm jump is a robust outlier for this dataset
    - start_dev          distance of the start pose from the dataset median start pose
    - lead/trail idle, longest mid-episode pause, action spikes, action->next-state
      tracking error (sim controller failing to follow = collision / IK trouble)
    - nn_dist            distance to the most similar other episode (diversity / duplicates)

Outputs (in --out/<dataset name>/): episodes.csv, summary.json, exclude_episodes.json,
plots, and optionally frame grids of flagged episodes. A comparison table is printed
and saved to --out/comparison.csv when more than one dataset is given.

Usage:
  python analyze_dataset_quality.py --repo-id ethanCSL/openarm_pringles_v0
  python analyze_dataset_quality.py \\
      --repo-id ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz ethanCSL/openarm_pringles_v0 \\
      --labels A_good B_poor --out outputs/dataset_audit --frames 12
  python analyze_dataset_quality.py --root /path/to/local/dataset
"""

import argparse
import glob
import json
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLAGS = [
    "right_pre_grasp_close",
    "right_regrasp",
    "right_post_release_close",
    "left_early_close",
    "left_abnormal",
    "gripper_twitch",
    "hold_slip",
    "drop_risk",
    "no_handover",
    "timing_outlier",
    "start_outlier",
    "long_pause",
    "action_spike",
    "tracking_error",
    "near_duplicate",
    "length_outlier",
]
# Flags severe enough that the episode should probably be regenerated or dropped.
EXCLUDE_FLAGS = {"right_regrasp", "right_post_release_close", "left_early_close", "left_abnormal", "drop_risk", "no_handover", "tracking_error"}


# ----------------------------------------------------------------------------- loading


def read_parquet(path):
    # fastparquet (pandas' fallback engine) chokes on LeRobot's dotted column names like
    # "observation.state", so insist on pyarrow.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("pyarrow is required: run this with the lerobot venv python, or `pip install pyarrow`") from None
    return pd.read_parquet(path, engine="pyarrow")


def load_dataset(repo_id: str | None, root: str | None, revision: str | None):
    if root is None:
        from huggingface_hub import snapshot_download

        root = snapshot_download(repo_id, repo_type="dataset", revision=revision, allow_patterns=["meta/*", "data/*"])
    info = json.load(open(os.path.join(root, "meta", "info.json")))
    data_files = sorted(glob.glob(os.path.join(root, "data", "chunk-*", "*.parquet")))
    df = pd.concat([read_parquet(f) for f in data_files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    ep_files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "*.parquet")))
    eps_meta = pd.concat([read_parquet(f) for f in ep_files], ignore_index=True).set_index("episode_index")
    eps_meta.attrs["fps"] = info.get("fps", 30)
    return root, info, df, eps_meta


# ----------------------------------------------------------------------------- gripper segmentation


def gripper_segments(x, open_val, open_tol, close_delta, min_hold):
    """Hysteresis segmentation of a gripper trace.

    Returns closed segments [(start, end_exclusive, min, median)] and the number of
    twitches (dips out of the open band that reopen without ever counting as closed).
    """
    open_thr = open_val - open_tol
    close_thr = open_val - close_delta
    segs, twitches = [], 0
    state, start, dipped = "open", None, False
    for i, v in enumerate(x):
        if state == "open":
            if v < close_thr:
                state, start, dipped = "closed", i, False
            elif v < open_thr:
                dipped = True
            elif dipped:
                twitches += 1
                dipped = False
        elif v >= open_thr:
            segs.append((start, i))
            state = "open"
    if state == "closed":
        segs.append((start, len(x)))
    if state == "open" and dipped:
        twitches += 1
    out = []
    for s, e in segs:
        seg = x[s:e]
        out.append({"start": s, "end": e, "len": e - s, "min": float(seg.min()), "median": float(np.median(seg)), "blip": (e - s) < min_hold})
    return out, twitches


def robust_z(v):
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    return (v - med) / (mad if mad > 1e-9 else 1e-9)


# ----------------------------------------------------------------------------- per-episode analysis


def analyze(df, info, args):
    names = info["features"]["action"]["names"]
    idx = {n: i for i, n in enumerate(names)}
    rg, lg = idx[args.right_gripper], idx[args.left_gripper]
    right_arm = [i for n, i in idx.items() if n.startswith(args.right_prefix) and i != rg]
    left_arm = [i for n, i in idx.items() if n.startswith(args.left_prefix) and i != lg]
    arm = right_arm + left_arm

    episodes = {e: g for e, g in df.groupby("episode_index")}
    act = {e: np.stack(g["action"].values).astype(np.float64) for e, g in episodes.items()}
    st = {e: np.stack(g["observation.state"].values).astype(np.float64) for e, g in episodes.items()}

    first = np.stack([a[0] for a in act.values()])
    open_val = float(np.median(np.concatenate([first[:, rg], first[:, lg]])))
    start_med = np.median(np.stack([s[0] for s in st.values()]), axis=0)

    rows, traces = [], {}
    for e, a in act.items():
        s = st[e]
        n = len(a)
        r = {"episode_index": int(e), "length": n}

        rseg, rtw = gripper_segments(a[:, rg], open_val, args.open_tol, args.close_delta, args.min_hold)
        lseg, ltw = gripper_segments(a[:, lg], open_val, args.open_tol, args.close_delta, args.min_hold)
        is_hold = lambda g: not g["blip"] and g["min"] >= args.empty_thresh  # noqa: E731
        rhold = [g for g in rseg if is_hold(g)]
        rmain = max(rhold, key=lambda g: g["len"]) if rhold else None
        lmain = lseg[-1] if lseg and lseg[-1]["end"] == n and is_hold(lseg[-1]) else None

        r["right_closes"] = len(rseg)
        r["right_blips"] = sum(g["blip"] for g in rseg)
        r["right_empty_closes"] = sum(g["min"] < args.empty_thresh for g in rseg)
        r["left_closes"] = len(lseg)
        r["left_blips"] = sum(g["blip"] for g in lseg)
        r["left_empty_closes"] = sum(g["min"] < args.empty_thresh for g in lseg)
        r["right_twitches"], r["left_twitches"] = rtw, ltw
        r["right_holds"] = len(rhold)
        r["right_pre_closes"] = sum(g["end"] <= rmain["start"] and not is_hold(g) for g in rseg) if rmain else 0
        # even short snaps count here: a close right after letting go is a learned habit, not noise
        r["right_post_closes"] = sum(g["start"] >= rmain["end"] for g in rseg) if rmain else 0
        r["left_early_closes"] = sum(not g["blip"] for g in lseg if g is not lmain)

        grasp = rhold[0]["start"] if rhold else np.nan
        right_release = rmain["end"] if rmain else np.nan
        left_grasp = lmain["start"] if lmain else np.nan
        left_holds_at_end = lmain is not None
        right_open_at_end = not rseg or rseg[-1]["end"] < n
        r["grasp_frame"] = grasp
        r["grasp_t"] = grasp / n
        r["left_grasp_frame"] = left_grasp
        r["right_release_frame"] = right_release
        r["handover_frame"] = left_grasp
        r["handover_t"] = left_grasp / n
        r["handover_overlap"] = right_release - left_grasp
        settled = a[rmain["start"] + args.settle : rmain["end"], rg] if rmain else np.array([])
        r["hold_drift"] = float(np.ptp(settled)) if len(settled) > 1 else np.nan
        r["right_hold_value"] = rmain["median"] if rmain else np.nan
        r["left_hold_value"] = lmain["median"] if lmain else np.nan
        r["left_holds_at_end"] = left_holds_at_end
        r["right_open_at_end"] = right_open_at_end

        # motion
        d = np.abs(np.diff(a[:, arm], axis=0))
        static = d.max(axis=1) < args.static_eps
        lead = int(np.argmax(~static)) if (~static).any() else n
        trail = int(np.argmax(~static[::-1])) if (~static).any() else n
        mid = static[lead : len(static) - trail]
        longest, cur = 0, 0
        for v in mid:
            cur = cur + 1 if v else 0
            longest = max(longest, cur)
        r["lead_idle"], r["trail_idle"], r["longest_pause"] = lead, trail, longest
        r["max_step"] = float(d.max()) if len(d) else 0.0
        r["jerk_rms"] = float(np.sqrt(np.mean(np.diff(a[:, arm], n=3, axis=0) ** 2))) if n > 3 else 0.0
        r["track_err"] = float(np.abs(a[:-1, arm] - s[1:, arm]).mean())
        r["start_jump"] = float(np.abs(a[0, arm] - s[0, arm]).max())
        r["start_dev"] = float(np.linalg.norm(s[0, arm] - start_med[arm]))
        r["start_dev_right"] = float(np.linalg.norm(s[0, right_arm] - start_med[right_arm]))
        r["path_right"] = float(np.abs(np.diff(a[:, right_arm], axis=0)).sum())
        r["path_left"] = float(np.abs(np.diff(a[:, left_arm], axis=0)).sum())
        rows.append(r)
        traces[e] = (a[:, rg], a[:, lg])

    ep = pd.DataFrame(rows).set_index("episode_index")

    # nearest-neighbour trajectory distance (arm actions resampled to 64 points)
    grid = np.linspace(0, 1, 64)
    feats = np.stack(
        [np.stack([np.interp(grid, np.linspace(0, 1, len(act[e])), act[e][:, j]) for j in arm], 1).ravel() for e in ep.index]
    )
    feats /= np.sqrt(len(grid))
    sq = (feats**2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * feats @ feats.T, 0))
    np.fill_diagonal(dist, np.inf)
    ep["nn_dist"] = dist.min(1)
    ep["nn_episode"] = ep.index.values[dist.argmin(1)]

    # flags
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tz = np.nan_to_num(np.nanmax(np.abs(np.stack([robust_z(ep["grasp_t"]), robust_z(ep["handover_t"])])), axis=0))
    ep["right_pre_grasp_close"] = ep["right_pre_closes"] > 0
    ep["right_regrasp"] = ep["right_holds"] > 1
    ep["right_post_release_close"] = ep["right_post_closes"] > 0
    ep["left_early_close"] = ep["left_early_closes"] > 0
    ep["left_abnormal"] = ~ep["left_holds_at_end"]
    ep["gripper_twitch"] = (ep["right_twitches"] + ep["left_twitches"] + ep["right_blips"] + ep["left_blips"]) > 0
    ep["hold_slip"] = ep["hold_drift"] > args.slip_thresh
    ep["no_handover"] = ep["right_release_frame"].isna() | ep["left_grasp_frame"].isna() | ~ep["left_holds_at_end"] | ~ep["right_open_at_end"]
    ep["drop_risk"] = ~ep["no_handover"] & (ep["handover_overlap"] < args.min_overlap)
    ep["timing_outlier"] = tz > args.z_thresh
    ep["start_outlier"] = np.abs(robust_z(ep["start_dev"])) > args.z_thresh * 2
    ep["long_pause"] = ep["longest_pause"] > args.pause_frames
    ep["action_spike"] = (robust_z(ep["max_step"]) > args.z_thresh) & (ep["max_step"] > args.spike_floor)
    ep["tracking_error"] = ep["track_err"] > args.track_mult * ep["track_err"].median()
    ep["near_duplicate"] = ep["nn_dist"] < args.dup_thresh
    ep["length_outlier"] = np.abs(robust_z(ep["length"])) > args.z_thresh
    ep["n_flags"] = ep[FLAGS].sum(1)
    ep["exclude"] = ep[list(EXCLUDE_FLAGS)].any(axis=1)
    ep["reasons"] = ep[FLAGS].apply(lambda row: ",".join(f for f in FLAGS if row[f]), axis=1)

    ctx = {"open_val": open_val, "right_arm": [names[i] for i in right_arm], "left_arm": [names[i] for i in left_arm],
           "start_std_right": np.stack([s[0, right_arm] for s in st.values()]).std(0),
           "start_std_left": np.stack([s[0, left_arm] for s in st.values()]).std(0),
           "grasp_pose_std": np.stack([st[e][int(g), right_arm] for e, g in ep["grasp_frame"].dropna().items()]).std(0)}
    return ep, traces, ctx


def summarize(ep, ctx, info):
    n = len(ep)
    pct = lambda m: round(100.0 * m.sum() / n, 1)  # noqa: E731
    s = {
        "episodes": n,
        "total_frames": int(ep["length"].sum()),
        "fps": info.get("fps"),
        "length_mean": round(ep["length"].mean(), 1),
        "length_std": round(ep["length"].std(), 1),
        "gripper_open_value": round(ctx["open_val"], 4),
        "right_hold_value_median": round(ep["right_hold_value"].median(), 4),
        "right_hold_value_std": round(ep["right_hold_value"].std(), 4),
        "left_hold_value_std": round(ep["left_hold_value"].std(), 4),
        "max_step_median": round(ep["max_step"].median(), 3),
        "grasp_frame_mean": round(ep["grasp_frame"].mean(), 1),
        "grasp_frame_std": round(ep["grasp_frame"].std(), 1),
        "grasp_t_std": round(ep["grasp_t"].std(), 3),
        "handover_frame_std": round(ep["handover_frame"].std(), 1),
        "handover_overlap_median": round(ep["handover_overlap"].median(), 1),
        "start_pose_std_right_mean": round(float(ctx["start_std_right"].mean()), 4),
        "start_pose_std_left_mean": round(float(ctx["start_std_left"].mean()), 4),
        "grasp_pose_std_right_mean": round(float(ctx["grasp_pose_std"].mean()), 3),
        "path_right_mean": round(ep["path_right"].mean(), 2),
        "path_left_mean": round(ep["path_left"].mean(), 2),
        "track_err_median": round(ep["track_err"].median(), 5),
        "jerk_rms_median": round(ep["jerk_rms"].median(), 5),
        "nn_dist_median": round(ep["nn_dist"].median(), 3),
        "lead_idle_mean": round(ep["lead_idle"].mean(), 1),
        "trail_idle_mean": round(ep["trail_idle"].mean(), 1),
    }
    for f in FLAGS:
        s[f"{f}_pct"] = pct(ep[f])
    s["exclude_pct"] = pct(ep["exclude"])
    s["clean_pct"] = pct(ep["n_flags"] == 0)
    return s


# ----------------------------------------------------------------------------- plots / frames


def plot_traces(traces, ep, label, path, max_eps=400):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    order = list(ep.index[: max_eps])
    for flagged_pass in (False, True):
        for e in order:
            bad = ep.loc[e, "exclude"]
            if bad != flagged_pass:
                continue
            r, l = traces[e]
            t = np.linspace(0, 1, len(r))
            kw = dict(color="tab:red", alpha=0.55, lw=0.9) if bad else dict(color="0.6", alpha=0.15, lw=0.7)
            axes[0].plot(t, r, **kw)
            axes[1].plot(t, l, **kw)
    axes[0].set_ylabel("right gripper")
    axes[1].set_ylabel("left gripper")
    axes[1].set_xlabel("normalised episode time")
    axes[0].set_title(f"{label}: gripper commands (red = excluded, {int(ep['exclude'].sum())}/{len(ep)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_distributions(results, path):
    cols = [("length", "episode length (frames)"), ("grasp_t", "right grasp time (norm)"),
            ("handover_t", "handover time (norm)"), ("handover_overlap", "both-hold overlap (frames)"),
            ("start_dev", "start pose deviation (rad)"), ("nn_dist", "nearest-episode distance"),
            ("track_err", "action->next state error"), ("right_hold_value", "right gripper hold level"),
            ("path_right", "right arm path length")]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, (c, title) in zip(axes.ravel(), cols):
        allv = np.concatenate([r[1][c].dropna().values for r in results])
        bins = np.linspace(np.nanpercentile(allv, 0.5), np.nanpercentile(allv, 99.5) + 1e-9, 40)
        for label, ep, *_ in results:
            ax.hist(ep[c].dropna().clip(bins[0], bins[-1]), bins=bins, alpha=0.5, label=label, density=True)
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_flag_bars(results, path):
    x = np.arange(len(FLAGS))
    w = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for k, (label, _, summ, *_) in enumerate(results):
        ax.bar(x + k * w, [summ[f"{f}_pct"] for f in FLAGS], w, label=label)
    ax.set_xticks(x + w * (len(results) - 1) / 2, FLAGS, rotation=35, ha="right")
    ax.set_ylabel("% of episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def dump_frames(repo_id, root, revision, eps_meta, ep, camera, n_eps, out_dir):
    """Frame strip (start / right grasp / left grasp / right release / end) for the worst flagged episodes.

    Frames are picked by decoded frame number, not by -ss seeking: timestamp seeking in these
    concatenated LeRobot videos can land in the neighbouring episode.
    """
    from huggingface_hub import hf_hub_download

    key = f"observation.images.{camera}"
    bad = ep[ep["exclude"]].sort_values("n_flags", ascending=False).head(n_eps)
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(old)
    fps = float(eps_meta.attrs.get("fps", 30.0))
    for e, row in bad.iterrows():
        m = eps_meta.loc[e]
        rel = f"videos/{key}/chunk-{int(m[f'videos/{key}/chunk_index']):03d}/file-{int(m[f'videos/{key}/file_index']):03d}.mp4"
        video = os.path.join(root, rel) if os.path.exists(os.path.join(root, rel)) else hf_hub_download(repo_id, rel, repo_type="dataset", revision=revision)
        base = int(round(float(m[f"videos/{key}/from_timestamp"]) * fps))
        events = [0, row["grasp_frame"], row["left_grasp_frame"], row["right_release_frame"], row["length"] - 1]
        frames = sorted({min(int(f), int(row["length"]) - 1) for f in events if not pd.isna(f)})
        select = "+".join(f"eq(n\\,{base + f})" for f in frames)
        vf = (f"drawtext=text='ep{e} f%{{eif\\:n-{base}\\:d}}':x=6:y=6:fontcolor=yellow:fontsize=28,"
              f"select='{select}',scale=320:240,tile={len(frames)}x1")
        out = os.path.join(out_dir, f"ep{e:04d}_{row['reasons'].replace(',', '+')[:80]}.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-vf", vf, "-frames:v", "1", out], check=False)


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", nargs="*", default=[], help="One or more HF dataset repo ids")
    ap.add_argument("--root", nargs="*", default=[], help="One or more local dataset roots (instead of / in addition to --repo-id)")
    ap.add_argument("--labels", nargs="*", default=None, help="Short names for each dataset, in order (repo ids first, then roots)")
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument("--out", type=str, default="outputs/dataset_audit")
    ap.add_argument("--right-gripper", type=str, default="RJ8.pos")
    ap.add_argument("--left-gripper", type=str, default="LJ8.pos")
    ap.add_argument("--right-prefix", type=str, default="RJ")
    ap.add_argument("--left-prefix", type=str, default="LJ")
    ap.add_argument("--open-tol", type=float, default=0.0015, help="Gripper counts as open within this of the open value")
    ap.add_argument("--close-delta", type=float, default=0.004, help="Gripper counts as closed this far below the open value (can stalls ~0.008 below)")
    ap.add_argument("--empty-thresh", type=float, default=0.016, help="A close that goes below this never touched the can (air grab); real holds sit ~0.018-0.038")
    ap.add_argument("--min-hold", type=int, default=10, help="Closed segments shorter than this (frames) are blips, not grasps")
    ap.add_argument("--min-overlap", type=int, default=3, help="Frames both grippers must hold the can during handover")
    ap.add_argument("--settle", type=int, default=15, help="Frames after the grasp ignored by the slip check (gripper tightening in)")
    ap.add_argument("--slip-thresh", type=float, default=0.005, help="Gripper-value drift while holding (after --settle) that counts as slip")
    ap.add_argument("--static-eps", type=float, default=1e-4, help="Per-frame arm motion below this is idle")
    ap.add_argument("--pause-frames", type=int, default=30, help="Mid-episode idle run longer than this is a long pause")
    ap.add_argument("--spike-floor", type=float, default=0.05, help="Per-frame arm jump (rad) below which nothing counts as a spike, whatever its z-score")
    ap.add_argument("--track-mult", type=float, default=3.0, help="Tracking error > this x dataset median is flagged")
    ap.add_argument("--dup-thresh", type=float, default=0.01, help="Nearest-episode distance below this is a near duplicate")
    ap.add_argument("--z-thresh", type=float, default=3.0, help="Robust z-score for timing/length outliers")
    ap.add_argument("--frames", type=int, default=0, help="Dump key-frame grids for this many worst excluded episodes (downloads video)")
    ap.add_argument("--camera", type=str, default="body_cam")
    args = ap.parse_args()

    sources = [(r, None) for r in args.repo_id] + [(None, r) for r in args.root]
    if not sources:
        ap.error("give at least one --repo-id or --root")
    labels = args.labels or [(r or os.path.basename(os.path.normpath(p))).split("/")[-1] for r, p in sources]
    if len(labels) != len(sources):
        ap.error("--labels must have one entry per dataset")

    os.makedirs(args.out, exist_ok=True)
    results = []
    for (repo_id, root), label in zip(sources, labels):
        print(f"\n=== {label} ({repo_id or root})")
        root, info, df, eps_meta = load_dataset(repo_id, root, args.revision)
        ep, traces, ctx = analyze(df, info, args)
        summ = summarize(ep, ctx, info)
        d = os.path.join(args.out, label)
        os.makedirs(d, exist_ok=True)
        ep.to_csv(os.path.join(d, "episodes.csv"))
        json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=2)
        excl = {int(e): r for e, r in ep.loc[ep["exclude"], "reasons"].items()}
        json.dump({"exclude_episodes": sorted(excl), "reasons": excl}, open(os.path.join(d, "exclude_episodes.json"), "w"), indent=2)
        plot_traces(traces, ep, label, os.path.join(d, "gripper_traces.png"))
        print(f"  start-pose std right {np.round(ctx['start_std_right'], 4)}")
        print(f"  grasp-pose std right {np.round(ctx['grasp_pose_std'], 3)}")
        for f in FLAGS:
            print(f"  {f:18s} {summ[f + '_pct']:5.1f}%  ({int(ep[f].sum())})")
        print(f"  -> suggest excluding {len(excl)} episodes, clean (no flags) {summ['clean_pct']}%")
        if args.frames:
            dump_frames(repo_id, root, args.revision, eps_meta, ep, args.camera, args.frames, os.path.join(d, "flagged_frames"))
        results.append((label, ep, summ, ctx))

    plot_distributions(results, os.path.join(args.out, "distributions.png"))
    plot_flag_bars(results, os.path.join(args.out, "flag_rates.png"))
    cmp = pd.DataFrame({label: summ for label, _, summ, _ in results})
    cmp.to_csv(os.path.join(args.out, "comparison.csv"))
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print("\n=== summary\n", cmp)
    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
