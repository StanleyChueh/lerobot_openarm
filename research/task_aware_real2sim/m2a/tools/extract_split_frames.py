"""Pull (camera frame, joint state) pairs for an explicit, frozen episode split (camera-calibration input).

Same output layout as IsaacLab scripts/tools/sysid/extract_real_frames.py (frames.npz: images, state, names,
episode, frame) plus ``split`` (per-frame set name), so render_real_states.py and segment_robot_real.py consume
it unchanged. Differences: episodes come from split.json instead of being evenly spaced over the dataset, and
the video is decoded once front to back (deterministic, no seeking).

Frames per episode are evenly spaced over [0.1, 0.9] of its length; each target moves to the nearest frame whose
fastest arm joint is below --max_speed (state and video are stamped by different devices, so fast motion turns a
small clock offset into a silhouette error). Plain python (pandas, pyarrow, av, numpy):

  python extract_split_frames.py <dataset root> <out dir> --camera wrist_cam --split split.json \
      --sets camera_cal_train camera_cal_val --per_episode 12 --max_speed 0.25
"""

import argparse
import json
import os

import av
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--per_episode", type=int, default=12)
    ap.add_argument("--max_speed", type=float, default=0.25)
    args = ap.parse_args()

    split = json.load(open(args.split))
    info = json.load(open(os.path.join(args.root, "meta/info.json")))
    fps, names = info["fps"], info["features"]["observation.state"]["names"]
    key = f"observation.images.{args.camera}"
    data = pd.concat(pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(args.root, "data"))
                     for f in sorted(fs) if f.endswith(".parquet"))
    eps = pd.concat(pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(args.root, "meta/episodes"))
                    for f in sorted(fs) if f.endswith(".parquet")).set_index("episode_index")
    arm = [i for i, n in enumerate(names) if not n.endswith("8.pos")]

    picks = []  # (video, timestamp in file, set, episode, frame, state)
    for set_name in args.sets:
        for ep in split[set_name]:
            rows = data[data.episode_index == ep].sort_values("frame_index")
            state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
            speed = np.abs(np.gradient(state[:, arm], axis=0)).max(1) * fps
            meta = eps.loc[ep]
            video = os.path.join(args.root, info["video_path"].format(
                video_key=key, chunk_index=int(meta[f"videos/{key}/chunk_index"]), file_index=int(meta[f"videos/{key}/file_index"])))
            used = set()
            for target in np.linspace(0.1, 0.9, args.per_episode) * (len(rows) - 1):
                order = np.argsort(np.abs(np.arange(len(rows)) - target), kind="stable")
                slow = [i for i in order[:30] if speed[i] <= args.max_speed and i not in used]
                if slow:
                    i = int(slow[0])
                    used.add(i)
                    picks.append((video, float(meta[f"videos/{key}/from_timestamp"]) + i / fps, set_name, int(ep), i, state[i]))

    images = {}
    for video in sorted({p[0] for p in picks}):
        wanted = sorted(p[1] for p in picks if p[0] == video)
        with av.open(video) as container:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                for w in wanted:
                    if (video, w) not in images and abs(t - w) < 0.5 / fps:
                        images[(video, w)] = frame.to_ndarray(format="rgb24")
                if t > wanted[-1] + 1.0:
                    break
    picks = [p for p in picks if (p[0], p[1]) in images]
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, "frames.npz"),
        images=np.stack([images[(p[0], p[1])] for p in picks]), state=np.stack([p[5] for p in picks]),
        names=np.array(names), episode=np.array([p[3] for p in picks]), frame=np.array([p[4] for p in picks]),
        split=np.array([p[2] for p in picks]),
    )
    counts = {s: sum(p[2] == s for p in picks) for s in args.sets}
    print(f"{len(picks)} frames {counts} -> {args.out}/frames.npz")


if __name__ == "__main__":
    main()
