"""Object and target position coverage: 10 sources vs the published synthetic dataset vs the unified (sfagen) dataset,
measured against the OTHER real recordings of the same task (never used for generation). All positions by forward
kinematics at gripper command events (no annotation):
  object position = TCP of the grasping arm at its first close (Pringles can: right; battery: right; station: left)
  target          = Pringles: right TCP at the left close (hand-over point); battery: right TCP at the first right open
                    (insertion point)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
S = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(S))  # tracked sfagen
from sfagen.robot import Robot
import yaml
X = Path("/home/stanley/Stanley_ws/sfa_hf_export"); W = S / "configs/work"  # X: old data roots only
rob = Robot(yaml.safe_load(open(S / "configs/_openarm_robot.yaml")))
SETS = {"pringles": {"source": Path.home() / ".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/cebdee98115ab6827a4ec2655335c536b5dd085b/data",
                     "published": X / "3cam_v4/dataset/data", "unified": W / "openarm_pringles_handover/dataset/data",
                     "pool": Path.home() / ".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d/data",
                     "pool_skip": list(range(10))},
        "battery": {"source": X / "battery/src/data", "published": X / "battery_v01/dataset/data", "unified": W / "openarm_battery_insertion/dataset/data",
                    "pool": X / "battery/pool/data", "pool_skip": [15, 20, 21, 22, 23, 30, 32, 36, 45, 53]}}


def positions(path, task, skip=()):
    df = pd.read_parquet(sorted(Path(path).rglob("*.parquet"))); out = []
    for e, d in df.groupby("episode_index"):
        if int(e) in skip:
            continue
        d = d.sort_values("frame_index"); S = np.stack(d["observation.state"].to_numpy()).astype(float); A = np.stack(d["action"].to_numpy()).astype(float)
        ev = rob.gripper_events(A)
        try:
            TR = rob.tcp_from_state(S, "R")[:, :3, 3]; TL = rob.tcp_from_state(S, "L")[:, :3, 3]
            if task == "pringles":
                r = {"can": TR[ev["R"]["closes"][0]], "handover": TR[[c for c in ev["L"]["closes"] if c > ev["R"]["closes"][0]][0]]}
            else:
                r = {"battery": TR[ev["R"]["closes"][0]], "station": TL[ev["L"]["closes"][0]], "insertion": TR[ev["R"]["opens"][0]]}
            out.append({k: v.tolist() for k, v in r.items()})
        except (IndexError, KeyError):
            pass
    return out


def hull(P):
    return float(ConvexHull(P).volume) if len(np.unique(np.round(P, 3), axis=0)) >= 3 else 0.0


res = {}; figs = []
for task, c in SETS.items():
    data = {k: positions(c[k], task, c["pool_skip"] if k == "pool" else ()) for k in ("source", "published", "unified", "pool")}
    keys = ["can", "handover"] if task == "pringles" else ["battery", "station", "insertion"]
    for key in keys:
        S = 100 * np.array([d[key] for d in data["source"]])[:, :2]; P = 100 * np.array([d[key] for d in data["pool"]])[:, :2]
        nn = lambda A, B: np.array([np.min(np.linalg.norm(B - a, axis=1)) for a in A])  # noqa: E731
        row = {"n_other_real": len(P), "other_real_hull_cm2": hull(P),
               "source": {"n": len(S), "hull_cm2": hull(S), "within_2cm": int((nn(P, S) <= 2).sum()), "median_nn_cm": float(np.median(nn(P, S)))}}
        for rv in ("published", "unified"):
            Y = 100 * np.array([d[key] for d in data[rv]])[:, :2]; Z = np.r_[S, Y]
            row[rv] = {"n": len(Y), "hull_cm2": hull(Z), "within_2cm": int((nn(P, Z) <= 2).sum()), "median_nn_cm": float(np.median(nn(P, Z))),
                       "new_positions_gt_1cm": int((nn(Y, S) > 1.0).sum())}
        res[f"{task}/{key}"] = row
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
        for ax, (name, Y) in zip(axs, (("10 real sources", None), (f"+ published ({len(data['published'])})", 100 * np.array([d[key] for d in data["published"]])[:, :2]),
                                        (f"+ unified sfagen ({len(data['unified'])})", 100 * np.array([d[key] for d in data["unified"]])[:, :2]))):
            ax.scatter(P[:, 0], P[:, 1], s=16, facecolor="none", edgecolor="#8f8e8a", lw=0.9, label=f"{len(P)} other real recordings")
            if Y is not None:
                ax.scatter(Y[:, 0], Y[:, 1], s=16, color="#eb6834", edgecolor="#fcfcfb", lw=0.8, zorder=4, label="synthetic")
            ax.scatter(S[:, 0], S[:, 1], s=30, color="#2a78d6", edgecolor="#fcfcfb", lw=1.2, zorder=5, label="source")
            ax.set_title(name, fontsize=10); ax.set_aspect("equal"); ax.grid(color="#e4e3df", lw=0.5); ax.set_xlabel("world x (cm)"); ax.set_ylabel("world y (cm)"); ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"{task}: {key} position (top-down)", fontweight="bold"); fig.savefig(S / f"logs/coverage_{task}_{key}.png", dpi=130); plt.close(fig)
(S / "logs/coverage.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print(k, "| other real", v["n_other_real"], "hull", round(v["other_real_hull_cm2"]), "|", " | ".join(f"{s}: n={v[s]['n']} hull={v[s]['hull_cm2']:.0f} within2cm={v[s]['within_2cm']} med={v[s]['median_nn_cm']:.2f}" + (f" new={v[s]['new_positions_gt_1cm']}" if 'new_positions_gt_1cm' in v[s] else "") for s in ("source", "published", "unified")))
