"""How much of a wrist camera's image does the robot (excluding the camera's own hand) cover, over EVERY frame of the
split episodes? CPU forward kinematics (IsaacLab sysid urdf_fk.py) + the visual-mesh points sampled by
render_real_states.py, projected through the cfg mount and through randomly perturbed mounts (the true mount is unknown,
so visibility must not hinge on the cfg being right). Identifiability check for silhouette-based wrist calibration.

  python visibility_all_frames.py <poses.npz> --camera wrist_cam --split split.json --out vis.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

SYSID = os.path.expanduser("~/Stanley_ws/IsaacLab/scripts/tools/sysid")
sys.path.insert(0, SYSID)
from real_states import real_state_to_sim  # noqa: E402
from urdf_fk import URDFChain  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_wrist_mount import qmat  # noqa: E402

ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fx", type=float, default=606.0)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--perturb", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    poses = np.load(args.poses)
    names = [str(b) for b in poses["body_names"]]
    parent = str(poses["parent_link"])
    side = "left" if "_left_" in parent else "right"
    chain = URDFChain(os.path.expanduser("~/Stanley_ws/IsaacLab/source/isaaclab_assets/data/v1_camera_isaac/urdf/v1_camera.urdf"))
    calib = json.load(open(os.path.expanduser("~/Stanley_ws/lerobot_openarm/calibration.json")))
    link, xyz = poses["surface_link"], poses["surface_xyz"]
    own = [f"openarm_{side}_{s}" for s in ("link7", "ee_tcp", "camera_link", "left_finger", "right_finger")]
    usable = []
    for li, n in enumerate(names):
        if n in own or n == "world":
            continue
        try:
            chain.pose(n, {})
            usable.append(li)
        except Exception:
            pass
    sel = np.nonzero(np.isin(link, usable))[0]
    sel = np.random.default_rng(0).choice(sel, min(40000, len(sel)), replace=False)
    link, xyz = link[sel], xyz[sel].astype(np.float64)

    split = json.load(open(args.split))
    eps = split["camera_cal_train"] + split["camera_cal_val"]
    data = pd.concat(pd.read_parquet(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(ROOT, "data"))
                     for f in sorted(fs) if f.endswith(".parquet"))
    state_names = json.load(open(os.path.join(ROOT, "meta/info.json")))["features"]["observation.state"]["names"]
    R0, t0 = qmat(poses["cam_offset_rot"]), np.array(poses["cam_offset_pos"], float)
    rng = np.random.default_rng(1)
    mounts = [(R0, t0)] + [(R0 @ Rotation.from_rotvec(np.radians(rng.normal(0, 10, 3))).as_matrix(), t0 + rng.normal(0, 0.03, 3))
                           for _ in range(args.perturb)]
    H, W, s = 120, 160, 4
    areas = [[] for _ in mounts]
    for ep in eps:
        rows = data[data.episode_index == ep].sort_values("frame_index")
        for st in np.stack(rows["observation.state"].to_numpy())[:: args.every]:
            q = real_state_to_sim(st, state_names, calib)
            T = {li: chain.pose(names[li], q) for li in usable}
            Tp = chain.pose(parent, q)
            Rl = np.stack([T[l][:3, :3] for l in link])
            tl = np.stack([T[l][:3, 3] for l in link])
            world = np.einsum("nij,nj->ni", Rl, xyz) + tl
            in_parent = (world - Tp[:3, 3]) @ Tp[:3, :3]
            for m, (R, t) in enumerate(mounts):
                c = (in_parent - t) @ R
                ok = c[:, 2] > 0.05
                u = (args.fx * c[ok, 0] / c[ok, 2] + 320) / s
                v = (args.fx * c[ok, 1] / c[ok, 2] + 240) / s
                g = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                img = np.zeros((H, W), bool)
                img[v[g].astype(int), u[g].astype(int)] = True
                areas[m].append(img.mean())
    a = np.array(areas)
    out = {"camera": args.camera, "episodes": len(eps), "frames": int(a.shape[1]), "every": args.every,
           "cfg_mount": {"area_p50_p90_p99": np.percentile(a[0], [50, 90, 99]).round(4).tolist(),
                         "frames_ge_1pct": int((a[0] >= 0.01).sum()), "frames_ge_5pct": int((a[0] >= 0.05).sum())},
           "perturbed_mounts_frames_ge_5pct_min_median_max": [int(x) for x in np.percentile((a[1:] >= 0.05).sum(1), [0, 50, 100])],
           "note": "area = share of a 160x120 grid hit by projected non-own robot surface points (point splat, no hole filling: a lower bound on silhouette area)"}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
