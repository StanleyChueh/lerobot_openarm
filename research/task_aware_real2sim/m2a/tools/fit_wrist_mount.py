"""Fit a wrist camera's mount on its (moving) parent link by robot-silhouette matching, with held-out validation.

Algorithm (no appearance is compared, only where the robot is):
  * render_real_states.py posed the sim robot at each real frame's joint state and saved, per frame, every link's
    world pose plus points sampled on the links' visual meshes. For a candidate mount X (camera pose in its parent
    link, ROS optical frame) the points of every frame are moved into that frame's parent-link frame, projected
    through a pinhole K, and splatted into a sim silhouette. This differs from calibrate_camera.py (IsaacLab
    scripts/tools/sysid), which assumes a parent fixed in the world and uses frame 0's parent pose for all frames.
  * Links rigidly attached to the camera's own hand (link7, ee_tcp, camera link, both fingers) are dropped from the
    sim points: the real fingertips carry foam pads the sim does not model, and their image position barely depends
    on the mount. Pixels where the real mask is on in >= --static_frac of the training frames (the visible own
    fingertips) form a static ignore region, removed from both masks before scoring.
  * Score = truncated two-sided chamfer between sim and real masks (calibrate_camera.py's cost), minimised by
    Nelder-Mead coarse-to-fine from the current cfg mount, from optional extra starts (--start_json), and from
    seeded random starts. Only frames whose real mask outside the ignore region covers >= --min_frac count.
  * Reported: held-out (val split) IoU and symmetric mean chamfer distance (px) for every candidate mount, multi-start
    agreement, episode-bootstrap spread, per-parameter cost sensitivity, and preview images.

  python fit_wrist_mount.py <poses.npz> <robot_masks.npz> <frames.npz> --camera wrist_cam \
      --intrinsics 605.44,605.13,320,240 --out <dir> --start_json name=<mount.json> --seed 0
"""

import argparse
import json
import os
import re
import time

import cv2
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def qmat(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def matq(R):
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if w >= 0 else -q


class MovingParentFit:
    def __init__(self, poses, masks, ignore, K, device, max_points):
        self.device, (n, self.H, self.W) = device, masks.shape
        names = [str(b) for b in poses["body_names"]]
        parent = str(poses["parent_link"])
        side = "left" if "_left_" in parent else "right"
        own = re.compile(rf"openarm_{side}_(link7|ee_tcp|camera|hand|.*finger)")
        self.own_links = [b for b in names if own.search(b)]
        keep_link = np.array([not own.search(b) for b in names])
        link, xyz = poses["surface_link"], poses["surface_xyz"]
        sel = np.nonzero(keep_link[link])[0]
        if len(sel) > max_points:
            sel = np.random.default_rng(0).choice(sel, max_points, replace=False)
        link, xyz = link[sel], xyz[sel]
        pi = names.index(parent)
        pts = []
        for k in range(n):
            R_b = Rotation.from_quat(poses["body_quat_w"][k][:, [1, 2, 3, 0]]).as_matrix()
            world = np.einsum("nij,nj->ni", R_b[link], xyz) + poses["body_pos_w"][k][link]
            R_wp, t_wp = qmat(poses["body_quat_w"][k][pi]), poses["body_pos_w"][k][pi]
            pts.append((world - t_wp) @ R_wp)  # this frame's parent-link frame
        self.pts = torch.tensor(np.stack(pts), dtype=torch.float32, device=device)
        self.K = torch.tensor(K, dtype=torch.float32, device=device)
        self.keep = ~ignore
        self.real = masks & self.keep
        self.real_dist = np.stack([cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for m in self.real])

    def silhouettes(self, R_pc, t_pc, idx, scale=1, chunk=16):
        R = torch.tensor(R_pc, dtype=torch.float32, device=self.device)
        t = torch.tensor(t_pc, dtype=torch.float32, device=self.device)
        H, W = self.H // scale, self.W // scale
        out_all = []
        for part in np.array_split(idx, max(1, len(idx) // chunk)):
            pts = self.pts[part]
            cam = (pts - t) @ R
            z = cam[..., 2].clamp(min=1e-3)
            u = (self.K[0, 0] * cam[..., 0] / z + self.K[0, 2]) / scale
            v = (self.K[1, 1] * cam[..., 1] / z + self.K[1, 2]) / scale
            ok = (cam[..., 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            fr = torch.arange(len(pts), device=self.device)[:, None].expand_as(u)
            out = torch.zeros(len(pts), H * W, device=self.device)
            out[fr[ok], v[ok].long() * W + u[ok].long()] = 1.0
            out = torch.nn.functional.max_pool2d(out.view(-1, 1, H, W), 3, 1, 1)
            out = -torch.nn.functional.max_pool2d(-torch.nn.functional.max_pool2d(out, 5, 1, 2), 5, 1, 2)
            out_all.append(out[:, 0].bool().cpu().numpy())
        sims = np.concatenate(out_all)
        keep = self.keep[::scale, ::scale]
        return sims & keep[None]

    def cost(self, R_pc, t_pc, idx, scale, tau):
        sims = self.silhouettes(R_pc, t_pc, idx, scale)
        total = 0.0
        for s, k in zip(sims, idx):
            real = self.real[k][::scale, ::scale]
            d_real = self.real_dist[k][::scale, ::scale] / scale
            d_sim = cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 5)
            total += np.minimum(d_real[s], tau).sum() + np.minimum(d_sim[real], tau).sum()
        return total * scale * scale / (len(idx) * self.H * self.W)

    def metrics(self, R_pc, t_pc, idx):
        """Per frame: IoU and symmetric mean chamfer distance in px (untruncated, capped at 200 px)."""
        sims = self.silhouettes(R_pc, t_pc, idx)
        iou, cd = [], []
        for s, k in zip(sims, idx):
            r = self.real[k]
            iou.append((s & r).sum() / max((s | r).sum(), 1))
            if s.any():
                d_sim = cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 5)
                cd.append(0.5 * (np.minimum(self.real_dist[k][s], 200).mean() + np.minimum(d_sim[r], 200).mean()))
            else:
                cd.append(200.0)
        return np.array(iou), np.array(cd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses")
    ap.add_argument("masks")
    ap.add_argument("frames")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--intrinsics", required=True, help="fx,fy,cx,cy used for projection")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start_json", action="append", default=[], help="name=path to a mount json (pos, rot_wxyz_ros)")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--spread_cm", type=float, default=5.0)
    ap.add_argument("--spread_deg", type=float, default=15.0)
    ap.add_argument("--bootstrap", type=int, default=8)
    ap.add_argument("--static_frac", type=float, default=0.3)
    ap.add_argument("--min_frac", type=float, default=0.005)
    ap.add_argument("--max_points", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()

    poses, frames = np.load(args.poses), np.load(args.frames)
    masks = np.load(args.masks)["masks"]
    split, episode = frames["split"].astype(str), frames["episode"]
    train_all = np.nonzero(split == "camera_cal_train")[0]
    ignore = masks[train_all].mean(0) >= args.static_frac
    ignore = cv2.dilate(ignore.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    fit = MovingParentFit(poses, masks, ignore, K, "cuda" if torch.cuda.is_available() else "cpu", args.max_points)
    frac = fit.real.reshape(len(masks), -1).mean(1)
    informative = frac >= args.min_frac
    tr = np.nonzero(informative & (split == "camera_cal_train"))[0]
    va = np.nonzero(informative & (split == "camera_cal_val"))[0]
    print(f"[{args.camera}] own-hand links dropped: {fit.own_links}; ignore region {ignore.mean() * 100:.1f}% of image")
    print(f"[{args.camera}] informative frames: train {len(tr)}/{len(train_all)}, val {len(va)}/{(split == 'camera_cal_val').sum()}")

    R0, t0 = qmat(poses["cam_offset_rot"]), np.array(poses["cam_offset_pos"], float)

    def unpack(x):
        return R0 @ Rotation.from_rotvec(np.radians(x[3:])).as_matrix(), t0 + x[:3] / 100.0

    def pack(R, t):
        return np.r_[(t - t0) * 100.0, np.degrees(Rotation.from_matrix(R0.T @ R).as_rotvec())]

    def solve(x, idx, scale, tau, iters):
        simplex = np.vstack([x, x + np.eye(6) * scale])
        r = minimize(lambda x: fit.cost(*unpack(x), idx, scale, tau), x, method="Nelder-Mead",
                     options={"maxfev": iters, "xatol": 0.05, "fatol": 1e-6, "initial_simplex": simplex})
        return r.x, r.fun

    def full(x, idx):
        x, _ = solve(x, idx, 4, 40.0, 600)
        x, _ = solve(x, idx, 2, 20.0, 600)
        return x, fit.cost(*unpack(x), idx, 2, 20.0)

    starts = {"cfg": np.zeros(6)}
    for s in args.start_json:
        name, path = s.split("=", 1)
        j = json.load(open(path))
        starts[name] = pack(qmat(j["rot_wxyz_ros"]), np.array(j["pos"], float))
    rng = np.random.default_rng(args.seed)
    for i in range(args.restarts):
        starts[f"rand{i}"] = np.r_[rng.normal(0, args.spread_cm, 3), rng.normal(0, args.spread_deg, 3)]

    results = {}
    for name, x in starts.items():
        x, f = full(x, tr)
        results[name] = (f, x)
        print(f"  start {name:8s}: cost {f:.5f}  dpos {np.round(x[:3], 2)} cm  drot {np.round(x[3:], 2)} deg", flush=True)
    best_name = min(results, key=lambda k: results[k][0])
    f_best, x_best = results[best_name]
    near = [k for k, (f, _) in results.items() if f <= 1.05 * f_best]

    def pose_gap(xa, xb):
        Ra, ta = unpack(xa)
        Rb, tb = unpack(xb)
        return float(np.linalg.norm(ta - tb) * 100), float(np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude()))

    agree = {k: pose_gap(results[k][1], x_best) for k in near}
    # sensitivity: relative cost increase for +-1 cm / +-1 deg per parameter at the optimum
    sens = []
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1.0
        up = fit.cost(*unpack(x_best + e), tr, 2, 20.0)
        dn = fit.cost(*unpack(x_best - e), tr, 2, 20.0)
        sens.append(round(float(0.5 * (up + dn) / f_best - 1.0), 4))
    # episode bootstrap
    boot = []
    eps_tr = np.unique(episode[tr])
    for b in range(args.bootstrap):
        pick = rng.choice(eps_tr, len(eps_tr), replace=True)
        idx = np.concatenate([tr[episode[tr] == e] for e in pick])
        xb, _ = solve(x_best, idx, 2, 20.0, 400)
        boot.append(xb)
    boot = np.array(boot)
    Rs = [unpack(x)[0] for x in boot]
    ts = np.array([unpack(x)[1] for x in boot])
    R_best, t_best = unpack(x_best)
    boot_rot = [float(np.degrees(Rotation.from_matrix(R_best.T @ R).magnitude())) for R in Rs]

    cands = {"cfg": np.zeros(6), "fitted": x_best}
    for s in args.start_json:
        cands[s.split("=", 1)[0]] = starts[s.split("=", 1)[0]]
    val = {}
    for name, x in cands.items():
        iou, cd = fit.metrics(*unpack(x), va)
        iou_t, cd_t = fit.metrics(*unpack(x), tr)
        val[name] = {"val_iou_median": round(float(np.median(iou)), 3), "val_iou_mean": round(float(iou.mean()), 3),
                     "val_chamfer_px_median": round(float(np.median(cd)), 2), "val_chamfer_px_p75": round(float(np.percentile(cd, 75)), 2),
                     "train_iou_median": round(float(np.median(iou_t)), 3), "train_chamfer_px_median": round(float(np.median(cd_t)), 2)}
        print(f"  {name:8s} val IoU median {val[name]['val_iou_median']:.3f}  chamfer median {val[name]['val_chamfer_px_median']:.1f} px"
              f"  (train IoU {val[name]['train_iou_median']:.3f}, chamfer {val[name]['train_chamfer_px_median']:.1f})")

    q = matq(R_best)
    d_cfg = pose_gap(np.zeros(6), x_best)
    out = {
        "camera": args.camera, "parent_link": str(poses["parent_link"]), "intrinsics_used": [fx, fy, cx, cy],
        "pos": t_best.tolist(), "rot_wxyz_ros": q.tolist(), "best_start": best_name, "train_cost": f_best,
        "change_vs_cfg_cm_deg": d_cfg, "starts": {k: {"cost": float(f), "x": np.round(x, 3).tolist()} for k, (f, x) in results.items()},
        "near_optimal_starts_gap_cm_deg": agree, "sensitivity_rel_cost_per_cm_deg": sens,
        "bootstrap_pos_std_cm": (ts.std(0) * 100).round(3).tolist(), "bootstrap_rot_dev_deg": np.round(boot_rot, 3).tolist(),
        "frames": {"train_informative": int(len(tr)), "val_informative": int(len(va)), "train_episodes": eps_tr.tolist(),
                   "val_episodes": np.unique(episode[va]).tolist()},
        "own_links_dropped": fit.own_links, "ignore_region_frac": float(ignore.mean()),
        "metrics": val, "seed": args.seed, "args": vars(args), "runtime_s": round(time.time() - t_start, 1),
        "cfg_patch": f"offset=CameraCfg.OffsetCfg(pos=({t_best[0]:.5f}, {t_best[1]:.5f}, {t_best[2]:.5f}), "
                     f"rot=({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f}), convention=\"ros\")",
    }
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "wrist_fit.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("change_vs_cfg_cm_deg", "near_optimal_starts_gap_cm_deg", "sensitivity_rel_cost_per_cm_deg",
                                          "bootstrap_pos_std_cm", "bootstrap_rot_dev_deg", "cfg_patch", "runtime_s")}, indent=1))

    # previews on val frames: real mask red, cfg magenta, fitted green, ignore region dimmed
    images = frames["images"]
    s_cfg, s_fit = fit.silhouettes(R0, t0, va), fit.silhouettes(R_best, t_best, va)
    pv = os.path.join(args.out, "val_previews")
    os.makedirs(pv, exist_ok=True)
    for j, k in enumerate(va):
        vis = cv2.cvtColor(images[k], cv2.COLOR_RGB2BGR)
        vis[ignore] = (vis[ignore] * 0.4).astype(np.uint8)
        for m, col in ((s_cfg[j], (255, 0, 255)), (fit.real[k], (0, 0, 255)), (s_fit[j], (0, 255, 0))):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cs, -1, col, 2)
        cv2.imwrite(os.path.join(pv, f"val_{j:03d}_ep{episode[k]}_f{frames['frame'][k]}.jpg"), vis)


if __name__ == "__main__":
    main()
