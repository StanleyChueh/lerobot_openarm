"""Diagnose wrist-camera calibration inputs from a render_real_states.py run and segment_robot_real.py masks.

  (a) transform composition: Isaac's recorded camera world pose vs parent-link pose composed with the cfg offset
      (ROS optical convention), per frame -- checks that the offset means what the cfg says.
  (b) robot visibility: share of the image covered by the sim robot's projection, split into the camera's own hand
      (link7, ee_tcp, camera link, fingers) and everything else (other arm, torso), at the cfg mount.
  (c) mask agreement: IoU between the real robot mask and the sim projection (non-own links) at the cfg mount, and the
      share of real-mask pixels lying more than 40 px from any sim robot pixel (mask content the robot model cannot explain).

  python diagnose_wrist.py <poses.npz> <robot_masks.npz> <frames.npz> --intrinsics fx,fy,cx,cy --out diag.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_wrist_mount import MovingParentFit, qmat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poses")
    ap.add_argument("masks")
    ap.add_argument("frames")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    poses, frames = np.load(args.poses), np.load(args.frames)
    masks = np.load(args.masks)["masks"]
    names = [str(b) for b in poses["body_names"]]
    pi = names.index(str(poses["parent_link"]))
    R_off, t_off = qmat(poses["cam_offset_rot"]), np.array(poses["cam_offset_pos"])

    # (a) composition
    dpos, drot = [], []
    for k in range(len(masks)):
        R_wp, t_wp = qmat(poses["body_quat_w"][k][pi]), poses["body_pos_w"][k][pi]
        t_pred, R_pred = t_wp + R_wp @ t_off, R_wp @ R_off
        dpos.append(np.linalg.norm(t_pred - poses["cam_pos_w"][k]) * 1000)
        drot.append(np.degrees(Rotation.from_matrix(R_pred.T @ qmat(poses["cam_quat_w_ros"][k])).magnitude()))

    fx, fy, cx, cy = (float(v) for v in args.intrinsics.split(","))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    no_ignore = np.zeros(masks.shape[1:], bool)
    other = MovingParentFit(poses, masks, no_ignore, K, "cuda", 300000)
    idx = np.arange(len(masks))
    sim_other = other.silhouettes(R_off, t_off, idx)
    # own-hand points only: same projection with the link filter inverted
    own_names = set(other.own_links)
    link = poses["surface_link"]
    own_sel = np.array([names[l] in own_names for l in link])
    own_area = []
    for k in idx:
        R_b = Rotation.from_quat(poses["body_quat_w"][k][:, [1, 2, 3, 0]]).as_matrix()
        p = np.einsum("nij,nj->ni", R_b[link[own_sel]], poses["surface_xyz"][own_sel]) + poses["body_pos_w"][k][link[own_sel]]
        R_wp, t_wp = qmat(poses["body_quat_w"][k][pi]), poses["body_pos_w"][k][pi]
        c = ((p - t_wp) @ R_wp - t_off) @ R_off
        c = c[c[:, 2] > 0.02]
        u, v = (fx * c[:, 0] / c[:, 2] + cx).astype(int), (fy * c[:, 1] / c[:, 2] + cy).astype(int)
        ok = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        img = np.zeros((480, 640), np.uint8)
        img[v[ok], u[ok]] = 1
        own_area.append(cv2.dilate(img, np.ones((5, 5), np.uint8)).mean())
    other_area = sim_other.reshape(len(idx), -1).mean(1)
    real_area = masks.reshape(len(idx), -1).mean(1)
    iou, unexplained = [], []
    for k in idx:
        s, r = sim_other[k], masks[k]
        iou.append((s & r).sum() / max((s | r).sum(), 1))
        if r.any():
            d = cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 5) if s.any() else np.full(r.shape, 1e3, np.float32)
            unexplained.append(float((d[r] > 40).mean()))
    q = lambda a, p=(10, 50, 90): [round(float(np.percentile(a, x)), 4) for x in p]
    out = {
        "frames": int(len(idx)),
        "composition_err_mm_max": round(float(max(dpos)), 4), "composition_err_deg_max": round(float(max(drot)), 5),
        "own_links": sorted(own_names),
        "sim_own_hand_area_p10_50_90": q(own_area),
        "sim_other_robot_area_p10_50_90": q(other_area),
        "frames_other_robot_ge_1pct": int((other_area >= 0.01).sum()), "frames_other_robot_ge_5pct": int((other_area >= 0.05).sum()),
        "real_mask_area_p10_50_90": q(real_area),
        "iou_real_vs_sim_other_p10_50_90": q(iou),
        "real_mask_unexplained_share_p10_50_90": q(unexplained) if unexplained else None,
        "frames_real_mask_mostly_unexplained_gt_50pct": int(sum(u > 0.5 for u in unexplained)),
        "frames_with_real_mask": int(len(unexplained)),
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
