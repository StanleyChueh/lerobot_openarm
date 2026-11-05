"""Task D: does the generated wrist video reflect the generated motion and object, compared with controls?

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/eval_wrist.py --episodes 0 3 8

For each generated candidate (full episode) and wrist camera:
  viewpoint   (v3.2 generators mostly TRANSLATE the wrist camera, so a rotation test is uninformative.) For frames
              whose generated camera is >= 1 cm from the source camera, take the REAL source frame k recorded at
              the camera pose nearest to the generated pose (other time, within 60% of the offset and 6 deg): the
              generated frame should match it (NCC on static, non-can pixels) better than the unchanged frame does.
  object      measured shift of the red-can centroid (generated vs source frame) against the shift commanded by the
              candidate (projected can centre, new vs source camera and can pose); the absolute calibration offset
              cancels in the difference. CONTROL: the unchanged source video (zero shift).
  grasp       own-hold frames: fraction of can pixels copied (attached) and red-pixel fraction, generated vs source.
  visibility  per frame: can predicted in view (projected centroid inside the image, z > 0) vs detected
              (red fraction >= 2%): agreement rate, generated vs source-control.
  novelty     mean |generated - source| (8-bit) and fraction of pixels changed by > 10, on frames with changed pose.
  provenance  label fractions (reprojected / attached / can warp / multiview / inpainted / unknown).
Nothing here is ground truth: the can pose and mounts are reconstructions (see calibration results).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
import calib_wrist as cw  # noqa: E402
from gen_wrist import CACHE, V3, load_cand  # noqa: E402
from sfa3 import recon3  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402
from sfa2 import render2  # noqa: E402
from sfa32 import render_wrist as rw, wristcam as wc  # noqa: E402
from sfa.recover import interp_q  # noqa: E402
from sfa import track as trk  # noqa: E402


def _unused_cam_rot_pairs(F, pairs, side, device):
    fw = trk.flow_pairs(F, pairs, device)
    Kc = wc.K(side)
    H, W = F.shape[1:3]
    gy, gx = np.mgrid[8:H - 8:10, 8:W - 8:10]
    P = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    out = []
    for f in fw:
        q = P + trk.sample(f, P)
        ok = (q[:, 0] > 0) & (q[:, 0] < W - 1) & (q[:, 1] > 0) & (q[:, 1] < H - 1) & (np.linalg.norm(q - P, axis=1) > 0.3)
        R = None
        if ok.sum() > 150:
            E, inl = cv2.findEssentialMat(P[ok], q[ok], Kc, cv2.RANSAC, 0.999, 1.0)
            if E is not None and E.shape == (3, 3):
                n, Rm, t, _ = cv2.recoverPose(E, P[ok], q[ok], Kc, mask=inl.copy())
                if n > 100:
                    R = Rm.T
        out.append(R)
    return out


def red_centroid(rgb):
    m = rw.can_mask(rgb)
    if m.mean() < 0.02:
        return None, float(m.mean())
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()]), float(m.mean())


def evaluate(E, key, robot, rec, ep, device):
    cand = load_cand(E, key)
    T = rec.T
    res = {}
    g_src = rb.Robot.split(ep.state)
    for side in "RL":
        gen = np.load(CACHE / f"ep{E}" / f"wrist{side}_{key}.npy", mmap_mode="r")
        lab = np.load(CACHE / f"ep{E}" / f"wristlab_{key}.npz")[side]
        X, lag = wc.mount(side), wc.lag(side)
        src = ep.frames[wc.CAM_OF[side]]
        j_of = np.array([int(np.clip(round(float(render2.tau_at(cand["tau"], i + lag)) - lag), 0, T - 1)) for i in range(T)])
        srcv = src[j_of]                                         # unchanged source frames on the candidate clock
        Tn = [wc.cam_pose(robot, interp_q(cand["q"]["R"], i + lag), interp_q(cand["q"]["L"], i + lag), side, X) for i in range(T)]
        Ts = [wc.cam_pose(robot, interp_q(g_src["R"], j + lag), interp_q(g_src["L"], j + lag), side, X) for j in j_of]
        # viewpoint test: the source frame whose camera pose is nearest to the GENERATED pose (other time, real image)
        # should resemble the generated frame more than it resembles the unchanged source frame, on static pixels
        S_T = [wc.cam_pose(robot, interp_q(g_src["R"], t + lag), interp_q(g_src["L"], t + lag), side, X) for t in range(T)]
        S_pos = np.array([M[:3, 3] for M in S_T])
        wins, gains, nview = 0, [], 0
        for i in range(T):
            D = np.linalg.inv(Tn[i]) @ Ts[i]
            off = np.linalg.norm(D[:3, 3])
            if off < 0.01:
                continue
            d = np.linalg.norm(S_pos - Tn[i][:3, 3], axis=1)
            k = int(np.argmin(d))
            Rk = S_T[k][:3, :3].T @ Tn[i][:3, :3]
            if d[k] > 0.6 * off or cw.angle(Rk) > 6.0 or abs(k - j_of[i]) < 2:
                continue
            ref = src[k]
            stat = ~(rw.can_mask(ref) | rw.can_mask(np.asarray(gen[i])) | rw.can_mask(np.asarray(srcv[i])))
            def ncc(a, b):
                a = cv2.cvtColor(np.asarray(a), cv2.COLOR_RGB2GRAY).astype(np.float32)[stat]
                b = cv2.cvtColor(np.asarray(b), cv2.COLOR_RGB2GRAY).astype(np.float32)[stat]
                a, b = a - a.mean(), b - b.mean()
                return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
            if stat.mean() < 0.2:
                continue
            ng, ns = ncc(gen[i], ref), ncc(srcv[i], ref)
            nview += 1
            wins += ng > ns
            gains.append(ng - ns)
        motion = {"n_frames_tested": nview, "frac_generated_closer_to_nearest_view": wins / nview if nview else None,
                  "median_ncc_gain": float(np.median(gains)) if gains else None}
        # object (right wrist pre-grasp), visibility, grasp evidence, novelty
        obj_err, obj_ctrl, vis_g, vis_s = [], [], [], []
        for i in range(T):
            can_new = render2.can_at(cand["can"], i + lag)
            can_old = rec.source_can(robot, j_of[i] + lag)
            cn, cs_ = wc.camera(side, Tn[i]), wc.camera(side, Ts[i])
            un, zn = cn.project(can_new[:3, 3][None] + 0.1 * can_new[:3, 2][None])
            uo, zo = cs_.project(can_old[:3, 3][None] + 0.1 * can_old[:3, 2][None])
            pred_in = bool(zn[0] > 0.05 and 0 <= un[0, 0] < 640 and 0 <= un[0, 1] < 480)
            cg, fg = red_centroid(np.asarray(gen[i]))
            cs, fs = red_centroid(np.asarray(srcv[i]))
            vis_g.append(pred_in == (fg >= 0.02))
            vis_s.append(pred_in == (fs >= 0.02))
            pred_shift = un[0] - uo[0]
            if cg is not None and cs is not None and zn[0] > 0.05 and zo[0] > 0.05 and np.linalg.norm(pred_shift) >= 10 and 0.02 <= fs <= 0.5:
                obj_err.append(float(np.linalg.norm((cg - cs) - pred_shift)))     # generated: measured shift vs commanded shift
                obj_ctrl.append(float(np.linalg.norm(pred_shift)))                 # unchanged source: zero shift
        g_c = rb.Robot.split(cand["state"])["g" + side]
        hold = [i for i in range(T) if g_c[i] < rw.CLOSED and rec.source_holder(min(T - 1, int(round(j_of[i] + lag)))) != "table"]
        changed = [i for i in range(T) if np.abs(np.linalg.inv(Tn[i]) @ Ts[i] - np.eye(4)).max() > 1e-6]
        diff = [float(np.abs(np.asarray(gen[i], np.int16) - np.asarray(srcv[i], np.int16)).mean()) for i in changed[::3]]
        chg = [float((np.abs(np.asarray(gen[i], np.int16) - np.asarray(srcv[i], np.int16)).max(2) > 10).mean()) for i in changed[::3]]
        res[side] = {"motion": motion,
                     "object_shift_error_px": {"generated_median": float(np.median(obj_err)) if obj_err else None,
                                               "source_control_median": float(np.median(obj_ctrl)) if obj_ctrl else None, "n": len(obj_err)},
                     "visibility_agreement": {"generated": float(np.mean(vis_g)), "source_control": float(np.mean(vis_s))},
                     "grasp_hold_frames": len(hold),
                     "grasp_attached_frac_in_hold": float(np.mean([(lab[i] == rw.LAB["attached"]).mean() for i in hold])) if hold else None,
                     "red_frac_hold": {"generated": float(np.mean([red_centroid(np.asarray(gen[i]))[1] for i in hold[::3]])) if hold else None,
                                       "source": float(np.mean([red_centroid(np.asarray(srcv[i]))[1] for i in hold[::3]])) if hold else None},
                     "novelty": {"frames_with_changed_pose": len(changed), "mean_abs_diff": float(np.mean(diff)) if diff else 0.0,
                                 "frac_pixels_changed_gt10": float(np.mean(chg)) if chg else 0.0},
                     "provenance": {k: float((lab == v).mean()) for k, v in rw.LAB.items()},
                     "provenance_changed_frames": {k: float(np.mean([(lab[i] == v).mean() for i in changed])) if changed else None for k, v in rw.LAB.items()}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 3, 8])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--max_per_episode", type=int, default=6)
    args = ap.parse_args()
    robot = rb.Robot()
    out = {}
    for E in args.episodes:
        ep = data.load(E)
        rec = recon3.build(E, with_plate=False, log=lambda *a: None)
        keys = sorted(p.name[len("wristR_"):-4] for p in (CACHE / f"ep{E}").glob("wristR_*.npy"))
        keys = [k for k in keys if k != "identity__C__s1.0"][: args.max_per_episode]
        for key in keys:
            out[f"ep{E}/{key}"] = evaluate(E, key, robot, rec, ep, args.device)
            r = out[f"ep{E}/{key}"]
            print(E, key, {s: (r[s]["motion"], r[s]["object_shift_error_px"]) for s in "RL"}, flush=True)
    (HERE / "outputs/wrist_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
