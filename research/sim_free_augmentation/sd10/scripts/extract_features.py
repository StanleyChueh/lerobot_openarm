"""STEP 2 - per-episode audit features for all 50 pool episodes (read-only; GPU used only for SAM2.1 inference).

* Can at frame 0: SAM2.1-small image mask prompted by the existing verified human click (body camera).
* Can over the episode: SAM2.1-small video propagation from that mask (visibility, displacement).
* Robot: commanded events, measured gripper openness (joint state), FK TCP poses (pinocchio, follower URDF).
* Image quality: Laplacian-variance sharpness (can crop, wrist frames at grasp/receive).
* Keyframes for manual review (start, rc, rc+20, lc, ro+10, last) are cached to outputs/keyframes/ (gitignored).

v4A locked records: body camera only; wrist pixels are never decoded (common.Dataset.decode refuses).

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/extract_features.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # research/sim_free_augmentation (sfa package)
from sd10 import common as C  # noqa: E402
from sfa.robot import Robot  # noqa: E402

SAM_IMG = "facebook/sam2.1-hiera-small"   # large weights are not cached locally; small = v1 pipeline model
SAM_VID = "facebook/sam2.1-hiera-small"
CLICKS = C.REPO / "research/dataset_diagnostics/annotations/real_v00.csv"
KEY = C.OUT / "keyframes"


def clicks() -> dict:
    return {int(r["episode_index"]): (float(r["u"]), float(r["v"]), r["timestamp"]) for r in csv.DictReader(open(CLICKS))
            if r["dataset"] == "real_v00_pool" and r["verified"] == "1" and r["frame_index"] == "0"}


def transitions(closed: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.diff(closed.astype(int)) != 0) + 1]


def sharp(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


class Sam:
    def __init__(self, dev):
        from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor
        self.dev = dev
        self.im = Sam2Model.from_pretrained(SAM_IMG).to(dev).eval()
        self.ip = Sam2Processor.from_pretrained(SAM_IMG)
        self.vm = Sam2VideoModel.from_pretrained(SAM_VID).to(dev, dtype=torch.bfloat16).eval()
        self.vp = Sam2VideoProcessor.from_pretrained(SAM_VID)

    @torch.inference_mode()
    def frame0(self, img, uv):
        inp = self.ip(images=img, input_points=[[[list(uv)]]], input_labels=[[[1]]], return_tensors="pt").to(self.dev)
        out = self.im(**inp, multimask_output=True)
        m = self.ip.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0].numpy().astype(bool)
        s = out.iou_scores[0, 0].float().cpu().numpy()
        return m, s

    @torch.inference_mode()
    def track(self, frames, mask0):
        sess = self.vp.init_video_session(video=list(frames), inference_device=self.dev, dtype=torch.bfloat16)
        self.vp.add_inputs_to_inference_session(sess, frame_idx=0, obj_ids=1, input_masks=[mask0])
        self.vm(inference_session=sess, frame_idx=0)
        T, H, W = frames.shape[:3]
        area, cen = np.zeros(T), np.full((T, 2), np.nan)
        for o in self.vm.propagate_in_video_iterator(sess, start_frame_idx=0):
            m = self.vp.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
            m = m[0].squeeze().cpu().numpy().astype(bool)
            area[o.frame_idx] = m.sum()
            if m.any():
                ys, xs = np.nonzero(m)
                cen[o.frame_idx] = (xs.mean(), ys.mean())
        return area, cen


def choose_mask(masks, scores, uv):
    """Whole-can choice among SAM's three proposals: must contain the click and have a plausible area; prefer the
    largest proposal whose IoU score is within 0.05 of the best eligible score (SAM's whole-object level)."""
    u, v = int(round(uv[0])), int(round(uv[1]))
    cand = []
    for k in range(len(masks)):
        a = int(masks[k].sum())
        cand.append({"k": k, "area": a, "score": float(scores[k]), "contains_click": bool(masks[k][v, u]),
                     "ok": bool(masks[k][v, u]) and 150 <= a <= 25000})
    ok = [c for c in cand if c["ok"]]
    if not ok:
        return None, cand
    best = max(c["score"] for c in ok)
    pick = max((c for c in ok if c["score"] >= best - 0.05), key=lambda c: c["area"])
    return pick["k"], cand


def robot_features(robot: Robot, S, A):
    g, gs = Robot.split(A), Robot.split(S)
    cR, cL = g["gR"] < 0.5, g["gL"] < 0.5
    tR, tL = transitions(cR), transitions(cL)
    T = len(S)
    f = {"T": T, "cmd_right_transitions": tR, "cmd_left_transitions": tL,
         "right_closed_at_start": bool(cR[0]), "left_closed_at_start": bool(cL[0])}
    rc = tR[0] if len(tR) >= 1 and not cR[0] else None
    ro = tR[1] if len(tR) >= 2 else None
    lc = tL[0] if len(tL) >= 1 and not cL[0] else None
    f.update(rc=rc, lc=lc, ro=ro)
    f["q2_command_structure"] = bool(len(tR) == 2 and len(tL) == 1 and not cR[0] and not cL[0]
                                     and rc is not None and lc is not None and ro is not None and rc < lc <= ro)
    oR, oL = gs["gR"], gs["gL"]
    f["meas_right_open_start"] = float(oR[0]); f["meas_left_open_start"] = float(oL[0])
    f["meas_right_open_last"] = float(oR[-1]); f["meas_left_open_last"] = float(oL[-1])
    if rc is not None and lc is not None:
        w = oR[rc + 10:lc]
        f["meas_right_grasp_window_frames"] = int(len(w))
        f["meas_right_grasp_median"] = float(np.median(w)) if len(w) else None
        f["meas_right_grasp_min"] = float(w.min()) if len(w) else None
    if lc is not None:
        w = oL[lc + 15:]
        f["meas_left_hold_frames"] = int(len(w))
        f["meas_left_hold_median"] = float(np.median(w)) if len(w) else None
        f["meas_left_hold_drift_p95_p5"] = float(np.percentile(w, 95) - np.percentile(w, 5)) if len(w) else None
    f["q3_measured_right_grasp"] = bool(f.get("meas_right_grasp_window_frames", 0) >= 10
                                        and 0.15 <= (f.get("meas_right_grasp_median") or -1) <= 0.80
                                        and f["meas_right_open_last"] > 0.80)
    f["q4_measured_left_hold"] = bool((f.get("meas_left_hold_frames") or 0) >= 1
                                      and 0.15 <= (f.get("meas_left_hold_median") or -1) <= 0.80
                                      and (f.get("meas_left_hold_drift_p95_p5") or 9) <= 0.10)
    # measured handover overlap: left reaches its hold plateau (within 0.05) vs right starts to open (> grasp +0.05)
    if rc is not None and lc is not None and ro is not None and f.get("meas_left_hold_median") is not None:
        lp = np.flatnonzero(oL[lc:] <= f["meas_left_hold_median"] + 0.05)
        rp = np.flatnonzero(oR[ro:] > (f["meas_right_grasp_median"] or 0) + 0.05)
        l_hold = int(lc + lp[0]) if len(lp) else None
        r_rel = int(ro + rp[0]) if len(rp) else None
        f["meas_left_hold_onset"], f["meas_right_release_onset"] = l_hold, r_rel
        f["meas_dual_hold_frames"] = (r_rel - l_hold) if (l_hold is not None and r_rel is not None) else None
    # FK at events (measured state)
    sp = Robot.split(S)

    def tcp(t):
        robot.fk(sp["R"][t], sp["L"][t])
        return {s: robot.frame(robot.tcp[s]) for s in "RL"}

    if rc is not None:
        P = tcp(rc)
        f["right_tcp_rc_xyz"] = P["R"][:3, 3].tolist()
        f["right_tcp_rc_approach"] = P["R"][:3, 2].tolist()
        f["right_tcp_rc_yaw_deg"] = float(np.degrees(np.arctan2(P["R"][1, 0], P["R"][0, 0])))
        f["right_q_rc"] = sp["R"][rc].tolist()
    if lc is not None:
        P = tcp(lc)
        f["left_tcp_lc_xyz"] = P["L"][:3, 3].tolist()
        f["left_tcp_lc_approach"] = P["L"][:3, 2].tolist()
        f["left_tcp_lc_yaw_deg"] = float(np.degrees(np.arctan2(P["L"][1, 0], P["L"][0, 0])))
        f["right_tcp_lc_xyz"] = P["R"][:3, 3].tolist()
        f["tcp_separation_lc_m"] = float(np.linalg.norm(P["L"][:3, 3] - P["R"][:3, 3]))
        f["left_q_lc"] = sp["L"][lc].tolist()
    if rc is not None and lc is not None:
        f["handover_dt_s"] = (lc - rc) / C.FPS
    if lc is not None and ro is not None:
        f["cmd_overlap_frames"] = ro - lc
    return f


def keyframe_idx(f):
    T = f["T"]
    rc, lc, ro = f.get("rc"), f.get("lc"), f.get("ro")
    ks = [0, rc, None if rc is None else rc + 20, lc, None if ro is None else ro + 10, T - 1]
    return [int(min(max(k, 0), T - 1)) if k is not None else None for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ds = C.pool()
    ck = clicks()
    robot = Robot()
    sam = Sam(a.device)
    KEY.mkdir(parents=True, exist_ok=True)
    rows = {}
    eps = a.episodes or ds.episodes
    for e in eps:
        t0 = time.time()
        S, A = ds.vec(e, "observation.state"), ds.vec(e, "action")
        f = {"episode": e, "v4a_locked": e in C.V4A_LOCKED, "click_uv": ck[e][:2], "click_time": ck[e][2]}
        f.update(robot_features(robot, S, A))
        body = ds.decode(C.BODY, e)
        masks, scores = sam.frame0(body[0], ck[e][:2])
        k, cand = choose_mask(masks, scores, ck[e][:2])
        f["sam_candidates"] = cand
        f["sam_choice"] = k
        if k is not None:
            m0 = masks[k]
            ys, xs = np.nonzero(m0)
            f["can_u"], f["can_v"] = float(xs.mean()), float(ys.mean())
            f["can_area_px"] = int(m0.sum())
            f["can_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            f["click_to_centroid_px"] = float(np.hypot(f["can_u"] - ck[e][0], f["can_v"] - ck[e][1]))
            x0, y0, x1, y1 = f["can_bbox"]
            pad = 10
            f["can_crop_sharpness"] = sharp(gray(body[0])[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad])
            area, cen = sam.track(body, m0)
            a0 = area[0] if area[0] > 0 else f["can_area_px"]
            f["track_visible_frac"] = float((area >= 0.3 * a0).mean())
            f["track_min_area_frac"] = float(area.min() / a0)
            ok = np.isfinite(cen).all(1)
            last = np.flatnonzero(ok)[-1] if ok.any() else None
            f["track_last_uv"] = cen[last].tolist() if last is not None else None
            f["track_displacement_last_px"] = float(np.hypot(*(cen[last] - cen[0]))) if last is not None else None
            if f.get("lc") is not None and ok[f["lc"]]:
                f["track_uv_lc"] = cen[f["lc"]].tolist()
                f["track_lift_lc_px"] = float(cen[0][1] - cen[f["lc"]][1])     # +: moved up in the image
            np.savez_compressed(KEY / f"track_{e:02d}.npz", area=area, cen=cen, mask0=np.packbits(m0, axis=-1))
        f["body_sharpness_median"] = float(np.median([sharp(gray(im)) for im in body[::10]]))
        ks = keyframe_idx(f)
        f["keyframes"] = ks
        kv = [k_ for k_ in ks if k_ is not None]
        pack = {"body": body[kv][:, ::2, ::2]}
        if not f["v4a_locked"]:
            for cam, nm in ((C.LEFT_WRIST, "left"), (C.RIGHT_WRIST, "right")):
                W = ds.decode(cam, e, sorted(set(kv)))
                Wd = {k_: W[i] for i, k_ in enumerate(sorted(set(kv)))}
                pack[nm] = np.stack([Wd[k_] for k_ in kv])[:, ::2, ::2]
            if f.get("rc") is not None:
                f["right_wrist_sharpness_rc"] = sharp(gray(ds.decode(C.RIGHT_WRIST, e, [f["rc"]])[0]))
            if f.get("lc") is not None:
                f["left_wrist_sharpness_lc"] = sharp(gray(ds.decode(C.LEFT_WRIST, e, [f["lc"]])[0]))
        np.savez_compressed(KEY / f"keys_{e:02d}.npz", idx=np.array(kv), **pack)
        rows[e] = f
        print(f"ep{e:02d} T={f['T']} rc/lc/ro={f.get('rc')}/{f.get('lc')}/{f.get('ro')} q2={f['q2_command_structure']} "
              f"q3={f['q3_measured_right_grasp']} q4={f['q4_measured_left_hold']} sam={k} "
              f"vis={f.get('track_visible_frac', float('nan')):.2f} {time.time() - t0:.1f}s", flush=True)
    prev = C.OUT / "features_raw.json"
    allrows = {}
    if prev.exists() and a.episodes:
        import json
        allrows = {int(k): v for k, v in json.loads(prev.read_text()).items()}
    allrows.update(rows)
    C.dump({k: allrows[k] for k in sorted(allrows)}, prev)


if __name__ == "__main__":
    main()
