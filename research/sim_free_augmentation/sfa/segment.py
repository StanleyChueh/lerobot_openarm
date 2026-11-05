"""Video segmentation of the two arms (incl. grippers) and the can with SAM2.1 (transformers Sam2VideoModel, GPU).

Prompts are generated automatically, never hand-drawn:
  * arms (default `arm_prompt="mask"`): the union of the projected URDF convex hulls of that arm's link4 .. fingers,
    given to SAM2 as a MASK prompt every 15 frames (the can's neighbourhood removed). The camera model is ~20-40 px off
    in places; SAM2 snaps to the real silhouette (incl. the wrist camera the URDF does not model). Box + point prompts
    (`arm_prompt="box"`) were tried first and failed: the dark arms on the dark table lost to the bright window.
  * can: a positive point on the lid (Gate 1 HSV lid tracker, automatic) and one on the red body (nearest red pixel
    cluster below the lid), at the keyframes where the lid tracker is `ok`.
The masks are automatic labels, not ground truth.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .robot import Camera, Robot

MODEL = "facebook/sam2.1-hiera-small"
OBJ = {"R": 1, "L": 2, "can": 3}
DISTAL = ("hand", "left_finger", "right_finger")


def arm_box(robot: Robot, cam: Camera, qR, qL, side, pad=20, H=480, W=640):
    robot.fk(qR, qL)
    pts = []
    for name, s, P in robot.link_points("hull"):
        if s == side and name.split("_", 1)[1] in DISTAL:
            uv, z = cam.project(P)
            if (z > 0.15).all():
                pts.append(uv)
    if not pts:
        return None
    uv = np.concatenate(pts)
    x0, y0 = np.clip(uv.min(0) - pad, 0, [W - 1, H - 1])
    x1, y1 = np.clip(uv.max(0) + pad, 0, [W - 1, H - 1])
    if (x1 - x0) < 20 or (y1 - y0) < 20:
        return None
    return [float(x0), float(y0), float(x1), float(y1)]


def hand_points(robot: Robot, cam: Camera, qR, qL, side, H=480, W=640):
    """projected hand-frame origin and TCP of one arm (inside the image) - positive prompts for that arm."""
    robot.fk(qR, qL)
    out = []
    for fid in (robot.hand[side], robot.tcp[side]):
        uv, z = cam.project(robot.frame(fid)[:3, 3][None])
        if z[0] > 0.15 and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H:
            out.append([float(uv[0, 0]), float(uv[0, 1])])
    return out


def red_point(rgb, lid_uv, radius=110):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    yy, xx = np.nonzero(m)
    if not len(xx):
        return None
    d = np.hypot(xx - lid_uv[0], yy - lid_uv[1])
    sel = (d > 25) & (d < radius)
    if sel.sum() < 30:
        return None
    return [float(np.median(xx[sel])), float(np.median(yy[sel]))]


def arm_prompt_mask(robot: Robot, cam: Camera, qR, qL, gR, gL, side, links=("link4", "link5", "link6", "link7", "hand",
                    "left_finger", "right_finger"), H=480, W=640):
    """union of the projected convex hulls of an arm's distal links (camera model ~20-40 px off -> a coarse prompt)."""
    robot.fk(qR, qL, gR, gL)
    m = np.zeros((H, W), np.uint8)
    for name, s, P in robot.link_points("hull"):
        if s == side and name.split("_", 1)[1] in links:
            uv, z = cam.project(P)
            if (z > 0.15).all():
                cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    return m


def segment(frames: np.ndarray, robot: Robot, cam: Camera, g_video: dict, lid_uv, lid_ok, keyframes, device="cuda:0",
            arm_prompt="mask", grip=None):
    """frames (T,H,W,3); g_video: joints at the video-aligned time {"R": (T,7), "L": (T,7)}.
    Returns {"R","L","can"} -> (T,H,W) uint8 masks, and the prompts used."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    model = Sam2VideoModel.from_pretrained(MODEL).to(device, dtype=torch.bfloat16)
    proc = Sam2VideoProcessor.from_pretrained(MODEL)
    T, H, W = frames.shape[:3]
    sess = proc.init_video_session(video=list(frames), inference_device=device, dtype=torch.bfloat16)
    prompts = []
    for k in keyframes:
        for side in "RL":
            if arm_prompt == "mask":
                gR = 1.0 if grip is None else grip["R"][k]
                gL = 1.0 if grip is None else grip["L"][k]
                pm = arm_prompt_mask(robot, cam, g_video["R"][k], g_video["L"][k], gR, gL, side)
                if lid_ok[k]:
                    cv2.circle(pm, tuple(int(v) for v in lid_uv[k]), 45, 0, -1)   # keep the can out of the arm prompt
                if pm.sum() < 300:
                    continue
                proc.add_inputs_to_inference_session(sess, frame_idx=int(k), obj_ids=OBJ[side], input_masks=[pm.astype(bool)])
                with torch.inference_mode():
                    model(inference_session=sess, frame_idx=int(k))
                prompts.append({"frame": int(k), "obj": side, "mask_px": int(pm.sum())})
                continue
            box = arm_box(robot, cam, g_video["R"][k], g_video["L"][k], side)
            if box is None:
                continue
            other = "L" if side == "R" else "R"
            pos = hand_points(robot, cam, g_video["R"][k], g_video["L"][k], side)
            neg = hand_points(robot, cam, g_video["R"][k], g_video["L"][k], other)[:1]
            if lid_ok[k]:
                neg.append(list(map(float, lid_uv[k])))
            pts = pos + neg
            kw = {"input_boxes": [[box]]}
            if pts:
                kw.update(input_points=[[pts]], input_labels=[[[1] * len(pos) + [0] * len(neg)]])
            proc.add_inputs_to_inference_session(sess, frame_idx=int(k), obj_ids=OBJ[side], **kw)
            with torch.inference_mode():
                model(inference_session=sess, frame_idx=int(k))   # encode this object's prompt on this frame
            prompts.append({"frame": int(k), "obj": side, **{a: b for a, b in kw.items()}})
        if lid_ok[k]:
            p = [list(map(float, lid_uv[k]))]
            r = red_point(frames[k], lid_uv[k])
            if r is not None:
                p.append(r)
            proc.add_inputs_to_inference_session(sess, frame_idx=int(k), obj_ids=OBJ["can"], input_points=[[p]],
                                                 input_labels=[[[1] * len(p)]])
            with torch.inference_mode():
                model(inference_session=sess, frame_idx=int(k))
            prompts.append({"frame": int(k), "obj": "can", "points": p})
    out = {k: np.zeros((T, H, W), np.uint8) for k in OBJ}
    with torch.inference_mode():
        for o in model.propagate_in_video_iterator(sess, start_frame_idx=int(min(keyframes))):
            m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
            ids = list(sess.obj_ids)
            for name, oid in OBJ.items():
                if oid in ids:
                    out[name][o.frame_idx] = m[ids.index(oid)].squeeze().cpu().numpy().astype(np.uint8)
    # one owner per pixel: can > arm nearer the camera is unknown -> keep can on top, arms exclusive
    both = (out["R"] > 0) & (out["L"] > 0)
    out["L"][both] = 0
    for side in "RL":
        out[side][out["can"] > 0] = 0
    del model
    torch.cuda.empty_cache()
    return out, prompts
