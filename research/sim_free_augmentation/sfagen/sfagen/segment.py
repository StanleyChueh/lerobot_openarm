"""Robot masks by SAM2.1 video segmentation (transformers Sam2VideoModel), prompted automatically from the robot model:
every `every` frames, each arm gets a MASK prompt = union of the projected convex hulls of its visual geometries (the
camera model only needs to be roughly right; SAM2 snaps to the real silhouette), with the neighbourhood of every object
removed from the prompt. Masks are propagated through the whole video. No manual prompt, no label. Robot-agnostic: the
arm geometries come from the URDF ownership in Robot.geom_arm. Cached per episode."""
from __future__ import annotations
import cv2, numpy as np, torch

MODEL = "facebook/sam2.1-hiera-small"


def arm_prompt(rec, j, side, exclude_links=()):
    q = {s: rec.gv[s][j] for s in rec.arms}; g = {s: float(rec.grip_v[s][j]) for s in rec.grip_v}
    rec.robot.fk(q, g); m = np.zeros((rec.H, rec.W), np.uint8)
    for name, s, P in rec.robot.link_points():
        if s != side or any(x in name for x in exclude_links):
            continue
        uv, z = rec.cam.project(P)
        if (z > 0.15).all():
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    for k in range(len(rec.objects)):                     # keep objects out of the arm prompts
        m[rec.obj_sil(k, rec.obj_pose(k, j + rec.lag), dil=25)] = 0
    return m


def segment_arms(rec, device="cuda:0", every=15, exclude_links=(), model_name=MODEL, chunk=150):
    """chunks of `chunk` frames, each its own SAM2 session (prompts are re-issued from the URDF every `every` frames, so
    chunks need no continuity): GPU memory is independent of the episode length."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    model = Sam2VideoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    proc = Sam2VideoProcessor.from_pretrained(model_name)
    ids = {s: i + 1 for i, s in enumerate(rec.arms)}; n_prompts = 0
    out = {s: np.zeros((rec.T, rec.H, rec.W), np.uint8) for s in rec.arms}
    for c0 in range(0, rec.T, chunk):
        c1 = min(rec.T, c0 + chunk)
        sess = proc.init_video_session(video=list(rec.F[c0:c1]), inference_device=device, dtype=torch.bfloat16)
        keys = list(range(0, c1 - c0, every)); used = []
        for k in keys:
            for s in rec.arms:
                pm = arm_prompt(rec, c0 + k, s, exclude_links)
                if pm.sum() < 300:
                    continue
                proc.add_inputs_to_inference_session(sess, frame_idx=int(k), obj_ids=ids[s], input_masks=[pm.astype(bool)])
                with torch.inference_mode():
                    model(inference_session=sess, frame_idx=int(k))
                n_prompts += 1; used.append(k)
        if not used:
            del sess; continue
        with torch.inference_mode():
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=int(min(used))):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[rec.H, rec.W]], binarize=True)[0]
                sid = list(sess.obj_ids)
                for s, oid in ids.items():
                    if oid in sid:
                        out[s][c0 + o.frame_idx] = m[sid.index(oid)].squeeze().cpu().numpy().astype(np.uint8)
        del sess; torch.cuda.empty_cache()
    for i, a in enumerate(rec.arms):                     # one owner per pixel
        for b in rec.arms[i + 1:]:
            out[b][(out[a] > 0) & (out[b] > 0)] = 0
    del model; torch.cuda.empty_cache()
    return out, {"model": model_name, "keyframe_every": every, "chunk": chunk, "prompts": n_prompts}
