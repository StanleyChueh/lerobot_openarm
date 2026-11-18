"""Robot masks by SAM2.1 video segmentation (transformers Sam2VideoModel), prompted automatically from the robot model:
every `every` frames, each arm gets a MASK prompt = union of the projected convex hulls of its visual geometries (the
camera model only needs to be roughly right; SAM2 snaps to the real silhouette), with the neighbourhood of every object
removed from the prompt. Masks are propagated through the whole video. No manual prompt, no label. Robot-agnostic: the
arm geometries come from the URDF ownership in Robot.geom_arm. Cached per episode."""
from __future__ import annotations
import cv2, numpy as np, torch

MODEL = "facebook/sam2.1-hiera-small"


def arm_prompt(rec, j, side, exclude_links=(), cut="neighbourhood"):
    q = {s: rec.gv[s][j] for s in rec.arms}; g = {s: float(rec.grip_v[s][j]) for s in rec.grip_v}
    rec.robot.fk(q, g); m = np.zeros((rec.H, rec.W), np.uint8)
    for name, s, P in rec.robot.link_points():
        if s != side or any(x in name for x in exclude_links):
            continue
        uv, z = rec.cam.project(P)
        if (z > 0.15).all():
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    if cut == "none":
        return m
    if cut == "depth":                                    # R05: keep the gripper at the object; drop only what a proxy hides
        lab = rec.scene_labels(j); r = max(1, round(5 * rec.W / 640))
        hidden = cv2.dilate((lab >= rec.G.ng).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
        m[hidden & ~((lab >= 0) & (lab < rec.G.ng))] = 0
        return m
    for k in range(len(rec.objects)):                     # keep objects out of the arm prompts
        m[rec.obj_sil(k, rec.obj_pose(k, j + rec.lag), dil=25)] = 0
    return m


def arm_prompt_select(rec, j, side, exclude_links, model, proc, dev):
    """R06: arm prompt whose edges come from the image. Per link: SAM2's candidate (part / whole / group) at the innermost point
    of the link's projected hull that agrees best with that hull; the hull itself where none agrees (IoU < 0.3) or the link is
    tiny. Filled hulls alone make SAM2 include the background between links."""
    rec.robot.fk({s: rec.gv[s][j] for s in rec.arms}, {s: float(rec.grip_v[s][j]) for s in rec.grip_v}); hulls = []
    for name, s, P in rec.robot.link_points():
        uv, z = rec.cam.project(P)
        if s == side and not any(x in name for x in exclude_links) and (z > 0.15).all():
            h = np.zeros((rec.H, rec.W), np.uint8); cv2.fillConvexPoly(h, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
            if h.any():
                hulls.append(h > 0)
    m = np.zeros((rec.H, rec.W), bool); mn = 150 * (rec.W / 640) ** 2; big = [h for h in hulls if h.sum() >= mn]
    for h in hulls:
        if h.sum() < mn:
            m |= h
    if big:
        pts = []
        for h in big:
            dt = cv2.distanceTransform(h.astype(np.uint8), cv2.DIST_L2, 5); y, x = np.unravel_index(dt.argmax(), dt.shape); pts.append([[float(x), float(y)]])
        inp = proc(images=rec.F[j], input_points=[pts], input_labels=[[[1]] * len(pts)], return_tensors="pt").to(dev)
        with torch.inference_mode():
            o = model(**inp, multimask_output=True)
        cand = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0].numpy() > 0          # [links, 3, H, W]
        for h, c in zip(big, cand):
            iou = [float((x & h).sum() / max(1, (x | h).sum())) for x in c]; m |= c[int(np.argmax(iou))] if max(iou) >= 0.3 else h
    return m.astype(np.uint8)


def cache_name(rc, e, has_objects):
    """SAM2 mask cache of one episode; every setting that changes the masks is in the name (old caches are never reused)."""
    cut = rc.get("prompt_object_cut", "neighbourhood"); so = rc.get("object_masks", "plate") == "sam2" and has_objects; op = rc.get("object_prompt", "mask")
    ap = rc.get("arm_prompt", "hull")
    return f"sam2_ep{e}.npz" if cut == "neighbourhood" and not so and ap == "hull" else f"sam2_{cut}{'_obj' if so else ''}{'_' + op if so and op != 'mask' else ''}{'_arm' + ap if ap != 'hull' else ''}_ep{e}.npz"


def object_prompt(rec, j, k):
    """R06: (prompt, clean). prompt = proxy silhouette that differs from the clean plate, minus the arms' projected hulls;
    clean = no arm hull within 25 px (x W/640) of the proxy silhouette (the object is seen alone)."""
    sc = rec.W / 640; sil = rec.obj_sil(k, rec.obj_pose(k, j + rec.lag)); hull = np.any([arm_prompt(rec, j, s, cut="none") > 0 for s in rec.arms], 0).astype(np.uint8)
    grow = lambda r: cv2.dilate(hull, np.ones((2 * max(1, round(r * sc)) + 1,) * 2, np.uint8)) > 0  # noqa: E731
    return sil & rec._fg(j, rec.plate, 25) & ~grow(5), not (sil & grow(25)).any()


def pick_object_mask(model, proc, frame, pm, dev):
    """R06: SAM2 offers three masks for one point (part / whole / group). Take the one that agrees best with the proxy-based
    prompt: the proxy chooses the scale, the image gives the edge. Falls back to the prompt if none agrees (IoU < 0.3)."""
    dt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 5); y, x = np.unravel_index(dt.argmax(), dt.shape)
    inp = proc(images=frame, input_points=[[[[float(x), float(y)]]]], input_labels=[[[1]]], return_tensors="pt").to(dev)
    with torch.inference_mode():
        o = model(**inp, multimask_output=True)
    c = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0
    iou = [float((m & pm).sum() / max(1, (m | pm).sum())) for m in c]
    return c[int(np.argmax(iou))] if max(iou) >= 0.3 else pm


def footprint_mask(frame, uv, changed, device="cuda:0", model_name=MODEL, min_px=150):
    """R07: the object at frame 0 = the largest SAM2 candidate at the grasp point whose pixels mostly (>= 80 %) change after
    the pickup. Robust where the scene also changes (wiping, shadows, moved cables): those make `changed` too large, but no
    SAM2 segment that includes them is mostly changed. None if no candidate qualifies."""
    from transformers import Sam2Model, Sam2Processor
    model = Sam2Model.from_pretrained(model_name).to(device).eval(); proc = Sam2Processor.from_pretrained(model_name)
    inp = proc(images=frame, input_points=[[[[float(uv[0]), float(uv[1])]]]], input_labels=[[[1]]], return_tensors="pt").to(device)
    with torch.inference_mode():
        o = model(**inp, multimask_output=True)
    cand = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy() > 0; del model; torch.cuda.empty_cache()
    ok = [c for c in cand if c.sum() >= min_px and (c & changed).sum() >= 0.8 * c.sum()]
    return max(ok, key=lambda c: c.sum()) if ok else None


def owners(logits):
    """one owner per pixel: the track with the highest SAM2 logit, where that logit is positive. logits [n, H, W] -> bool [n, H, W]"""
    return (logits > 0) & (logits.argmax(0)[None] == torch.arange(len(logits), device=logits.device)[:, None, None])


def segment_arms(rec, device="cuda:0", every=15, exclude_links=(), model_name=MODEL, chunk=150, cut="neighbourhood", objects=False, object_prompt_kind="mask", arm_prompt_kind="hull"):
    """chunks of `chunk` frames, each its own SAM2 session (prompts are re-issued from the URDF every `every` frames, so
    chunks need no continuity): GPU memory is independent of the episode length."""
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    model = Sam2VideoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    proc = Sam2VideoProcessor.from_pretrained(model_name)
    ids = {s: i + 1 for i, s in enumerate(rec.arms)}; n_prompts = 0; img = None
    out = {s: np.zeros((rec.T, rec.H, rec.W), np.uint8) for s in rec.arms}
    if objects:                                          # R06: every object is its own track in the same session; tracks compete per pixel
        ids.update({f"obj{k}": len(rec.arms) + 1 + k for k in range(len(rec.objects))})
        out.update({f"obj{k}": np.zeros((rec.T, rec.H, rec.W), np.uint8) for k in range(len(rec.objects))})
        out.update({f"objok{k}": np.zeros(rec.T, bool) for k in range(len(rec.objects))})
    for c0 in range(0, rec.T, chunk):
        c1 = min(rec.T, c0 + chunk)
        sess = proc.init_video_session(video=list(rec.F[c0:c1]), inference_device=device, dtype=torch.bfloat16)
        keys = list(range(0, c1 - c0, every)); used = []
        for k in keys:
            for s in rec.arms:
                if arm_prompt_kind == "select":
                    if img is None:
                        from transformers import Sam2Model, Sam2Processor
                        img = (Sam2Model.from_pretrained(model_name).to(device).eval(), Sam2Processor.from_pretrained(model_name))
                    pm = arm_prompt_select(rec, c0 + k, s, exclude_links, *img, device)
                else:
                    pm = arm_prompt(rec, c0 + k, s, exclude_links, cut)
                if pm.sum() < 300:
                    continue
                proc.add_inputs_to_inference_session(sess, frame_idx=int(k), obj_ids=ids[s], input_masks=[pm.astype(bool)])
                with torch.inference_mode():
                    model(inference_session=sess, frame_idx=int(k))
                n_prompts += 1; used.append(k)
        for k in range(len(rec.objects)) if objects else ():
            pr = [(kf, *object_prompt(rec, c0 + kf, k)) for kf in keys]; mn = 150 * (rec.W / 640) ** 2
            pr = [x for x in pr if x[1].sum() >= mn]; pr = [x for x in pr if x[2]] or pr[:1]     # clean views; if none, the first usable one
            for kf, pm, _ in pr:
                if object_prompt_kind == "select":
                    if img is None:
                        from transformers import Sam2Model, Sam2Processor
                        img = (Sam2Model.from_pretrained(model_name).to(device).eval(), Sam2Processor.from_pretrained(model_name))
                    pm = pick_object_mask(*img, rec.F[c0 + kf], pm, device)
                if object_prompt_kind in ("mask", "select"):
                    proc.add_inputs_to_inference_session(sess, frame_idx=int(kf), obj_ids=ids[f"obj{k}"], input_masks=[pm])
                else:                                    # one point deep inside the prompt (and its box): SAM2 finds the object's own edge
                    dt = cv2.distanceTransform(pm.astype(np.uint8), cv2.DIST_L2, 5); y, x = np.unravel_index(dt.argmax(), dt.shape); ys, xs = np.nonzero(pm)
                    box = {"input_boxes": [[[float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]]]} if object_prompt_kind == "box" else {}
                    pts = [[float(x), float(y)]]
                    if object_prompt_kind == "points":   # up to 5 points spread over the inner part: one point can select only a part (inside of a cup)
                        cy, cx = np.nonzero(dt >= 0.3 * dt.max()); P = np.c_[cx, cy].astype(np.float32); sel = [int(dt[cy, cx].argmax())]
                        for _ in range(min(4, len(P) - 1)):
                            sel.append(int(np.linalg.norm(P[:, None] - P[sel][None], axis=2).min(1).argmax()))
                        pts = P[sel].tolist()
                    proc.add_inputs_to_inference_session(sess, frame_idx=int(kf), obj_ids=ids[f"obj{k}"], input_points=[[pts]], input_labels=[[[1] * len(pts)]], **box)
                with torch.inference_mode():
                    model(inference_session=sess, frame_idx=int(kf))
                n_prompts += 1; used.append(kf)
            out[f"objok{k}"][c0:c1] = bool(pr)
        if not used:
            del sess; continue
        with torch.inference_mode():
            for o in model.propagate_in_video_iterator(sess, start_frame_idx=int(min(used))):
                m = proc.post_process_masks([o.pred_masks], original_sizes=[[rec.H, rec.W]], binarize=not objects)[0]
                sid = list(sess.obj_ids)
                if objects:
                    m = owners(m.reshape(len(sid), rec.H, rec.W).float())
                for s, oid in ids.items():
                    if oid in sid:
                        out[s][c0 + o.frame_idx] = m[sid.index(oid)].squeeze().cpu().numpy().astype(np.uint8)
        del sess; torch.cuda.empty_cache()
    for i, a in enumerate(() if objects else rec.arms):  # one owner per pixel
        for b in rec.arms[i + 1:]:
            out[b][(out[a] > 0) & (out[b] > 0)] = 0
    del model; torch.cuda.empty_cache()
    return out, {"model": model_name, "keyframe_every": every, "chunk": chunk, "prompts": n_prompts}
