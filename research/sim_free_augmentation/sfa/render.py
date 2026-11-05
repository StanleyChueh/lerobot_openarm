"""Simulation-free video synthesis by layered warping + background inpainting (body_cam only).

For each output frame t:
  1. background: the original frame with every MOVED layer's original pixels replaced by a clean plate. The plate is
     the per-pixel temporal median over frames where SAM2 marks that pixel free of robot arms and can; pixels never
     free are filled with OpenCV Telea inpainting.
  2. layers: each robot link (pixels = SAM2 arm mask, assigned to links by the projected URDF link hulls) and the can
     (SAM2 can mask) is warped by the homography that maps its 3D model points projected at the ORIGINAL pose to the
     same points projected at the NEW pose (rigid 3D motion from FK / the attachment model -> 2D warp).
  3. composite far-to-near by the layer's median depth, feathered alpha.
No generative model is involved; this is a geometry-controlled re-rendering of real pixels. Disocclusions are filled
from the plate, so they are only as good as the plate.
"""

from __future__ import annotations

import cv2
import numpy as np

from .recover import cylinder_points, interp_q
from .robot import Camera, Robot

FEATHER = 2.0


def background_plate(frames: np.ndarray, occ: np.ndarray, dilate=9) -> tuple[np.ndarray, np.ndarray]:
    """frames (T,H,W,3), occ (T,H,W) bool (robot/can) -> plate (H,W,3), never_free (H,W) bool."""
    k = np.ones((dilate, dilate), np.uint8)
    T = len(frames)
    occd = np.stack([cv2.dilate(o.astype(np.uint8), k) > 0 for o in occ])
    plate = np.zeros(frames.shape[1:], np.uint8)
    never = occd.all(0)
    F = frames.astype(np.float32)
    for c in range(3):
        x = np.where(occd, np.nan, F[..., c])
        med = np.nanmedian(x, axis=0)
        plate[..., c] = np.nan_to_num(med, nan=0).astype(np.uint8)
    plate = cv2.inpaint(plate, never.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    return plate, never


def link_label_map(robot: Robot, cam: Camera, qR, qL, gR, gL, arm_mask: dict, H=480, W=640):
    """assign every SAM2 arm pixel to one URDF link of that arm (front-most projected hull; else nearest hull)."""
    robot.fk(qR, qL, gR, gL)
    labels = {}
    for side in "RL":
        hulls = []
        for i, (name, s, P) in enumerate(robot.link_points("hull")):
            if s != side:
                continue
            uv, z = cam.project(P)
            if (z < 0.05).any():
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
            hulls.append((i, name, m, float(np.median(z))))
        lab = np.full((H, W), -1, np.int32)
        if not hulls:
            labels[side] = lab
            continue
        best_d = np.full((H, W), np.inf)
        for i, name, m, z in sorted(hulls, key=lambda h: -h[3]):        # far first, near overwrites
            lab[m > 0] = i
        # pixels of the SAM2 mask outside every hull: nearest hull by distance transform
        inside = lab >= 0
        miss = (arm_mask[side] > 0) & ~inside
        if miss.any():
            for i, name, m, z in hulls:
                d = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)
                upd = miss & (d < best_d)
                lab[upd] = i
                best_d[upd] = d[upd]
        lab[arm_mask[side] == 0] = -1
        labels[side] = lab
    return labels


def homography(src, dst):
    H, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
    if H is None:
        A, _ = cv2.estimateAffine2D(src.astype(np.float32), dst.astype(np.float32))
        H = np.vstack([A, [0, 0, 1]])
    return H


def feather(m):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), FEATHER) if FEATHER > 0 else m.astype(np.float32)


def render_frame(t, frame, plate, robot, cam, q_old, q_new, grip, masks, can_old, can_new, can_axis_local,
                 min_disp=0.5):
    """frame t of the generated video. q_old/q_new: {"R": (7,), "L": (7,)} at the video-aligned time.
    masks: {"R": arm mask, "L": arm mask, "can": can mask}. can_old/new: (4,4) can frames or None."""
    H, W = frame.shape[:2]
    labels = link_label_map(robot, cam, q_old["R"], q_old["L"], grip["R"], grip["L"], masks)
    robot.fk(q_old["R"], q_old["L"], grip["R"], grip["L"])
    pts_old = robot.link_points("sample")
    robot.fk(q_new["R"], q_new["L"], grip["R"], grip["L"])
    pts_new = robot.link_points("sample")
    layers, removed = [], np.zeros((H, W), bool)
    info = {"homographies": {}}
    for i, ((name, side, Po), (_, _, Pn)) in enumerate(zip(pts_old, pts_new)):
        if side == "B":
            continue
        m = (labels[side] == i)
        if not m.any():
            continue
        uo, zo = cam.project(Po)
        un, zn = cam.project(Pn)
        disp = np.linalg.norm(un - uo, axis=1).max()
        if disp < min_disp:
            continue
        Hm = homography(uo, un)
        layers.append((float(np.median(zn)), m, Hm, f"{side}_{name}"))
        removed |= m
        info["homographies"][f"{side}_{name}"] = Hm.tolist()
    if can_old is not None and can_new is not None and masks["can"].any():
        P = cylinder_points(can_axis_local, 24, 6)
        Po = (can_old[:3, :3] @ P.T).T + can_old[:3, 3]
        Pn = (can_new[:3, :3] @ P.T).T + can_new[:3, 3]
        uo, zo = cam.project(Po)
        un, zn = cam.project(Pn)
        if np.linalg.norm(un - uo, axis=1).max() >= min_disp:
            Hc = homography(uo, un)
            layers.append((float(np.median(zn)), masks["can"] > 0, Hc, "can"))
            removed |= masks["can"] > 0
            info["homographies"]["can"] = Hc.tolist()
    # 1. background with moved layers removed
    rem = cv2.dilate(removed.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = feather(rem)[..., None]
    out = frame.astype(np.float32) * (1 - a) + plate.astype(np.float32) * a
    # static (unmoved) arm/can pixels must stay on top of the plate where a moved layer was removed
    static = ((masks["R"] > 0) | (masks["L"] > 0) | (masks["can"] > 0)) & ~removed
    out[static] = frame[static]
    # 2-3. warped layers far -> near
    new_mask = np.zeros((H, W), bool)
    for z, m, Hm, name in sorted(layers, key=lambda l: -l[0]):
        wf = cv2.warpPerspective(frame, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wm = cv2.warpPerspective(m.astype(np.float32), Hm, (W, H), flags=cv2.INTER_LINEAR)
        al = np.clip(cv2.GaussianBlur(wm, (0, 0), 1.0), 0, 1)[..., None]
        out = out * (1 - al) + wf.astype(np.float32) * al
        new_mask |= wm > 0.5
    info["n_layers"] = len(layers)
    info["moved_px"] = int(removed.sum())
    return np.clip(out, 0, 255).astype(np.uint8), new_mask, info


# ----------------------------------------------------------------------------- keypoint control panel
SKELETON = ["link3", "link4", "link5", "link6", "link7", "hand", "hand_tcp"]


def keypoints_3d(robot: Robot, qR, qL, gR, gL, can):
    import pinocchio as pin
    robot.fk(qR, qL, gR, gL)
    kp = {}
    for s, side in (("R", "right"), ("L", "left")):
        kp[s] = np.stack([robot.d.oMf[robot.m.getFrameId(f"openarm_{side}_{n}")].translation for n in SKELETON])
        kp[s + "_fingers"] = np.stack([robot.d.oMf[robot.m.getFrameId(f"openarm_{side}_{n}")].translation
                                       for n in ("left_finger", "right_finger")])
    if can is not None:
        from .recover import cylinder_points
        a = can[:3, 2]
        kp["can"] = np.stack([can[:3, 3], can[:3, 3] + 0.23 * a])
        P = cylinder_points([0, 0, 1.0], 24, 2)
        kp["can_surface"] = (can[:3, :3] @ P.T).T + can[:3, 3]
    return kp


def draw_keypoints(img, cam, kp, color_R=(255, 80, 80), color_L=(80, 160, 255), color_can=(255, 220, 0)):
    out = img.copy()
    for s, col in (("R", color_R), ("L", color_L)):
        uv, z = cam.project(kp[s])
        pts = [tuple(int(round(v)) for v in p) for p in uv]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, a, b, col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, p, 4, col, -1, cv2.LINE_AA)
        fu, _ = cam.project(kp[s + "_fingers"])
        for p in fu:
            cv2.line(out, pts[-2], tuple(int(round(v)) for v in p), col, 2, cv2.LINE_AA)
        cv2.circle(out, pts[-1], 7, col, 2, cv2.LINE_AA)       # TCP
    if "can" in kp:
        su, _ = cam.project(kp["can_surface"])
        cv2.polylines(out, [cv2.convexHull(su.astype(np.float32)).astype(np.int32)], True, color_can, 2, cv2.LINE_AA)
        uv, _ = cam.project(kp["can"])
        a, b = (tuple(int(round(v)) for v in p) for p in uv)
        cv2.line(out, a, b, color_can, 1, cv2.LINE_AA)
        cv2.circle(out, a, 6, color_can, -1, cv2.LINE_AA)
    return out


def write_mp4(path, frames, fps=30, crf=20):
    import av
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=int(fps))
        st.width, st.height = frames[0].shape[1], frames[0].shape[0]
        st.pix_fmt = "yuv420p"
        st.options = {"crf": str(crf), "preset": "medium"}
        for f in frames:
            for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def label(img, text, pos=(8, 22)):
    out = img.copy()
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
