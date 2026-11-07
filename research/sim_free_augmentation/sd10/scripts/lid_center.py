"""Amendment A1 position feature: frame-0 lid-ellipse centre (body camera).

1. Within the SAM2.1 frame-0 can mask, find the non-red region nearest the top of the mask (the wrapper is saturated
   red); its upper part gives one positive point on the lid. Three red-wrapper pixels lowest in the mask give
   negative points (kept only if outside the lid box). The lid box spans the full silhouette width over its top 30 %
   and is as tall as it is wide; the positive point is the box centre.
2. Re-prompt SAM2.1 (image) with the box and points; keep the proposal that contains the positive point, excludes the
   negatives and has the best score. Fit an ellipse to its convex hull; the ellipse centre is the feature.
Rationale: SAM returns lid-only or lid+body masks inconsistently, and under the oblique view the whole-mask centroid
shifts with the visible body length; the lid top is a fixed-height landmark.

Writes outputs/lid_centers.json and outputs/review/lid_qa.jpg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sd10 import common as C  # noqa: E402

KEY = C.OUT / "keyframes"
RED_H, RED_S = (8, 170), 110          # red wrapper: hue <= 8 or >= 170 (OpenCV 0-179) and saturation >= 110


def lid(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    red = ((h <= RED_H[0]) | (h >= RED_H[1])) & (s >= RED_S)
    cand = (mask & ~red).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        return None
    ys = np.nonzero(mask)[0]
    top, height = ys.min(), ys.max() - ys.min() + 1
    best = None
    for k in range(1, n):
        a = st[k, cv2.CC_STAT_AREA]
        if a < 1500:
            continue
        rel_top = (st[k, cv2.CC_STAT_TOP] - top) / height
        if rel_top > 0.25:                       # lid must start in the top quarter of the can mask
            continue
        if best is None or a > st[best, cv2.CC_STAT_AREA]:
            best = k
    if best is None:
        return None
    comp = (lab == best).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(hull)
    ell_area = np.pi * MA * ma / 4
    return {"lid_u": float(cx), "lid_v": float(cy), "lid_axes_px": [float(MA), float(ma)], "lid_angle_deg": float(ang),
            "lid_area_px": int(comp.sum()), "lid_fill_ratio": float(comp.sum() / ell_area)}


def prompts(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    red = ((h <= RED_H[0]) | (h >= RED_H[1])) & (s >= RED_S) & mask
    r = lid(img, mask)
    if r is None:
        return None
    ys, xs = np.nonzero(mask)
    top = ys.min()
    # positive: centre of the colour-lid component's upper 40 %
    cand = (mask & ~red).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
    k = max(range(1, n), key=lambda k: st[k, cv2.CC_STAT_AREA] if st[k, cv2.CC_STAT_TOP] - top < 0.25 * (ys.max() - top + 1) else -1)
    cy, cx = np.nonzero(lab == k)
    sel = cy <= cy.min() + 0.4 * (cy.max() - cy.min())
    pos = [float(cx[sel].mean()), float(cy[sel].mean())]
    ry, rx = np.nonzero(red)
    neg = []
    if len(ry):
        o = np.argsort(-ry)[: max(1, len(ry) // 20)]           # lowest 5 % of red pixels
        pick = o[np.linspace(0, len(o) - 1, 3).astype(int)]
        neg = [[float(rx[i]), float(ry[i])] for i in pick]
    return pos, neg


def lid_box(mask):
    """Lid box from the whole-can silhouette: full width of its top 30 %, height = that width (clipped)."""
    ys, xs = np.nonzero(mask)
    top, bot = ys.min(), ys.max()
    sel = ys <= top + 0.3 * (bot - top)
    x0, x1 = xs[sel].min(), xs[sel].max()
    w = x1 - x0
    return [float(x0), float(top), float(x1), float(min(bot, top + w))]


def sam_lid(sam, img, pos, neg, box):
    import torch
    neg = [p for p in neg if not (box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3])]
    pos = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    pts, labs = [pos] + neg, [1] + [0] * len(neg)
    with torch.inference_mode():
        inp = sam.ip(images=img, input_points=[[pts]], input_labels=[[labs]], input_boxes=[[box]],
                     return_tensors="pt").to(sam.dev)
        out = sam.im(**inp, multimask_output=True)
        M = sam.ip.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0][0].numpy().astype(bool)
        S = out.iou_scores[0, 0].float().cpu().numpy()
    best = None
    for k in range(len(M)):
        ok = M[k][int(pos[1]), int(pos[0])] and not any(M[k][int(v), int(u)] for u, v in neg) and 2500 <= M[k].sum() <= 20000
        if ok and (best is None or S[k] > S[best]):
            best = k
    if best is None:
        return None, S.tolist()
    m = M[best].astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(cv2.convexHull(c))
    return {"lid_u": float(cx), "lid_v": float(cy), "lid_axes_px": [float(MA), float(ma)], "lid_angle_deg": float(ang),
            "lid_area_px": int(m.sum()), "lid_fill_ratio": float(m.sum() / (np.pi * MA * ma / 4)),
            "sam_choice": int(best), "sam_scores": S.tolist(), "pos_prompt": pos, "neg_prompts": neg, "box_prompt": box}, S.tolist()


def main():
    sys.path.insert(0, str(HERE / "scripts"))
    from extract_features import Sam
    sam = Sam("cuda:0")
    ds = C.pool()
    F = {int(k): v for k, v in json.loads((C.OUT / "features_raw.json").read_text()).items()}
    out, tiles = {}, []
    for e, f in F.items():
        im = ds.decode(C.BODY, e, [0])[0]
        z = np.load(KEY / f"track_{e:02d}.npz")
        m = np.unpackbits(z["mask0"], axis=-1)[:, :im.shape[1]].astype(bool)
        pr = prompts(im, m)
        r = sam_lid(sam, im, *pr, lid_box(m))[0] if pr else None
        out[e] = r
        vis = im.copy()
        if r is not None:
            cv2.ellipse(vis, ((r["lid_u"], r["lid_v"]), tuple(r["lid_axes_px"]), r["lid_angle_deg"]), (0, 255, 0), 1)
            cv2.drawMarker(vis, (int(r["lid_u"]), int(r["lid_v"])), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
        u, v = map(int, f["click_uv"])
        cv2.drawMarker(vis, (u, v), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        cu, cv_ = (int(r["lid_u"]), int(r["lid_v"])) if r else (u, v)
        y0, x0 = max(0, cv_ - 70), max(0, cu - 70)
        crop = cv2.resize(vis[y0:y0 + 140, x0:x0 + 140], (180, 180))
        txt = f"{e}{' L' if f['v4a_locked'] else ''} " + (f"fill={r['lid_fill_ratio']:.2f}" if r else "NO LID")
        cv2.putText(crop, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(crop, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
    cv2.imwrite(str(C.OUT / "review/lid_qa.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    C.dump(out, C.OUT / "lid_centers.json")
    print("no lid:", [e for e, r in out.items() if r is None])
    print("fill ratio min:", min(r["lid_fill_ratio"] for r in out.values() if r), "axes range:",
          min(r["lid_axes_px"][1] for r in out.values() if r), max(r["lid_axes_px"][1] for r in out.values() if r))


if __name__ == "__main__":
    main()
