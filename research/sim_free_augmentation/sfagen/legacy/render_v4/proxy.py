"""No-reference render-quality score: agreement between the rendered image's robot/can foreground (difference from the
clean plate) and the expected target silhouette (z-buffered mesh render of the target pose), inside a region of
interest around both. Higher = better (IoU)."""
import numpy as np, cv2


def score(img, plate, target_mask, roi_extra=None, thr=25):
    fg = np.abs(img.astype(np.int16) - plate.astype(np.int16)).max(2) > thr
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    roi = target_mask.copy()
    if roi_extra is not None:
        roi |= roi_extra
    roi = cv2.dilate(roi.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    f = fg & roi
    return float((f & target_mask).sum() / max(1, (f | target_mask).sum()))
