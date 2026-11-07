"""Review sheets for the manual Q5 (visual transfer) and Q6 (frame-0 can mask) gates.

outputs/review/mask_qa.jpg            - all 50 frame-0 can crops: SAM contour, human click (x), centroid (+)
outputs/review/transfer_XX_YY.jpg     - per episode: body keyframes (start, rc, rc+20, lc, ro+10, last) and, for
                                        non-locked records, right wrist @ rc+20 and left wrist @ ro+10 / last
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

REV = C.OUT / "review"
KEY = C.OUT / "keyframes"


def label(im, txt, scale=0.5, org=(4, 16)):
    cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def mask_qa(F):
    ds = C.pool()
    tiles = []
    for e, f in F.items():
        im = ds.decode(C.BODY, e, [0])[0].copy()
        z = np.load(KEY / f"track_{e:02d}.npz")
        m = np.unpackbits(z["mask0"], axis=-1)[:, :im.shape[1]].astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (0, 255, 0), 1)
        u, v = map(int, f["click_uv"])
        cv2.drawMarker(im, (u, v), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        cu, cv_ = int(f["can_u"]), int(f["can_v"])
        cv2.drawMarker(im, (cu, cv_), (255, 255, 0), cv2.MARKER_CROSS, 10, 2)
        y0, x0 = max(0, cv_ - 90), max(0, cu - 90)
        crop = cv2.resize(im[y0:y0 + 180, x0:x0 + 180], (180, 180))
        tiles.append(label(crop, f"{e}{' L' if f['v4a_locked'] else ''} a={f['can_area_px']}", 0.45))
    rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
    cv2.imwrite(str(REV / "mask_qa.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def transfer(F, per=6):
    names = ["start", "rc", "rc+20", "lc", "ro+10", "last"]
    eps = list(F)
    for i in range(0, len(eps), per):
        rows = []
        for e in eps[i:i + per]:
            z = np.load(KEY / f"keys_{e:02d}.npz")
            idx = list(z["idx"])
            tiles = [label(cv2.resize(z["body"][k], (240, 180)), f"ep{e} {names[k]} t={idx[k]}") for k in range(len(idx))]
            if "right" in z:
                tiles.append(label(cv2.resize(z["right"][2], (240, 180)), "R-wrist rc+20"))
                tiles.append(label(cv2.resize(z["left"][4], (240, 180)), "L-wrist ro+10"))
                tiles.append(label(cv2.resize(z["left"][5], (240, 180)), "L-wrist last"))
            else:
                blank = np.full((180, 240, 3), 40, np.uint8)
                tiles += [label(blank.copy(), "v4A locked: wrist withheld")] * 3
            rows.append(np.hstack(tiles))
        out = REV / f"transfer_{eps[i]:02d}_{eps[min(i + per, len(eps)) - 1]:02d}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    REV.mkdir(parents=True, exist_ok=True)
    F = {int(k): v for k, v in json.loads((C.OUT / "features_raw.json").read_text()).items()}
    mask_qa(F)
    transfer(F)
    print(sorted(p.name for p in REV.iterdir()))


if __name__ == "__main__":
    main()
