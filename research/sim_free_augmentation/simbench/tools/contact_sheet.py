"""Source contact sheet: one body-camera frame per episode (at FRAC of its length) for each task config.
Usage: contact_sheet.py OUT.jpg FRAC CONFIG [CONFIG ...]"""
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
out, frac, rows = sys.argv[1], float(sys.argv[2]), []
for c in sys.argv[3:]:
    ctx = Context(c); cam = ctx.cfg["source"]["body_camera"]; tiles = []
    for e in ctx.episodes():
        F = ctx.src.decode(cam, e); f = np.ascontiguousarray(F[int(frac * (len(F) - 1))][..., ::-1]); f = cv2.resize(f, (256, 192))
        cv2.putText(f, f"{Path(c).stem.replace('_base', '')} ep{e}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1); tiles.append(f)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
