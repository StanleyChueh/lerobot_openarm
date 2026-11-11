import sys
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "render_v4"))
import mgw
from sfa3 import recon3
V31 = mgw.SFA / "v3_1/outputs"
E, key, t, dev, out = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
rec = recon3.build(E, with_plate=True, log=lambda *a: None)
R = mgw.MGWForward(rec, device=dev)
z = np.load(V31 / f"ep{E}/candidates/{key}.npz"); vid = mgw.candidate_video(rec, z)
j, q, g, can = vid[t]; img, info = R.render(j, q, g, can)
S = R.src_points(j); part = S["part"].cpu().numpy(); lab = S["lab"].cpu().numpy()
Mt = R.placements(q, g, can); _, lt, _ = R.zbuf(Mt); lt = lt.cpu().numpy()
src = rec.F[j].copy()
ov = src.copy(); ov[(part >= 0) & (lab < 0)] = (255, 0, 255)          # masked pixels outside the source mesh render
ed = cv2.Canny((lab >= 0).astype(np.uint8) * 255, 50, 150) > 0; ov[ed] = (0, 255, 0)
ov2 = img.copy(); ed2 = cv2.Canny((lt >= 0).astype(np.uint8) * 255, 50, 150) > 0; ov2[ed2] = (0, 255, 0)
diff = np.abs(src.astype(int) - rec.plate.astype(int)).max(2)
cv2.imwrite(out, cv2.cvtColor(np.concatenate([np.concatenate([src, img], 1), np.concatenate([ov, ov2], 1), np.concatenate([rec.plate, np.repeat(np.clip(diff * 3, 0, 255).astype(np.uint8)[..., None], 3, 2)], 1)], 0), cv2.COLOR_RGB2BGR))
print(info)
