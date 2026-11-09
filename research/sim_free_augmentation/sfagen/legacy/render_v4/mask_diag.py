import sys, numpy as np, cv2
sys.path.insert(0, '.')
import mgw
from sfa3 import recon3
OUT = "/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad/"
rec = recon3.build(8, with_plate=True, log=lambda *a: None)
R = mgw.MGW(rec, device="cuda:1")
tiles = []
for j in (43, 143, 183):
    S = R.source_state(j); sam = S["sam"].cpu().numpy()
    im = rec.F[j].copy()
    for k, col in ((0, (255, 60, 60)), (1, (60, 120, 255)), (2, (60, 255, 60))):
        m = sam == k; im[m] = (0.45 * im[m] + 0.55 * np.array(col)).astype(np.uint8)
        cs, _ = cv2.findContours((rec.masks[["R", "L", "can"][k]][j] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (255, 255, 0), 1)
    tiles.append(im)
cv2.imwrite(OUT + "robust_masks.jpg", np.concatenate(tiles, 1)[..., ::-1])
