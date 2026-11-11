import sys; sys.path.insert(0, '.')
import numpy as np, cv2, srender, srecon
E, key, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
rec = srecon.StationRecon(E, device="cuda:0"); R = srender.renderer(rec, "cuda:0", "station")
z = np.load(f"candidates_station/ep{E}/{key}.npz"); vid = srender.mgw.candidate_video(rec, z)
ts = [0, 70, rec.tLg - 5, rec.tLg + 40, rec.tLg + 90, 230]
cols = []
for t in ts:
    j, q, g, can = vid[t]; img, info = R.render(j, q, g, can); print(t, info)
    cols.append(np.concatenate([rec.F[j], img], 0))
cv2.imwrite(out, cv2.cvtColor(cv2.resize(np.concatenate(cols, 1), None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
