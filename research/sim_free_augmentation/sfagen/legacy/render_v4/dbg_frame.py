import json, sys, numpy as np, cv2, torch
sys.path.insert(0, '.')
import mgw
from sfa3 import recon3
OUT = "/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad/"
V31 = mgw.SFA / "v3_1/outputs"
E, key, t = 0, "init_x-6__G+__s1.0", int(sys.argv[1])
sel = {(e["E"], e["key"]): e for e in json.load(open("../3cam_v3/selection.json"))["selected"]}
e = sel[(E, key)]; a, b = e["rows"]
rec = recon3.build(E, with_plate=True, log=lambda *a: None)
R = mgw.MGWForward(rec, device="cuda:1")
z = np.load(V31 / f"ep{E}/candidates/{key}.npz"); vid = mgw.candidate_video(rec, z)
j, q, g, can = vid[a + t]
img, info = R.render(j, q, g, can); print(info, "src", j)
S = R.src_points(j); part = S["part"].cpu().numpy()
rng = np.random.default_rng(3); pal = rng.integers(60, 255, (40, 3)).astype(np.uint8)
pm = rec.F[j].copy(); m = part >= 0; pm[m] = (0.4 * pm[m] + 0.6 * pal[part[m] % 40]).astype(np.uint8); pm[part == -2] = (255, 0, 255)
Mt = R.placements(q, g, can); d, lab, _ = R.zbuf(Mt); lab = lab.cpu().numpy()
tm = img.copy(); m2 = lab >= 0; tm[m2] = (0.4 * tm[m2] + 0.6 * pal[lab[m2] % 40]).astype(np.uint8)
Ms = R.placements({s: rec.gv[s][j] for s in "RL"}, {s: float(rec.grip_v[s][j]) for s in "RL"}, rec.source_can(R.robot, j + rec.lag))
d2, lab2, _ = R.zbuf(Ms); lab2 = lab2.cpu().numpy()
sm = rec.F[j].copy(); m3 = lab2 >= 0; sm[m3] = (0.4 * sm[m3] + 0.6 * pal[lab2[m3] % 40]).astype(np.uint8)
cv2.imwrite(OUT + f"dbg_{t}.jpg", cv2.cvtColor(np.concatenate([np.concatenate([rec.F[j], sm], 1), np.concatenate([pm, tm], 1)], 0), cv2.COLOR_RGB2BGR))
