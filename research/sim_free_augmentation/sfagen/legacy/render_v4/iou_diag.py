import sys, numpy as np, torch, cv2
from pathlib import Path
sys.path.insert(0, '.')
import mesh
sys.path[:0] = [str(mesh.SFA / "v3"), str(mesh.SFA / "v2")]
from sfa3 import recon3
from sfa.robot import Robot, Camera
DEV = "cuda:1"; robot, cam = Robot(), Camera(); G = mesh.Geometry(robot, device=DEV)
SIDE = np.array(G.side + ["X"])
OUT = Path("/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad")
rows = []
for E in (1, 5, 8):
    rec = recon3.build(E, with_plate=False, log=lambda *a: None)
    for j in range(3, rec.T, 20):
        Ms = G.placements(rec.gv["R"][j], rec.gv["L"][j], rec.grip_v["R"][j], rec.grip_v["L"][j])
        _, lab, _ = mesh.splat(G.world(Ms), G.gid, cam, dev=DEV)
        sd = SIDE[lab.cpu().numpy()]
        r = {"E": E, "j": j}
        for s in "RL":
            a, b = rec.masks[s][j] > 0, sd == s
            r[s] = round(float((a & b).sum() / max(1, (a | b).sum())), 3); r[s + "_sam"] = int(a.sum()); r[s + "_ren"] = int(b.sum())
        rows.append(r)
        if j in (43, 143, 183):
            im = rec.F[j].copy()
            for s, col in (("R", (255, 60, 60)), ("L", (60, 120, 255))):
                cs, _ = cv2.findContours((rec.masks[s][j] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cs, -1, (255, 255, 0), 2)
                m = sd == s; im[m] = (0.5 * im[m] + 0.5 * np.array(col)).astype(np.uint8)
            cv2.imwrite(str(OUT / f"diag_{E}_{j}.jpg"), im[..., ::-1])
for r in rows: print(r)
