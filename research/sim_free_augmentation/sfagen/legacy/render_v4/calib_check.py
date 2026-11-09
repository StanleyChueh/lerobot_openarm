"""How well does the inherited body_cam model align the URDF meshes with the SAM2 arm masks? (per episode)"""
import sys, time, json
from pathlib import Path
import numpy as np, torch, cv2
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import mesh
SFA = mesh.SFA
sys.path[:0] = [str(SFA / "v3"), str(SFA / "v2")]
from sfa3 import recon3  # noqa
from sfa.robot import Robot, Camera

def main(E=0):
    t0 = time.time()
    rec = recon3.build(E, with_plate=True, log=lambda *a: None)
    print("recon", round(time.time() - t0, 1), "s", rec.T, rec.lag, list(rec.masks), rec.F.shape)
    robot, cam = Robot(), Camera(); G = mesh.Geometry(robot)
    print("points", G.n, [ (n, s) for n, s in zip(G.names, G.side)][:40])
    out = []
    for j in range(0, rec.T, 30):
        Ms = G.placements(rec.gv["R"][j], rec.gv["L"][j], rec.grip_v["R"][j], rec.grip_v["L"][j])
        d, lab, _ = mesh.splat(G.world(Ms), G.gid, cam)
        lab = lab.cpu().numpy()
        side = np.array(G.side + ["X"])[lab]
        im = rec.F[j].copy()
        for s, col in (("R", (255, 60, 60)), ("L", (60, 120, 255))):
            sam = rec.masks[s][j] > 0; ren = side == s
            iou = (sam & ren).sum() / max(1, (sam | ren).sum())
            out.append((j, s, round(float(iou), 3)))
            cs, _ = cv2.findContours(sam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, (255, 255, 0), 1)
            im[ren] = (0.5 * im[ren] + 0.5 * np.array(col)).astype(np.uint8)
        cv2.imwrite(f"/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad/calib_{E}_{j}.jpg", im[..., ::-1])
    print(out)

main(int(sys.argv[1]))
