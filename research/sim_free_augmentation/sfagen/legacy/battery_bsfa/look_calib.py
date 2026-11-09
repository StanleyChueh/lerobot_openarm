import sys, numpy as np, cv2
sys.path[:0] = ["bsfa", "/home/stanley/Stanley_ws/sfa_hf_export/render_v4"]
import bdata, mesh
from sfa.robot import Robot, Camera
import torch
OUT = "/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad/"
robot, cam = Robot(), Camera(); G = mesh.Geometry(robot, device="cuda:0")
SIDE = np.array(G.side + ["X"])
for e in range(10):
    d = bdata.table(); d = d[d.episode_index == e]; S = np.stack(d["observation.state"].to_numpy()).astype(float); A = np.stack(d["action"].to_numpy())
    ev = bdata.grip_events(A); tg = ev["R"]["closes"][0]
    sp = Robot.split(S); T = robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "R")[tg]
    TL = robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "L")[ev["L"]["closes"][0]]
    print(e, "R grasp t", tg, "tcp", T[:3, 3].round(3), "x-axis", T[:3, 0].round(2), "y-axis", T[:3, 1].round(2), "z-axis", T[:3, 2].round(2), "| L grasp tcp", TL[:3, 3].round(3))
ep = bdata.load(0, cams=(bdata.CAMS[0],)); F = ep.frames[bdata.CAMS[0]]
ev = bdata.grip_events(ep.action); tiles = []
for j in (0, ev["L"]["closes"][0], ev["R"]["closes"][0], ev["R"]["opens"][0], 560):
    sp = Robot.split(ep.state[j:j + 1])
    Ms = G.placements(sp["R"][0], sp["L"][0], sp["gR"][0], sp["gL"][0]); _, lab, _ = mesh.splat(G.world(Ms), G.gid, cam)
    sd = SIDE[lab.cpu().numpy()]; im = F[j].copy()
    for s, col in (("R", (255, 60, 60)), ("L", (60, 120, 255))):
        m = sd == s; cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(im, cs, -1, col, 2)
    tiles.append(cv2.resize(im, (480, 360)))
cv2.imwrite(OUT + "bat_calib.jpg", cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
