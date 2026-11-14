"""Setup check of a robot config: draw the URDF link hulls (FK of the dataset state) and both TCPs projected with a
given camera over decoded body-camera frames. Used once with the simulator's true camera to validate the joint
mapping before sfagen runs (sfagen itself never gets the truth). Usage: CONFIG CAMERA_JSON|fitted|initial OUT.jpg EP FRAME[,FRAME..]"""
import json, sys
import cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
from sfagen.camera import Camera

ctx = Context(sys.argv[1])                       # config work_dir: its calibration.json gives the fitted camera
if sys.argv[2] in ("fitted", "initial"):
    cam = ctx.cam if sys.argv[2] == "fitted" else ctx.cam0
else:
    cj = json.load(open(sys.argv[2])); cam = Camera({"intrinsics": cj["intrinsics"], "extrinsics": cj})
ep = int(sys.argv[4]); js = [int(x) for x in sys.argv[5].split(",")]
S, _ = ctx.src.arrays(ep); F = ctx.src.decode(ctx.src.body, ep); sp = ctx.robot.split(S); tiles = []
for j in js:
    ctx.robot.fk({s: sp[s][j] for s in ctx.robot.arms}, {s: sp["g" + s][j] for s in ctx.robot.arms if "g" + s in sp})
    im = F[j].copy()
    for name, arm, P in ctx.robot.link_points():
        uv, z = cam.project(P)
        if (z > 0).all() and len(uv) >= 3:
            col = {"L": (255, 0, 0), "R": (0, 255, 0)}.get(arm, (255, 255, 0))
            cv2.polylines(im, [cv2.convexHull(uv.astype(np.int32))], True, col, 1)
    for s in ctx.robot.arms:
        uv, _ = cam.project(ctx.robot.frame(ctx.robot.tcp[s])[:3, 3][None]); cv2.circle(im, tuple(int(v) for v in uv[0]), 6, (255, 0, 255), 2)
    cv2.putText(im, f"ep{ep} t{j}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2); tiles.append(im)
cv2.imwrite(sys.argv[3], cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
print(sys.argv[3])
