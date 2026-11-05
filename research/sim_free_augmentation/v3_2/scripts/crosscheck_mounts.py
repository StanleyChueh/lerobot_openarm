"""Cross-check candidate wrist-mount rotations against the independent hand-eye rotation residuals (held-out eps 5-9)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
import calib_wrist as cw
from sfa32 import wristcam as wc
from sfa3.recon3 import data, rb
robot = rb.Robot(); calib = json.loads(wc.CALIB.read_text()); out = {}
for side in "RL":
    P = []
    for e in cw.VAL:
        ep = data.load(e, cams=(wc.CAM_OF[side],))
        P += [dict(p, state=ep.state) for p in cw.pairs_for(ep, side, robot, "cuda:1")]
    lag = calib[side]["lag_frames"]
    for p in P:
        g = rb.Robot.split(p["state"]); p["B"] = cw.hand_rel(robot, g, side, p["j"] + lag, p["j"] + cw.GAP + lag)
    cands = {"nominal": wc.mount(side, "nominal")[:3, :3], "hand_eye_fit": np.array(calib[side]["fitted_rotation"]),
             "can_silhouette_fit": np.array(calib[side]["can_silhouette_fit"]["X"])[:3, :3]}
    out[side] = {k: float(np.median([cw.angle(R @ p["A"][:3, :3] @ R.T @ p["B"][:3, :3].T) for p in P])) for k, R in cands.items()}
    out[side]["n_pairs"] = len(P)
    print(side, out[side], flush=True)
calib["mount_crosscheck_handeye_residual_deg_val"] = out
wc.CALIB.write_text(json.dumps(calib, indent=1))
