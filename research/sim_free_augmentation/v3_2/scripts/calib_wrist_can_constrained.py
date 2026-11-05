"""Right-wrist mount: can-silhouette fit with the rotation change from nominal penalised beyond 10 deg (the free
fit moved 61 deg / 20 cm and contradicted the independent hand-eye rotation residuals: 1.25 -> 3.11 deg).
Adoption rule for every wrist mount: held-out can IoU +0.05 or more AND held-out hand-eye residual worse by <= 0.25 deg."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
import calib_wrist_can as cc, calib_wrist as cw
from sfa32 import wristcam as wc
from sfa3.recon3 import data, rb
robot = rb.Robot(); calib = json.loads(wc.CALIB.read_text())
side = "R"
S_fit = [s for e in cc.FIT for s in cc.samples(e, side, robot)]
S_val = [s for e in cc.VAL for s in cc.samples(e, side, robot)]
X0 = wc.mount(side, "nominal")
def obj(v):
    pen = max(0.0, np.degrees(np.linalg.norm(v[:3])) - 10.0) * 0.02 + max(0.0, np.linalg.norm(v[3:]) - 0.08) * 5
    return -cc.iou_for(cc.X_of(v, X0), S_fit, side) + pen
best = None
for st in ([0]*6, [0, 0, 0, 0.05, 0, 0], [0, 0, 0, -0.05, 0, 0], [0, 0, 0, 0, 0.05, 0], [0, 0, 0, 0, -0.05, 0]):
    r = minimize(obj, np.array(st, float), method="Powell", options={"maxiter": 3000, "xtol": 1e-3, "ftol": 1e-4})
    best = r if best is None or r.fun < best.fun else best
X1 = cc.X_of(best.x, X0)
P = []
for e in cw.VAL:
    ep = data.load(e, cams=(wc.CAM_OF[side],)); P += [dict(p, state=ep.state) for p in cw.pairs_for(ep, side, robot, "cuda:1")]
lag = calib[side]["lag_frames"]
for p in P:
    g = rb.Robot.split(p["state"]); p["B"] = cw.hand_rel(robot, g, side, p["j"] + lag, p["j"] + cw.GAP + lag)
he = lambda R: float(np.median([cw.angle(R @ p["A"][:3, :3] @ R.T @ p["B"][:3, :3].T) for p in P]))
res = {"iou_fit": cc.iou_for(X1, S_fit, side), "iou_val": cc.iou_for(X1, S_val, side), "iou_val_nominal": cc.iou_for(X0, S_val, side),
       "handeye_val_deg": he(X1[:3, :3]), "handeye_val_deg_nominal": he(X0[:3, :3]),
       "rotation_change_deg": float(np.degrees(np.linalg.norm(best.x[:3]))), "translation_change_m": float(np.linalg.norm(best.x[3:])), "X": X1.tolist()}
res["adopted"] = bool(res["iou_val"] >= res["iou_val_nominal"] + 0.05 and res["handeye_val_deg"] <= res["handeye_val_deg_nominal"] + 0.25)
calib[side]["can_silhouette_fit_constrained"] = res
# apply the rule to both cameras
for s2 in "RL":
    c = calib[s2]
    cand = c.get("can_silhouette_fit_constrained") if s2 == "R" else c["can_silhouette_fit"]
    hex_ = cand["handeye_val_deg"] if s2 == "R" else calib["mount_crosscheck_handeye_residual_deg_val"][s2]["can_silhouette_fit"]
    hen = calib["mount_crosscheck_handeye_residual_deg_val"][s2]["nominal"]
    ok = (cand["iou_val"] if s2 == "R" else cand["iou_val_fitted"]) >= (cand["iou_val_nominal"]) + 0.05 and hex_ <= hen + 0.25
    X = np.array(cand["X"]) if ok else wc.mount(s2, "nominal")
    c["mount"] = {"R": X[:3, :3].tolist(), "p": X[:3, 3].tolist(), "rule": "held-out can IoU +0.05 and hand-eye residual +<=0.25 deg",
                  "R_source": ("constrained " if s2 == "R" else "") + "can-silhouette fit" if ok else "nominal", "p_source": "same", "adopted_fit": bool(ok)}
    print(s2, "adopted" if ok else "nominal kept", flush=True)
print(json.dumps({k: v for k, v in res.items() if k != "X"}))
wc.CALIB.write_text(json.dumps(calib, indent=1))
