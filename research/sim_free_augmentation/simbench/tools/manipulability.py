"""Evaluation only: 6-D manipulability sqrt(det(J J^T)) and smallest singular value of the grasping arm's TCP Jacobian
(LOCAL_WORLD_ALIGNED, arm joints only) along a source episode. Usage: manipulability.py CONFIG EPISODE [EVERY]"""
import sys
from pathlib import Path
import numpy as np, pinocchio as pin
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
ctx = Context(sys.argv[1]); e = int(sys.argv[2]); k = int(sys.argv[3]) if len(sys.argv) > 3 else 6; r = ctx.robot
arm = ctx.cfg["objects"][ctx.cfg["variations"]["object_targets"]["object"]]["grasped_by"]
S = ctx.src.arrays(e)[0]
sp = r.split(S); cols = [r.m.joints[r.m.getJointId(j)].idx_v for j in ctx.cfg["robot"]["arms"][arm]["joints"]]
for t in range(0, len(S), k):
    x = r.q_full({s: sp[s][t] for s in r.arms}); J = pin.computeFrameJacobian(r.m, r.d, x, r.tcp[arm], pin.LOCAL_WORLD_ALIGNED)[:, cols]
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"t={t / ctx.src.fps:4.1f}s  q={np.round(sp[arm][t], 2)}  manip={np.prod(sv):.2e}  sigma_min={sv[-1]:.3f}")
