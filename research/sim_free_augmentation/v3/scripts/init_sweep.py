"""Plan-only feasibility sweep of new initial can positions (grid in the table plane) for all source episodes,
both conditions, scale 1.0 (no repair): records acceptance and every failed check. No rendering.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/init_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa3 import labels as lb, planner3 as pl, recon3, verify3 as v3  # noqa: E402
from sfa3.recon3 import rb  # noqa: E402

G_ = {}


def job(a):
    dx, dy, cond = a
    rec, robot, V, cfg = G_["rec"], G_["robot"], G_["V"], G_["cfg"]
    var = {"id": f"init_{dx:+.3f}_{dy:+.3f}", "init_shift_m": [dx, dy]}
    p = pl.plan(rec, robot, pl.make_spec(rec, robot, var, 1.0), cond)
    v = V(p)
    L = lb.validate(rec, p, G_["sl"], G_["sc"])
    tok = pl.targets_ok(p, cfg["common_target_tolerance"])
    cam = rb.Camera()
    uv, z = cam.project(p["table_new"][:3, 3][None])
    fails = v["failed"] + ([] if p["ik"]["ok"] else ["ik"]) + ([] if tok else ["targets"]) + ([] if L["ok"] else ["labels"])
    return {"dx": dx, "dy": dy, "cond": cond, "accepted": not fails, "failed": fails,
            "init_lid_uv": uv[0].tolist(), "ik_err_mm": p["ik"]["max_pos_err_mm"],
            "speed": v["checks"]["joint_speed"]["value"], "penetration_mm": v["checks"]["can_penetration"]["value"]}


def main():
    cfg = json.loads((HERE / "configs/v3.json").read_text())
    grid = cfg["init_sweep_grid_m"]
    out = {"grid": grid, "episodes": {}}
    t0 = time.time()
    ctx = get_context("fork")
    for e in cfg["source_episodes"]:
        t1 = time.time()
        rec = recon3.build(e, with_plate=False, log=lambda *a: None)
        robot = rb.Robot()
        G_.update(rec=rec, robot=robot, V=v3.Verifier3(rec, robot), cfg=cfg,
                  sl=lb.joint_leads(rec.state, rec.action, np.ones(rec.T, bool)),
                  sc=lb.gripper_chronology(rec.state, rec.action, np.ones(rec.T, bool)))
        jobs = [(dx, dy, c) for dx in grid["x"] for dy in grid["y"] for c in pl.CONDITIONS]
        with ctx.Pool(22) as P:
            rows = P.map(job, jobs)
        cam = rb.Camera()
        uv0, _ = cam.project(rec.table[:3, 3][None])
        out["episodes"][e] = {"rows": rows, "source_init_lid_uv": uv0[0].tolist(), "source_table_xy": rec.table[:2, 3].tolist(),
                              "s": time.time() - t1}
        acc = {c: sum(r["accepted"] for r in rows if r["cond"] == c) for c in pl.CONDITIONS}
        print(f"ep{e}: accepted {acc} of {len(rows) // 2} offsets ({time.time() - t1:.0f} s)", flush=True)
    out["wall_s"] = time.time() - t0
    (HERE / "outputs/init_sweep.json").write_text(json.dumps(out, indent=0))


if __name__ == "__main__":
    main()
