"""One SD10-G1 geometry attempt (selection -> SE(3) -> G+ plan -> [3,T-6) export -> G1-A gates), condition-agnostic.

The condition label only selects which locked assignment is read; no code path depends on it.
"""
from __future__ import annotations

import time

import numpy as np

import g1gates as GG
import g1lib as G

_CACHE = {}


def robot_geom():
    if "robot" not in _CACHE:
        from sfa import robot as rb
        r = rb.Robot()
        _CACHE["robot"], _CACHE["geom"] = r, GG.Geometry(r)
    return _CACHE["robot"], _CACHE["geom"]


def ranking(target, cond):
    """Recompute the frozen rule from locked initial-lid metadata; must equal the locked ranking."""
    u = np.asarray(target["requested_lid_uv_native_px"])
    ids = G.CFG["conditions"][cond]
    sc = {e: float(((np.asarray(G.CFG["source_metadata"][str(e)]["initial_lid_uv"]) - u) ** 2).sum()) for e in ids}
    return sorted(ids, key=lambda e: (sc[e], e)), sc


def run(target, cond, source=None, rec=None, kind="primary"):
    t0 = time.time()
    a = target["assignment"][cond]
    rank, sc = ranking(target, cond)
    sel_ok = rank == a["ranked_source_ids"] and rank[0] == a["source_episode"]
    src = a["source_episode"] if source is None else source
    assert src in G.UNION and src not in G.CFG["data"]["excluded_v4a"]
    rec_ = rec if rec is not None else G.load_light(src)
    robot, geom = robot_geom()
    out = {"kind": kind, "target_id": target["target_id"], "condition": cond, "source_episode": src,
           "locked_source_episode": a["source_episode"], "selection_recomputed_equal": sel_ok,
           "selection_score_sq_px": a["score_squared_pixels"], "selection_distance_px": a["distance_pixels"],
           "near_far": target["near_far"], "region": target["region"], "spatial_block": target["spatial_block"],
           "seed": G.CFG["compute"]["seed"], "source_T": rec_.T, "source_lead": rec_.lead,
           "asset_sha256": G.asset_hash(rec_)}
    tt = {}
    try:
        x = time.time()
        init, tinfo = G.init_transform(rec_, target["candidate_lid_xyz_robot_base_m"])
        spec = G.make_spec(rec_, robot, init)
        tt["spec_s"] = time.time() - x
        x = time.time()
        p = G.plan(rec_, robot, spec)
        tt["plan_s"] = time.time() - x
        ex = G.export(rec_, p)
        x = time.time()
        res = GG.evaluate(rec_, robot, geom, spec, p, ex, target)
        tt["gates_s"] = time.time() - x
        out.update(transform=tinfo, Y_A=res["Y_A"], failed_gates=res["failed_gates"], gates=res["gates"],
                   status="complete_plan" if all(np.isfinite(ex[k]).all() for k in ("state", "action")) else "nonfinite_plan",
                   export_sha256={k: G.arr_sha(ex[k]) for k in ("state", "action", "can", "native_rows")},
                   ik_planner=p["ik"])
        arrays = {"state": ex["state"], "action": ex["action"], "can": ex["can"], "native_rows": ex["native_rows"],
                  "timestamp_s": ex["timestamp_s"], "qR_full": p["q"]["R"], "qL_full": p["q"]["L"], "can_full": p["can"],
                  "state_full": p["state"], "action_full": p["action"], "table_new": p["table_new"], "init": init}
    except Exception as exc:  # noqa: BLE001  - an exception is a failed attempt (Y_A=0), never a missing row
        import traceback
        out.update(Y_A=False, failed_gates=["exception"], status="exception", error=traceback.format_exc()[-3000:])
        arrays = None
    tt["total_s"] = time.time() - t0
    out["timing_s"] = tt
    return out, arrays
