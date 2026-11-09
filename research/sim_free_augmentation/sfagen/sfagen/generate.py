"""Plan + replay-check every configured variation for one source episode; writes candidates/ep{E}/{key}.npz + checks.json."""
from __future__ import annotations
import json
import numpy as np
from . import plan


def variation_list(cfg, rec):
    V = cfg.get("variations", {}); out = []
    op = V.get("object_position")
    if op:
        for name in op["objects"]:
            for s in op["shifts_cm"]:
                out.append((f"{name}_x{s[0]:+g}_y{s[1]:+g}", "object_position", name, np.array([s[0], s[1], 0.0]) / 100))
    il = V.get("interaction_location")
    if il and len(rec.arms) > 1:
        for s in il["shifts_cm"]:
            s3 = list(s) + [0.0] * (3 - len(s))
            out.append((f"interact_x{s3[0]:+g}_y{s3[1]:+g}_z{s3[2]:+g}", "interaction_location", None, np.array(s3) / 100))
    return out


def run_episode(ctx, rec, checker, log=print):
    out = ctx.work / f"candidates/ep{rec.e}"; out.mkdir(parents=True, exist_ok=True); checks = {}
    il = ctx.cfg.get("variations", {}).get("interaction_location", {})
    names = [o["name"] for o in rec.objects]
    for key, fam, obj, d in variation_list(ctx.cfg, rec):
        if fam == "object_position":
            c = plan.f1_plan(rec, names.index(obj), d)
        else:
            c = plan.f2_plan(rec, d, il.get("d_interact_m", 0.20), il.get("ramp_frames", 45))
        if c is None:
            checks[key] = {"pass": False, "family": fam, "error": "no valid schedule (see plan.py rules)"}; log(rec.e, key, "NO SCHEDULE"); continue
        ck = checker.check(rec, c)
        ck.update({"family": fam, "object": obj, "shift_m": d.tolist(), "marks": c["marks"], "lead": c["lead"], "t_rejoin": c["t_rejoin"]})
        checks[key] = ck
        np.savez_compressed(out / f"{key}.npz", state=c["state"].astype(np.float32), action=c["action"].astype(np.float32),
                            w=c["w"], edited=c["edited"], ik_err=c["ik_err"], objects=np.stack(c["objects"]) if c["objects"] else np.zeros((0,)),
                            **{f"q_{s}": c["q"][s] for s in rec.arms})
        log(rec.e, key, "PASS" if ck["pass"] else "FAIL", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ck.items()
            if k in ("speed_ratio_to_real_max", "accel_ratio_to_real_max", "min_mesh_distance_mm", "min_hand_distance_mm", "ik_pos_err_mm_max")},
            ck.get("placement", {}).get("ok", "-"))
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float))
    return checks


def load_candidate(ctx, rec, key):
    z = np.load(ctx.work / f"candidates/ep{rec.e}/{key}.npz")
    return {"state": z["state"].astype(np.float64), "action": z["action"].astype(np.float64), "q": {s: z[f"q_{s}"] for s in rec.arms},
            "objects": list(z["objects"]) if z["objects"].size else [], "edited": z["edited"], "w": z["w"]}
