"""Plan + replay-check every configured variation for one source episode; writes candidates/ep{E}/{key}.npz + checks.json."""
from __future__ import annotations
import json,hashlib,time
from pathlib import Path
import numpy as np
from . import plan


def variation_list(cfg, rec):
    V = cfg.get("variations", {}); out = []
    op = V.get("object_position")
    if op:
        for name in op["objects"]:
            for s in op["shifts_cm"]:
                out.append((f"{name}_x{s[0]:+g}_y{s[1]:+g}", "object_position", name, np.array([s[0], s[1], 0.0]) / 100))
    ow = V.get("object_workspace")
    if ow:
        from .workspace_plan import support_targets
        names = [o["name"] for o in rec.objects]
        for name in ow["objects"]:
            D, info, _ = support_targets(rec, names.index(name), cfg.get("checks", {}), ow.get("n_targets", 48), ow.get("grid_step_m", 0.01), ow.get("min_shift_m", 0.02),near_fraction=ow.get("near_fraction",.5))
            rec.info.setdefault("workspace", {})[name] = info
            c0 = info.get("original_centre_xy", [0, 0])
            for d in D:
                out.append((f"{name}_at_x{100 * (c0[0] + d[0]):+.0f}_y{100 * (c0[1] + d[1]):+.0f}", "object_workspace", name, np.asarray(d, float)))
    il = V.get("interaction_location")
    if il and len(rec.arms) > 1:
        for s in il["shifts_cm"]:
            s3 = list(s) + [0.0] * (3 - len(s))
            out.append((f"interact_x{s3[0]:+g}_y{s3[1]:+g}_z{s3[2]:+g}", "interaction_location", None, np.array(s3) / 100))
    keys=[row[0] for row in out]
    if len(keys)!=len(set(keys)):raise ValueError('Variation IDs collide; use distinct shifts and at least a 1 cm workspace grid')
    return out


def run_episode(ctx, rec, checker, log=print):
    out = ctx.work / f"candidates/ep{rec.e}"; out.mkdir(parents=True, exist_ok=True); checks = {}
    code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    manifest={'complete':False,'episode':rec.e,'started_unix':time.time(),'source':ctx.cfg['source'],'config':ctx.cfg,
              'source_state_sha256':hashlib.sha256(rec.state.astype(np.float32).tobytes()).hexdigest(),
              'source_action_sha256':hashlib.sha256(rec.action.astype(np.float32).tobytes()).hexdigest(),
              'replay_envelope':checker.env,'code_sha256':code,'camera_calibration':ctx.calib,'lag':rec.lag,
              'reconstruction_sha256':hashlib.sha256((ctx.cache_dir/f'recon_ep{rec.e}.npz').read_bytes()).hexdigest()}
    mp=out/'generation_manifest.json';mp.write_text(json.dumps(manifest,indent=1))
    il = ctx.cfg.get("variations", {}).get("interaction_location", {})
    names = [o["name"] for o in rec.objects]
    for key, fam, obj, d in variation_list(ctx.cfg, rec):
        if fam == "object_workspace":
            from .workspace_plan import plan_with_time_scale
            ow = ctx.cfg["variations"]["object_workspace"]
            c = plan_with_time_scale(rec, names.index(obj), d, checker, tuple(ow.get("time_scales", (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0))), ow.get("approach_margin_m", 0.10))
        elif fam == "object_position":
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
                            **{f"q_{s}": c["q"][s] for s in rec.arms}, **({"tau": c["tau"], "ref_state": c["ref_state"].astype(np.float32), "ref_action": c["ref_action"].astype(np.float32)} if "tau" in c else {}),**({"native_offset":c["native_offset"],**{f"native_q_{s}":c["native_q"][s] for s in rec.arms}} if "native_q" in c else {}))
        log(rec.e, key, "PASS" if ck["pass"] else "FAIL", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ck.items()
            if k in ("speed_ratio_to_real_max", "accel_ratio_to_real_max", "min_mesh_distance_mm", "min_hand_distance_mm", "ik_pos_err_mm_max")},
            ck.get("placement", {}).get("ok", "-"))
    if "workspace" in rec.info:
        (out / "workspace.json").write_text(json.dumps(rec.info["workspace"], indent=1, default=float))
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float))
    manifest.update(complete=True,finished_unix=time.time(),candidates=len(checks),passed=sum(v['pass'] for v in checks.values()),
                    candidate_sha256={key:hashlib.sha256((out/f'{key}.npz').read_bytes()).hexdigest() for key in checks if (out/f'{key}.npz').exists()})
    mp.write_text(json.dumps(manifest,indent=1))
    return checks


def load_candidate(ctx, rec, key):
    root=ctx.work/f'candidates/ep{rec.e}';mp=root/'generation_manifest.json'
    if mp.exists():
        m=json.loads(mp.read_text())
        if not m['complete']:raise ValueError(f'Generation is incomplete: {root}')
        if hashlib.sha256((root/f'{key}.npz').read_bytes()).hexdigest()!=m['candidate_sha256'][key]:raise ValueError(f'Candidate hash mismatch: {key}')
    z = np.load(root/f'{key}.npz')
    meta=json.loads((ctx.work/f"candidates/ep{rec.e}/checks.json").read_text())[key]
    return {**{k:meta[k] for k in ("family","object","shift_m","marks","lead","t_rejoin")},"moved_arms":([rec.objects[[o["name"] for o in rec.objects].index(meta["object"])]["arm"]] if meta["object"] else list(rec.arms)),"state": z["state"].astype(np.float64), "action": z["action"].astype(np.float64), "q": {s: z[f"q_{s}"] for s in rec.arms},
            "objects": list(z["objects"]) if z["objects"].size else [], "edited": z["edited"], "w": z["w"],"ik_err":z["ik_err"],
            **({"tau": z["tau"], "ref_state": z["ref_state"].astype(np.float64), "ref_action": z["ref_action"].astype(np.float64)} if "tau" in z.files else {})}
