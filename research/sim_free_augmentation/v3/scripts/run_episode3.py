"""v3 generation for one real source episode (G+ and C, four variation families), with per-stage benchmarking.

    cd research/sim_free_augmentation/v3
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/run_episode3.py --episode 1 --device cuda:1

Every proposed candidate (every condition x variation x tried scale) is recorded with its verifier, label and target
results. No simulator / physics engine; no robot I/O; nothing is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa3 import labels as lb, planner3 as pl, recon3, render3, verify3 as v3  # noqa: E402
from sfa3.recon3 import rb  # noqa: E402
from sfa2 import metrics2 as M2, sync  # noqa: E402
from sfa2.recon import rd  # noqa: E402

G_: dict = {}
TIMES: dict = {}


class timed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        TIMES[self.name] = round(time.time() - self.t, 2)
        print(f"[ep{G_.get('E')} {self.name}] {TIMES[self.name]} s", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def variations(cfg):
    out = [("identity", {"id": "identity"})]
    for fam, vs in cfg["families"].items():
        out += [(fam, v) for v in vs]
    return out


# ----------------------------------------------------------------------------- planning
def evaluate(var, cond, scale):
    rec, robot, V, cfg = G_["rec"], G_["robot"], G_["V"], G_["cfg"]
    t0 = time.time()
    spec = pl.make_spec(rec, robot, var, scale)
    p = pl.plan(rec, robot, spec, cond)
    t1 = time.time()
    v = V(p)
    t2 = time.time()
    L = lb.validate(rec, p, G_["src_leads"], G_["src_chron"])
    tok = pl.targets_ok(p, cfg["common_target_tolerance"])
    p.update(verify=v, labels=L, targets_ok=tok, accepted=bool(p["ik"]["ok"] and tok and v["ok"] and L["ok"]),
             hash=L["hash"], runtime={"plan_s": t1 - t0, "verify_s": t2 - t1, "labels_s": time.time() - t2})
    p["waypoints"] = {s: [(int(t), k) for t, _, k in spec.waypoints[s]] for s in "RL"}
    return p


def plan_job(args):
    fam, var = args
    res = {}
    for cond in pl.CONDITIONS:
        res[cond] = {}
    own = {c: None for c in pl.CONDITIONS}
    common = None
    for s in pl.SCALES:
        for c in pl.CONDITIONS:
            if own[c] is None or common is None:
                if s not in res[c]:
                    res[c][s] = evaluate(var, c, s)
                if res[c][s]["accepted"] and own[c] is None:
                    own[c] = s
        if common is None and all(s in res[c] and res[c][s]["accepted"] for c in pl.CONDITIONS):
            common = s
        if common is not None and all(own[c] is not None for c in pl.CONDITIONS):
            break
    return var["id"], {"family": fam, "var": var, "tries": res, "own_scale": own, "common_scale": common}


def summary(p):
    return {k: p[k] for k in ("cond", "var", "scale", "accepted", "targets_ok", "ik", "hash", "runtime", "marks",
                              "init_shift_m", "pass_can_displacement_mm", "waypoints")} | {
        "verify_failed": p["verify"]["failed"], "labels_ok": p["labels"]["ok"],
        "labels": {k: p["labels"][k] for k in ("construction", "per_joint_lead")} | {"gripper_problems": p["labels"]["gripper"]["problems"]},
        "target_err_max_mm": max([t["err_mm"] for t in p["targets"] if t["kind"] == "contact"] or [0.0]),
        "target_err_max_deg": max([t["err_deg"] for t in p["targets"] if t["kind"] == "contact"] or [0.0]),
        "verify_values": {k: v.get("value") for k, v in p["verify"]["checks"].items() if isinstance(v, dict) and "value" in v}}


# ----------------------------------------------------------------------------- render + CPU metrics
def windows(rec, m):
    return {"pregrasp": (0, max(1, m["right_close"] - 2)), "giver_hold": (m["grasp"] + 10, m["capture"] - 3),
            "receiver_hold": (m["giver_released"] + 8, rec.T)}


def render_job(key):
    rec, cfg = G_["rec"], G_["cfg"]
    robot, cam = rb.Robot(), rb.Camera()
    cand = G_["render"][key]
    OUT, RAW = G_["OUT"], G_["RAW"]
    t0 = time.time()
    frames, newm, infos, j_of = render3.render(rec, robot, cam, cand)
    t_render = time.time() - t0
    t1 = time.time()
    np.save(RAW / f"{key}.npy", frames)
    rd.write_mp4(OUT / "clips" / f"{key}.mp4", frames, crf=20)
    t_encode = time.time() - t1
    t2 = time.time()
    T = rec.T
    qv, gripv, canv = render3.video_states(rec, cand)
    q_old = {s: rec.gv[s][j_of] for s in "RL"}
    g_old = {s: rec.grip_v[s][j_of] for s in "RL"}
    can_old = [rec.source_can(robot, j + rec.lag) for j in j_of]
    edited = [i for i in range(T) if infos[i]["n_layers"] > 0]
    met = {"edited_frames": len(edited)}
    proj_new = M2.projected(robot, cam, qv, gripv, canv, T)
    proj_old = M2.projected(robot, cam, q_old, g_old, can_old, T)
    if edited:
        met["prefers_own_labels"] = M2.label_consistency(frames, rec.plate, proj_new, proj_old, edited[::3])
        met["ghost"] = M2.ghost(frames, rec.plate, proj_old, proj_new, edited[::3])
    uv, ok = M2.lid_track(frames, cam, canv)
    W = windows(rec, cand["marks"])
    met["lid"] = {w: M2.lid_error(uv, ok, cam, canv, range(*W[w])) for w in W}
    lags = np.arange(-8.0, 6.01, 0.25)
    met["lag_giver"] = M2.lag_scan(robot, cam, cand["state"], rec.X["R"], uv, ok, range(*W["giver_hold"]), lags)
    # receiver window: attachment of the candidate's receiver (median of inv(T_L) can over its hold); unknown if no evidence
    gq = rb.Robot.split(cand["state"])
    rel = []
    for t in range(*W["receiver_hold"]):
        robot.fk(gq["R"][t], gq["L"][t])
        rel.append(np.linalg.inv(robot.frame(robot.tcp["L"])) @ cand["can"][t])
    XLc = rel[len(rel) // 2] if rel else rec.X["L"]
    n_rec = int(sum(ok[i] for i in range(*W["receiver_hold"])))
    if n_rec >= 10:
        ls = lag_scan_side(robot, cam, cand["state"], "L", XLc, uv, ok, range(*W["receiver_hold"]), lags)
        met["lag_receiver"] = ls | {"status": "ok" if abs(ls["lag"] - rec.lag) <= 0.5 else "mismatch"}
    else:
        met["lag_receiver"] = {"status": "unknown", "n": n_rec}
    # new initial position: is the standing can where it should be, and is the old location empty?
    if np.linalg.norm(cand["init_shift_m"]) > 0:
        met["relocation"] = relocation_metrics(frames, cam, rec, cand, W["pregrasp"], uv, ok, canv)
    met["timing_s"] = {"render": t_render, "encode": t_encode, "metrics": time.time() - t2}
    if cand["cond"] == "C" or key.startswith("identity"):
        ov = []
        for i in range(T):
            kp = rd.keypoints_3d(robot, qv["R"][i], qv["L"][i], gripv["R"][i], gripv["L"][i], canv[i])
            im = rd.draw_keypoints(frames[i], cam, kp)
            if ok[i]:
                cv2.drawMarker(im, tuple(int(round(v)) for v in uv[i]), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
            ov.append(rd.label(im, f"ep{rec.e} {key} t={i}"))
        rd.write_mp4(OUT / "clips" / f"{key}_overlay.mp4", ov, crf=26)
    return key, {"metrics": met, "j_of": j_of, "lid_uv": uv, "lid_ok": ok}


def lag_scan_side(robot, cam, state16, side, X, lid_uv, lid_ok, idx, lags):
    g = rb.Robot.split(state16)
    idx = [i for i in idx if lid_ok[i]]
    from sfa2.recon import rc
    curve = {}
    for l in lags:
        e = []
        for i in idx:
            robot.fk(rc.interp_q(g["R"], i + l), rc.interp_q(g["L"], i + l))
            p = (robot.frame(robot.tcp[side]) @ X)[:3, 3]
            e.append(np.linalg.norm(cam.project(p[None])[0][0] - lid_uv[i]))
        curve[float(l)] = float(np.median(e)) if e else np.inf
    best = min(curve, key=curve.get)
    return {"lag": best, "min_err_px": curve[best], "n": len(idx)}


def relocation_metrics(frames, cam, rec, cand, win, uv, ok, canv):
    a, b = win
    idx = list(range(a, b))
    lid = M2.lid_error(uv, ok, cam, canv, idx)
    # old location: red-body pixels inside the ORIGINAL standing-can silhouette but outside the new one
    from sfa2.recon import rc
    P = rc.cylinder_points([0, 0, 1.0], 24, 6)
    def sil(C):
        uvp, _ = cam.project((C[:3, :3] @ P.T).T + C[:3, 3])
        m = np.zeros(frames.shape[1:3], np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(uvp.astype(np.float32)).astype(np.int32), 1)
        return m > 0
    old, new = sil(rec.table), sil(cand["table_new"])
    region = old & ~cv2.dilate(new.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    res = []
    for t in idx[::3]:
        if region.sum() > 100:
            res.append(float(M2.red_mask(frames[t])[region].mean()))
    src_red = [float(M2.red_mask(rec.F[t])[region].mean()) for t in idx[::3]] if region.sum() > 100 else []
    return {"pregrasp_lid": lid, "old_location_red_frac_median": float(np.median(res)) if res else None,
            "old_location_red_frac_in_source": float(np.median(src_red)) if src_red else None,
            "old_region_px": int(region.sum())}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--config", default=str(HERE / "configs/v3.json"))
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg_bytes = Path(args.config).read_bytes()
    cfg = json.loads(cfg_bytes)
    if args.families:
        cfg["families"] = {k: v for k, v in cfg["families"].items() if k in args.families}
    E = args.episode
    G_.update(E=E, cfg=cfg)
    OUT = HERE / "outputs" / f"ep{E}{args.tag}"
    RAW = HERE / "outputs/cache" / f"ep{E}" / f"raw{args.tag}"
    for d in (OUT / "clips", OUT / "candidates", RAW):
        d.mkdir(parents=True, exist_ok=True)
    res = {"episode": E, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
           "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "repo_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, cwd=HERE).stdout.strip()),
           "simulation_used": False, "physics_used": False, "device": args.device}
    t_all = time.time()
    # 1. one-time preprocessing: lid track (CPU) and SAM2 (GPU, child process: reports peak VRAM)
    with timed("preprocess_lid_track"):
        res["lid_track"] = recon3.ensure_lid_track(E)
    with timed("preprocess_sam2"):
        code = (f"import sys, json, torch; sys.path.insert(0, {str(HERE)!r}); from sfa3 import recon3; "
                f"r = recon3.build({E}, device={args.device!r}, with_plate=False, log=lambda *a: None); "
                f"print('SAM2INFO', json.dumps({{'cached': r.info['sam2_cached'], 'sam2_s': r.info['runtime_s']['sam2'], "
                f"'peak_vram_gb': torch.cuda.max_memory_allocated({args.device!r}) / 1e9 if torch.cuda.is_initialized() else 0.0}}))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("SAM2INFO")]
        res["sam2"] = json.loads(line[0][9:]) if line else {"error": out.stderr[-2000:]}
    with timed("preprocess_recon_plate"):
        rec = recon3.build(E, with_plate=True, log=lambda *a: None)
    res["recon"] = rec.info
    robot = rb.Robot()
    with timed("verifier_init"):
        V = v3.Verifier3(rec, robot)
    src = {"state": rec.state, "action": rec.action, "can": V.src_can, "tau": np.arange(rec.T, dtype=float),
           "valid": np.arange(rec.T) <= rec.T - 1 - rec.lead}
    G_.update(rec=rec, robot=robot, V=V, OUT=OUT, RAW=RAW, src_leads=lb.joint_leads(rec.state, rec.action, np.ones(rec.T, bool)),
              src_chron=lb.gripper_chronology(rec.state, rec.action, np.ones(rec.T, bool)))
    res["source_verify"] = V(src)
    res["source_labels"] = lb.validate(rec, src, G_["src_leads"], G_["src_chron"])
    ctx = get_context("fork")
    # 2. planning (all proposed candidates)
    with timed("plans"):
        with ctx.Pool(args.workers) as P:
            plans = dict(P.map(plan_job, variations(cfg)))
    allc, render = {}, {}
    for vid, r in plans.items():
        for c, tries in r["tries"].items():
            for s, p in tries.items():
                k = f"{vid}__{c}__s{s}"
                allc[k] = summary(p)
                if (r["own_scale"][c] == s) or (r["common_scale"] == s) or (vid == "identity"):
                    render[k] = p
                    np.savez_compressed(OUT / "candidates" / f"{k}.npz", state=p["state"], action=p["action"], valid=p["valid"],
                                        can=p["can"], tau=p["tau"], qR=p["q"]["R"], qL=p["q"]["L"], table_new=p["table_new"])
    # representative rejected candidates (scale 1.0, first 4) for failure clips
    rej = [k for k, s in allc.items() if not s["accepted"] and k.endswith("s1.0")][:4]
    for k in rej:
        vid, c, s = k.split("__")
        render[k] = plans[vid]["tries"][c][1.0]
        p = render[k]
        np.savez_compressed(OUT / "candidates" / f"{k}.npz", state=p["state"], action=p["action"], valid=p["valid"],
                            can=p["can"], tau=p["tau"], qR=p["q"]["R"], qL=p["q"]["L"], table_new=p["table_new"])
    uniq, alias = {}, {}
    for k, p in render.items():
        uniq.setdefault(p["hash"], k)
        alias[k] = uniq[p["hash"]]
    keys = sorted(set(alias.values()))
    G_["render"] = render
    print(f"ep{E}: {len(allc)} proposed candidates, {len(render)} to render ({len(keys)} unique)", flush=True)
    # 3. render (CPU pool) while RAFT runs on the GPU
    from sfa import track as trk
    import torch
    rendered = {}
    gpu = {"raft_s": 0.0, "n": 0}
    with timed("render_metrics"):
        pool = ctx.Pool(args.workers)
        it = pool.imap_unordered(render_job, keys)
        torch.cuda.reset_peak_memory_stats(args.device)
        def half(F):
            return np.stack([cv2.resize(f, (320, 240), interpolation=cv2.INTER_AREA) for f in F])
        t0 = time.time()
        Fo = half(rec.F)
        fl_o = trk.video_flows(Fo, args.device)
        occ = (rec.masks["R"] > 0) | (rec.masks["L"] > 0) | (rec.masks["can"] > 0)
        occ_h = np.stack([cv2.resize(o.astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST) > 0 for o in occ])
        k9 = np.ones((9, 9), np.uint8)
        dyn_o = np.stack([cv2.dilate(o.astype(np.uint8), k9) > 0 for o in occ_h])
        res["original_temporal_halfres"] = M2.warp_error_regions(Fo, fl_o, {"robot_object": dyn_o, "background": ~dyn_o})
        gpu["raft_s"] += time.time() - t0
        for key, r in it:
            t0 = time.time()
            Fh = half(np.load(RAW / f"{key}.npy"))
            fl = trk.video_flows(Fh, args.device)
            dyn = np.stack([cv2.dilate(occ_h[r["j_of"][t]].astype(np.uint8), k9) > 0 for t in range(rec.T)])
            r["metrics"]["temporal_halfres"] = M2.warp_error_regions(Fh, fl, {"robot_object": dyn, "background": ~dyn})
            gpu["raft_s"] += time.time() - t0
            gpu["n"] += 1
            del fl, Fh
            rendered[key] = r
        pool.close()
        pool.join()
        gpu["peak_vram_gb_raft"] = torch.cuda.max_memory_allocated(args.device) / 1e9
    res["gpu"] = gpu
    # 4. corruption controls (label detectors + verifier + image lag)
    with timed("corruption_tests"):
        res["corruption_tests"] = corruptions(rec, robot, V, plans, rendered, alias)
    # 5. comparison videos: original | G+ | C for every variation (own scale), 960x240 per row
    with timed("comparison_videos"):
        for vid, r in plans.items():
            row = []
            for c in pl.CONDITIONS:
                s = r["own_scale"][c]
                k = f"{vid}__{c}__s{s if s is not None else 1.0}"
                if k not in alias:
                    row.append(None)
                    continue
                row.append((np.load(RAW / f"{alias[k]}.npy", mmap_mode="r"), f"{c} {'ACC x' + str(s) if s else 'REJECTED'}"))
            if all(x is None for x in row):
                continue
            frames = []
            for t in range(rec.T):
                panels = [rd.label(rec.F[t], f"original ep{E} t={t}")]
                for c, x in zip(pl.CONDITIONS, row):
                    panels.append(rd.label(np.array(x[0][t]), x[1]) if x is not None else rd.label(rec.plate // 3, f"{c}: rejected at every scale"))
                frames.append(cv2.resize(np.concatenate(panels, 1), (1440, 360), interpolation=cv2.INTER_AREA))
            rd.write_mp4(OUT / "clips" / f"compare_{vid}.mp4", frames, crf=24)
    res["candidates"] = allc
    res["plans"] = {vid: {"family": r["family"], "own_scale": r["own_scale"], "common_scale": r["common_scale"]} for vid, r in plans.items()}
    res["render_alias"] = alias
    res["clips"] = {k: {"metrics": r["metrics"]} for k, r in rendered.items()}
    np.savez_compressed(OUT / "render_maps.npz", **{k: r["j_of"] for k, r in rendered.items()})
    TIMES["total_episode_s"] = round(time.time() - t_all, 2)
    res["runtime_s"] = TIMES
    (OUT / "results.json").write_text(json.dumps(jsonable(res), indent=1))
    acc = {c: sum(r["own_scale"][c] is not None for vid, r in plans.items() if vid != "identity") for c in pl.CONDITIONS}
    print(json.dumps({"episode": E, "accepted_own": acc, "common": sum(r["common_scale"] is not None for v, r in plans.items() if v != "identity"),
                      "runtime_s": TIMES, "gpu": gpu}, indent=1), flush=True)


def corruptions(rec, robot, V, plans, rendered, alias) -> dict:
    s = plans["pos_right_up"]["own_scale"]["C"]
    if s is None:
        return {"skipped": "pos_right_up/C not accepted"}
    key = f"pos_right_up__C__s{s}"
    c = plans["pos_right_up"]["tries"]["C"][s]
    r = rendered[alias[key]]
    uv, ok = r["lid_uv"], r["lid_ok"]
    W = windows(rec, c["marks"])
    lags = np.arange(-10.0, 7.01, 0.25)
    src_l, src_c = G_["src_leads"], G_["src_chron"]
    import copy

    def detectors(d, image_check=True):
        L = lb.validate(rec, d, src_l, src_c)
        v = V(d)
        out = {"labels_construction": L["construction"]["ok"], "labels_per_joint": L["per_joint_lead"]["ok"],
               "labels_gripper": L["gripper"]["ok"], "verifier": v["ok"], "verifier_failed": v["failed"],
               "global_lead": sync.action_lead(d["state"], d["action"])["lead"]}
        if image_check:
            ls = M2.lag_scan(robot, cam, d["state"], rec.X["R"], uv, ok, range(*W["giver_hold"]), lags)
            out["image_lag"] = ls["lag"]
            out["image_lag_ok"] = abs(ls["lag"] - rec.lag) <= 0.5
        out["detected"] = not (L["ok"] and v["ok"] and out.get("image_lag_ok", True) and out["global_lead"] == rec.lead)
        return out

    cam = rb.Camera()
    tests = {"intended": detectors(c)}
    T = rec.T
    def edit(fn):
        d = copy.deepcopy({k: c[k] for k in ("state", "action", "valid", "can", "tau", "marks")})
        fn(d)
        d["valid"] = np.isfinite(d["action"]).all(1) & c["valid"]
        return d
    def delay_col(d, name, k):
        i = rb.NAMES.index(name)
        x = d["action"][:, i].copy()
        d["action"][k:, i] = x[:-k]
        d["action"][:k, i] = x[0]
    tests["source_actions_substituted"] = detectors(edit(lambda d: d.__setitem__("action", np.where(np.isfinite(c["action"]), rec.action, np.nan))))
    for name in [f"{s}J{i}.pos" for s in "RL" for i in range(1, 8)]:
        tests[f"{name}_delayed6"] = detectors(edit(lambda d, n=name: delay_col(d, n, 6)), image_check=False)
    tests["L_gripper_command_only_delayed3"] = detectors(edit(lambda d: delay_col(d, "LJ8.pos", 3)), image_check=False)
    def premature(d):
        i = rb.NAMES.index("RJ8.pos")
        ro = c["marks"]["right_open"]
        d["action"][ro - 3:ro, i] = d["action"][ro, i]
    tests["R_gripper_command_only_opens3early"] = detectors(edit(premature), image_check=False)
    tests["actions_delayed6"] = detectors(sync.delay_actions({k: c[k] for k in ("state", "action", "valid", "can", "tau")}, 6))
    for k in (3, 6):
        tests[f"state_delayed{k}_vs_image"] = detectors(sync.delay_state({k2: c[k2] for k2 in ("state", "action", "valid", "can", "tau")}, k))
    st_L = next(t for t in range(T) if rb.Robot.split(c["state"])["gL"][t] < pl.CLOSED)
    for deg in (20, 90):
        tests[f"object_rotated{deg}_after_receiver_grasp"] = detectors(sync.rotate_object({k: c[k] for k in ("state", "action", "valid", "can", "tau")}, deg, st_L), image_check=False)
    ro_gap = c["marks"]["right_open"] - c["marks"]["left_close"]
    tests["receiver_closes_after_giver_opens"] = detectors(sync.shift_gripper({k: c[k] for k in ("state", "action", "valid", "can", "tau")}, "L", ro_gap + 6), image_check=False)
    base = {k: c[k] for k in ("state", "action", "valid", "can", "tau")}
    for k in (-15, 15):
        tests[f"giver_gripper_shift{k}"] = detectors(sync.shift_gripper(base, "R", k), image_check=False)
    # export-integrity: a single changed byte must change the hash
    d = copy.deepcopy(base)
    d["action"][100, 0] += 1e-6
    tests["tampered_hash_differs"] = {"detected": lb.candidate_hash(d) != c["hash"]}
    intended_ok = not tests["intended"]["detected"]
    neg = {k: v["detected"] for k, v in tests.items() if k != "intended"}
    return {"base": key, "tests": tests, "intended_passes": intended_ok, "detected": sum(neg.values()), "n": len(neg),
            "missed": [k for k, v in neg.items() if not v]}


if __name__ == "__main__":
    main()
