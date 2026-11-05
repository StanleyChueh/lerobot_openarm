"""v2 experiment on one real Pringles episode: matched G/P/C candidates for every frozen variation, independent
verification, rendering, image metrics, synchronisation and corruption tests, comparison videos.

    cd research/sim_free_augmentation/v2
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/run_episode.py \
        --config configs/frozen_v2.json --episode 0

No simulator / physics engine anywhere; no robot I/O; nothing is uploaded.
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
from sfa2 import metrics2 as M2, planner as pl, recon, render2, se3, sync, verify as vf  # noqa: E402
from sfa2.recon import rb, rd  # noqa: E402

TIMES: dict = {}
G_: dict = {}          # fork-shared state (recon, verifier, candidates)


class timed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        TIMES[self.name] = round(time.time() - self.t, 2)
        print(f"[{self.name}] {TIMES[self.name]} s", flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def source_tree_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(list((HERE / "sfa2").glob("*.py")) + list((HERE / "scripts").glob("*.py")) +
                    list((HERE.parent / "sfa").glob("*.py"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- workers
def plan_job(args):
    var, cond = args
    rec, robot, V, sigma = G_["rec"], G_["robot"], G_["V"], G_["sigma"]
    t0 = time.time()
    p = pl.plan_with_repair(rec, robot, var, cond, sigma, lambda p, s: V(p))
    spec = p.pop("spec")
    p["spec"] = {"marks": spec.marks, "D": spec.D, "Gamma": spec.Gamma, "a": spec.a,
                 "targets": {k: (int(v[0]), np.asarray(v[1]).tolist()) for k, v in spec.targets.items()}}
    p["runtime_s"] = time.time() - t0
    return f"{var['id']}__{cond}", p


def cand_hash(p) -> str:
    h = hashlib.sha256()
    for k in ("state", "action", "can"):
        h.update(np.round(np.nan_to_num(np.asarray(p[k]), nan=-999), 9).tobytes())
    return h.hexdigest()[:16]


def windows(rec, m):
    T = rec.T
    return {"giver_hold": (m["grasp"] + 10, m["capture"] - 3), "receiver_hold": (m["giver_released"] + 8, T),
            "handover": (m["right_close"], min(T, m["giver_released"] + 40)), "all": (0, T)}


def render_job(key):
    rec, robot, cam = G_["rec"], rb.Robot(), rb.Camera()
    cand = G_["cands"][key]
    OUT, RAW = G_["OUT"], G_["RAW"]
    t0 = time.time()
    frames, newm, infos, j_of = render2.render(rec, robot, cam, cand)
    t_render = time.time() - t0
    np.save(RAW / f"{key}.npy", frames)
    np.save(RAW / f"{key}_newmask.npy", np.packbits(newm, axis=-1))
    rd.write_mp4(OUT / "clips" / f"{key}.mp4", frames, crf=18)
    T = rec.T
    qv, gripv, canv = render2.video_states(rec, cand)
    q_old = {s: rec.gv[s][j_of] for s in "RL"}
    grip_old = {s: rec.grip_v[s][j_of] for s in "RL"}
    can_old = [rec.source_can(robot, j + rec.lag) for j in j_of]
    proj_new = M2.projected(robot, cam, qv, gripv, canv, T)
    proj_old = M2.projected(robot, cam, q_old, grip_old, can_old, T)
    edited = [i for i in range(T) if infos[i]["n_layers"] > 0]
    m = cand["marks"]
    W = windows(rec, m)
    met = {"render_s": t_render, "edited_frames": len(edited)}
    if edited:
        met["pose_discrimination_vs_source_labels"] = M2.label_consistency(frames, rec.plate, proj_new, proj_old, edited[::3])
        met["ghost"] = M2.ghost(frames, rec.plate, proj_old, proj_new, edited[::3])
    uv, ok = M2.lid_track(frames, cam, canv)
    met["lid"] = {w: M2.lid_error(uv, ok, cam, canv, range(*W[w])) for w in ("giver_hold", "receiver_hold")}
    met["lid"]["lost_frames"] = int((~np.isfinite(uv).all(1)).sum())
    met["can_axis_2d"] = {w: M2.can_axis_error(frames, cam, canv, range(W[w][0], W[w][1], 2)) for w in ("giver_hold", "receiver_hold")}
    lags = np.arange(-10.0, 7.01, 0.25)
    met["lag_scan"] = M2.lag_scan(robot, cam, cand["state"], rec.X["R"], uv, ok, range(*W["giver_hold"]), lags)
    met["action_lead"] = sync.action_lead(cand["state"], cand["action"])
    # overlay: commanded keypoints on the generated frame + independently detected lid (HSV) as a cross
    ov = []
    for i in range(T):
        kp = rd.keypoints_3d(robot, qv["R"][i], qv["L"][i], gripv["R"][i], gripv["L"][i], canv[i])
        im = rd.draw_keypoints(frames[i], cam, kp)
        if ok[i]:
            u = tuple(int(round(v)) for v in uv[i])
            cv2.drawMarker(im, u, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        ov.append(rd.label(im, f"{key} t={i}  target keypoints + HSV lid (green)"))
    rd.write_mp4(OUT / "clips" / f"{key}_overlay.mp4", ov, crf=23)
    hom = [{k: v for k, v in inf["homographies"].items()} for inf in infos]
    return key, {"metrics": met, "j_of": j_of, "homographies": hom, "lid_uv": uv, "lid_ok": ok, "runtime_s": time.time() - t0}


# ----------------------------------------------------------------------------- keypoint query points
def query_points(rec, robot, cam, ref):
    labels = rd.link_label_map(robot, cam, rec.gv["R"][ref], rec.gv["L"][ref], rec.grip_v["R"][ref],
                               rec.grip_v["L"][ref], {k: rec.masks[k][ref] for k in rec.masks})
    rng = np.random.default_rng(0)
    names = [n for n, s, P in robot.link_points("hull")]
    sides = [s for n, s, P in robot.link_points("hull")]
    pts, lay = [], []
    for side in "RL":
        for i, (n, s) in enumerate(zip(names, sides)):
            if s == side and n.split("_", 1)[1] in ("hand", "link7", "left_finger", "right_finger"):
                yy, xx = np.nonzero(cv2.erode((labels[side] == i).astype(np.uint8), np.ones((7, 7), np.uint8)))
                for j in (rng.choice(len(xx), size=min(8, len(xx)), replace=False) if len(xx) else []):
                    pts.append([xx[j], yy[j]])
                    lay.append(f"{side}_{n}")
    yy, xx = np.nonzero(cv2.erode(rec.masks["can"][ref], np.ones((9, 9), np.uint8)))
    for j in rng.choice(len(xx), size=min(16, len(xx)), replace=False):
        pts.append([xx[j], yy[j]])
        lay.append("can")
    return np.array(pts, float), lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cfg_bytes = Path(args.config).read_bytes()
    cfg = json.loads(cfg_bytes)
    E = args.episode
    if E != cfg["dev_episode"]:
        assert cfg["sigma_G_frames"] is not None, "config must be frozen before test episodes"
    OUT = HERE / "outputs" / f"ep{E}"
    CACHE = HERE / "outputs" / "cache" / f"ep{E}"
    RAW = CACHE / "raw"
    for d in (OUT / "clips", OUT / "candidates", RAW):
        d.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    res = {"episode": E, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(), "source_tree_sha256": source_tree_hash(),
           "repo_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip(),
           "repo_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, cwd=HERE).stdout.strip()),
           "dataset": cfg["dataset"], "simulation_used": False, "physics_used": False}
    # 1. SAM2 (GPU) in a child process so this process forks workers before touching CUDA
    with timed("sam2_or_cache"):
        code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); from pathlib import Path; from sfa2 import recon; "
                f"recon.build({E}, Path({str(CACHE)!r}), device={args.device!r}, keyframe_step={cfg['sam2_keyframe_step']}, with_plate=False)")
        subprocess.run([py, "-c", code], check=True)
    with timed("recon"):
        rec = recon.build(E, CACHE, keyframe_step=cfg["sam2_keyframe_step"])
        assert rec.info["sam2_cached"]
    res["recon"] = rec.info
    robot, cam = rb.Robot(), rb.Camera()
    with timed("verifier_init"):
        V = vf.Verifier(rec, robot)
    src = {"state": rec.state, "action": rec.action, "can": V.src_can, "tau": np.arange(rec.T, dtype=float),
           "valid": np.ones(rec.T, bool)}
    res["source_verify"] = V(src)
    res["verifier_tolerances"] = vf.TOL
    G_.update(rec=rec, robot=robot, V=V, sigma=float(cfg["sigma_G_frames"] or 20.0), OUT=OUT, RAW=RAW)

    # 2. plans
    jobs = [(v, c) for v in cfg["variations"] for c in pl.CONDITIONS]
    ctx = get_context("fork")
    with timed("plans"):
        with ctx.Pool(min(args.workers, len(jobs))) as P:
            cands = dict(P.map(plan_job, jobs))
    for k, p in cands.items():
        p["hash"] = cand_hash(p)
        np.savez_compressed(OUT / "candidates" / f"{k}.npz", state=p["state"], action=p["action"], valid=p["valid"],
                            can=p["can"], tau=p["tau"], qR=p["q"]["R"], qL=p["q"]["L"], names=np.array(rb.NAMES))
    G_["cands"] = cands
    uniq = {}
    for k, p in cands.items():
        uniq.setdefault(p["hash"], k)
    render_keys = sorted(set(uniq.values()))
    alias = {k: uniq[p["hash"]] for k, p in cands.items()}
    print(f"{len(cands)} candidates, {len(render_keys)} unique renders", flush=True)

    # 3. render (CPU pool) overlapped with flows / keypoints (GPU, this process)
    from sfa import track as trk
    rendered = {}
    with timed("render_and_gpu_metrics"):
        pool = ctx.Pool(min(args.workers, len(render_keys)))
        it = pool.imap_unordered(render_job, render_keys)
        with timed("flows_original"):
            flows_o = trk.video_flows(rec.F, args.device)
        ref = int(rec.ev["right_close"] + (2 * (rec.ev["left_close"] - rec.ev["right_close"])) // 3)
        q_pts, q_lay = query_points(rec, robot, cam, ref)
        uv_o, vis_o = trk.track_points(rec.F, ref, q_pts, args.device, flows_o)
        occ = (rec.masks["R"] > 0) | (rec.masks["L"] > 0) | (rec.masks["can"] > 0)
        k15 = np.ones((15, 15), np.uint8)
        dyn_o = np.stack([cv2.dilate(o.astype(np.uint8), k15) > 0 for o in occ])
        res["original_temporal"] = M2.warp_error_regions(rec.F, flows_o, {"robot_object": dyn_o, "background": ~dyn_o})
        for key, r in it:
            frames = np.load(RAW / f"{key}.npy")
            newm = np.unpackbits(np.load(RAW / f"{key}_newmask.npy"), axis=-1)[..., :frames.shape[2]].astype(bool)
            fl = trk.video_flows(frames, args.device)
            j_of = r["j_of"]
            dyn = np.stack([cv2.dilate((occ[j_of[t]] | newm[t]).astype(np.uint8), k15) > 0 for t in range(rec.T)])
            m = cands[key]["marks"]
            Wd = windows(rec, m)
            hw = np.zeros(rec.T, bool)
            hw[slice(*Wd["handover"])] = True
            r["metrics"]["temporal"] = M2.warp_error_regions(frames, fl, {
                "robot_object": dyn, "background": ~dyn, "robot_object_handover": dyn & hw[:, None, None]})
            # keypoint following: target = this clip's layer homography applied to the original track
            tgt = np.full((rec.T, len(q_pts), 2), np.nan)
            vis_t = np.zeros((rec.T, len(q_pts)), bool)
            for i in range(rec.T):
                j = int(j_of[i])
                for n, ly in enumerate(q_lay):
                    Hm = np.array(r["homographies"][i][ly]) if ly in r["homographies"][i] else np.eye(3)
                    v = Hm @ np.r_[uv_o[j, n], 1.0]
                    tgt[i, n] = v[:2] / v[2]
                    vis_t[i, n] = vis_o[j, n]
            i_ref = int(np.argmin(np.abs(j_of - ref)))
            uv_g, vis_g = trk.track_points(frames, i_ref, tgt[i_ref], args.device, fl)
            r["metrics"]["keypoints"] = M2.keypoint_follow(uv_o, vis_t, uv_g, vis_g, tgt, q_lay,
                                                           {"all": Wd["all"], "handover": Wd["handover"]})
            disp = np.linalg.norm(tgt - uv_o[j_of], axis=-1)
            r["metrics"]["keypoints"]["commanded_displacement_px"] = {
                "median_handover": float(np.nanmedian(disp[slice(*Wd["handover"])])), "max": float(np.nanmax(disp))}
            del fl, frames
            rendered[key] = r
            print(f"  done {key} ({r['runtime_s']:.0f} s render+cpu metrics)", flush=True)
        pool.close()
        pool.join()

    # 4. synchronisation detectors and corruption controls
    with timed("corruption_tests"):
        res["corruption_tests"] = corruption_tests(rec, robot, cam, V, cands, rendered, alias)

    # 5. comparison videos (original | G | P | C) per variation
    with timed("comparison_videos"):
        for v in cfg["variations"]:
            vid = v["id"]
            panels = {}
            for c in pl.CONDITIONS:
                k = f"{vid}__{c}"
                p = cands[k]
                st = f"{c}: ACCEPTED x{p['scale']}" if p["accepted"] else f"{c}: REJECTED ({','.join(p['verify']['failed'][:2])})"
                panels[c] = (np.load(RAW / f"{alias[k]}.npy", mmap_mode="r"), st)
            out = []
            for t in range(rec.T):
                row1 = np.concatenate([rd.label(rec.F[t], f"original ep{E} t={t}"), rd.label(np.array(panels["G"][0][t]), panels["G"][1])], 1)
                row2 = np.concatenate([rd.label(np.array(panels["P"][0][t]), panels["P"][1]), rd.label(np.array(panels["C"][0][t]), panels["C"][1])], 1)
                out.append(cv2.resize(np.concatenate([row1, row2], 0), (960, 720), interpolation=cv2.INTER_AREA))
            rd.write_mp4(OUT / "clips" / f"compare_{vid}.mp4", out, crf=23)

    # 6. results
    res["sigma_G_frames"] = G_["sigma"]
    res["candidates"] = {k: {kk: p[kk] for kk in ("cond", "var", "scale", "accepted", "tries", "ik", "targets", "verify",
                                                   "pass_can_displacement_mm", "runtime_s", "hash", "marks")}
                         | {"render_alias": alias[k]} for k, p in cands.items()}
    res["clips"] = {k: {"metrics": r["metrics"], "runtime_s": r["runtime_s"]} for k, r in rendered.items()}
    res["keypoint_ref_frame"] = ref
    res["runtime_s"] = TIMES
    (OUT / "results.json").write_text(json.dumps(jsonable(res), indent=1))
    np.savez_compressed(CACHE / "render_maps.npz", **{k: r["j_of"] for k, r in rendered.items()})
    acc = {c: sum(cands[f"{v['id']}__{c}"]["accepted"] for v in cfg["variations"]) for c in pl.CONDITIONS}
    print(json.dumps({"episode": E, "accepted": acc, "runtime_s": TIMES}, indent=1))


def corruption_tests(rec, robot, cam, V, cands, rendered, alias) -> dict:
    """detectors must PASS the intended candidate and FLAG each corrupted copy."""
    base_key = next((k for k in ("pos_right_up__C", "pos_left_down__C", "pos_forward__C") if cands[k]["accepted"]), None)
    if base_key is None:
        return {"skipped": "no accepted C position candidate"}
    c = cands[base_key]
    r = rendered[alias[base_key]]
    uv, ok = r["lid_uv"], r["lid_ok"]
    Wd = windows(rec, c["marks"])
    lags = np.arange(-10.0, 7.01, 0.25)
    out = {"base": base_key, "expected_lead": rec.lead, "expected_lag": rec.lag}

    def lag_of(state):
        s = M2.lag_scan(robot, cam, state, rec.X["R"], uv, ok, range(*Wd["giver_hold"]), lags)
        return s["lag"], s["curve"][s["lag"]]

    lag0, e0 = lag_of(c["state"])
    lead0 = sync.action_lead(c["state"], c["action"])["lead"]
    out["intended"] = {"lead": lead0, "lag": lag0, "lag_min_err_px": e0, "verify_ok": V(c)["ok"],
                       "pose_discrimination": r["metrics"].get("pose_discrimination_vs_source_labels")}
    tests = {}
    d = sync.delay_actions(c, 6)
    tests["actions_delayed_6"] = {"lead": sync.action_lead(d["state"], d["action"])["lead"], "verify": V(d)["failed"]}
    tests["actions_delayed_6"]["detected"] = tests["actions_delayed_6"]["lead"] != rec.lead
    for k in (3, 6):
        d = sync.delay_state(c, k)
        lg, e = lag_of(d["state"])
        tests[f"state_delayed_{k}_vs_image"] = {"lag": lg, "lag_min_err_px": e, "detected": abs(lg - rec.lag) > 0.5}
    st_L = next(t for t in range(rec.T) if rb.Robot.split(c["state"])["gL"][t] < recon.CLOSED)
    d = sync.rotate_object(c, 90.0, st_L)
    v = V(d)
    tests["object_rotated_90_after_receiver_grasp"] = {"failed": v["failed"], "detected": not v["ok"]}
    d = sync.rotate_object(c, 20.0, st_L)
    v = V(d)
    tests["object_rotated_20_after_receiver_grasp"] = {"failed": v["failed"], "detected": not v["ok"]}
    ro_gap = c["marks"]["right_open"] - c["marks"]["left_close"]
    d = sync.shift_gripper(c, "L", ro_gap + 6)
    v = V(d)
    tests["receiver_closes_after_giver_opens"] = {"shift": ro_gap + 6, "failed": v["failed"], "detected": not v["ok"]}
    d = sync.shift_gripper(c, "R", -15)
    v = V(d)
    tests["giver_closes_15_early"] = {"failed": v["failed"], "detected": not v["ok"]}
    d = sync.shift_gripper(c, "R", 15)
    v = V(d)
    tests["giver_closes_15_late"] = {"failed": v["failed"], "detected": not v["ok"]}
    # source labels paired with the generated video
    d = sync.source_labels(c, rec)
    lg, e = lag_of(d["state"])
    pdisc = r["metrics"].get("pose_discrimination_vs_source_labels", {})
    tests["source_labels_with_generated_video"] = {
        "lag": lg, "lag_min_err_px": e, "frac_frames_image_prefers_exported_labels": pdisc.get("frac_frames_commanded_better"),
        "detected": bool(e > 2.0 * max(e0, 1.0) or (pdisc.get("frac_frames_commanded_better", 1.0) > 0.5))}
    out["tests"] = tests
    out["all_detected"] = all(t["detected"] for t in tests.values())
    out["intended_passes"] = bool(lead0 == rec.lead and abs(lag0 - rec.lag) <= 0.5 and out["intended"]["verify_ok"])
    return out


if __name__ == "__main__":
    main()
