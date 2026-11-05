"""Simulation-free, contact-aware hand-over augmentation: end-to-end prototype on ONE real Pringles episode.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python \
        research/sim_free_augmentation/scripts/run_pipeline.py --config research/sim_free_augmentation/configs/ep0.json

Stages: load -> lid track (Gate 1, automatic) -> lag + can 3D (attachment fits) -> SAM2 masks -> can axis -> plans
(geometry-only G, contact-constrained K) -> candidate actions -> layered warp rendering -> keypoint-control panel
-> side-by-side MP4s -> Stage-3 metrics. No simulator or physics engine is used anywhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa import data, metrics as M, plan as pl, recover as rc, render as rd, robot as rb, segment as sg, track as trk  # noqa: E402

TIMES = {}


def timed(name):
    class _T:
        def __enter__(self):
            self.t = time.time()

        def __exit__(self, *a):
            TIMES[name] = round(time.time() - self.t, 2)
            print(f"[{name}] {TIMES[name]} s", flush=True)
    return _T()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def frame_X(fit):
    a = np.asarray(fit["axis"], float)
    a /= np.linalg.norm(a)
    u = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
    u /= np.linalg.norm(u)
    X = np.eye(4)
    X[:3, :3] = np.stack([u, np.cross(a, u), a], 1)
    X[:3, 3] = fit["p"]
    return X


def video_q(g, Q, lag, T):
    return {s: np.stack([rc.interp_q(Q[s], t + lag) for t in range(T)]) for s in "RL"}


def can_video(robot, Qv, holder_of, X, table, lag, T):
    out = []
    for t in range(T):
        h = holder_of(int(round(t + lag)))
        if h == "table":
            out.append(table)
        else:
            robot.fk(Qv["R"][t], Qv["L"][t])
            out.append(robot.frame(robot.tcp[h]) @ X[h])
    return out


PROMPT_LINKS = ("link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger")


def projected_masks(robot, cam, Qv, grip, cans, T, H=480, W=640):
    """union of projected link4..finger hulls of both arms + the projected can cylinder, per frame."""
    out = np.zeros((T, H, W), np.uint8)
    P = rc.cylinder_points([0, 0, 1.0], 24, 6)
    for t in range(T):
        for name, side, m, z, hull in rb.link_silhouettes(robot, cam, Qv["R"][t], Qv["L"][t], grip["R"][t], grip["L"][t]):
            if name.split("_", 1)[1] in PROMPT_LINKS:
                out[t] |= m
        if cans[t] is not None:
            uv, z = cam.project((cans[t][:3, :3] @ P.T).T + cans[t][:3, 3])
            if (z > 0.05).all():
                cv2.fillConvexPoly(out[t], cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    E = cfg["episode"]
    OUT = HERE / "outputs" / f"ep{E}"
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    (OUT / "cache").mkdir(exist_ok=True)
    dev = cfg.get("device", "cuda:0")
    res = {"config": cfg, "repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                         cwd=HERE).stdout.strip(),
           "dataset": f"{data.DATASET}@{data.REV}", "simulation_used": False}

    # ---------------------------------------------------------------- 1-2 data and formats
    with timed("load"):
        ep = data.load(E)
        F = ep.frames[data.CAMS[0]]
        T, H, W = F.shape[:3]
    robot, cam = rb.Robot(), rb.Camera()
    g, ga = rb.Robot.split(ep.state), rb.Robot.split(ep.action)
    ev = ep.events
    lead = data.action_lead(ep.state, ep.action)
    res["formats"] = {"T": T, "fps": ep.fps, "cameras": {c: list(ep.frames[c].shape) for c in ep.frames},
                      "state_action_names": rb.NAMES, "gripper_open_raw": rb.GRIP_OPEN_RAW, "events": ev,
                      "action_lead": lead, "camera_model": rb.BODY_CAM["source"]}

    # ---------------------------------------------------------------- lid track (automatic, Gate 1 tracker output)
    z = np.load(Path(cfg["lid_track"]))
    lid_uv, lid_ok = z["uv"], z["status"] == "ok"
    res["lid_track"] = {"source": cfg["lid_track"], "ok_frac": float(lid_ok.mean())}

    # ---------------------------------------------------------------- 6 motion recovery
    with timed("recover"):
        giv = [t for t in range(ev["right_close"] + 10, ev["left_close"] - 3) if lid_ok[t]]
        rec = [t for t in range(ev["right_open"] + 8, T) if lid_ok[t]]
        lagfit = rc.estimate_lag(robot, cam, g, "R", giv, lid_uv)
        lag = lagfit["lag"]
        fits = {"R": rc.fit_point(robot, cam, g, "R", giv, lid_uv, lag),
                "L": rc.fit_point(robot, cam, g, "L", rec, lid_uv, lag)}
        gv = video_q(g, g, lag, T)
        grip = {s: np.array([np.interp(t + lag, np.arange(T), g["g" + s]) for t in range(T)]) for s in "RL"}

    # ---------------------------------------------------------------- 4 segmentation (SAM2)
    keyframes = list(range(0, T, int(cfg["sam2_keyframe_step"]))) + [T - 1]
    mpath = OUT / "cache/sam2_masks_original.npz"
    with timed("sam2_original"):
        if mpath.exists():
            masks = dict(np.load(mpath))
            prompts = json.loads((OUT / "cache/sam2_prompts.json").read_text())
        else:
            masks, prompts = sg.segment(F, robot, cam, gv, lid_uv, lid_ok, keyframes, dev, grip=grip)
            np.savez_compressed(mpath, **masks)
            (OUT / "cache/sam2_prompts.json").write_text(json.dumps(jsonable(prompts)))
    res["segmentation"] = {"model": sg.MODEL, "keyframes": keyframes, "n_prompts": len(prompts), "arm_prompt": "mask",
                           "mask_area_median_px": {k: float(np.median(v.reshape(T, -1).sum(1))) for k, v in masks.items()},
                           "frames_with_empty_mask": {k: int((v.reshape(T, -1).sum(1) == 0).sum()) for k, v in masks.items()}}

    with timed("can_axis"):
        sel = lambda fr: [t for t in fr[:: max(1, len(fr) // 12)] if masks["can"][t].any()]   # noqa: E731
        for s, fr in (("R", giv), ("L", rec)):
            fits[s].update(rc.fit_axis(robot, cam, g, s, fits[s]["p"], sel(fr), masks["can"], lag))
        X = {s: frame_X(fits[s]) for s in "RL"}
        # table phase: the standing can lies on its observed lid ray at the lid height recovered from the grasp
        tg = ev["right_close"] + 8
        T_att = rc.tcp_pose(robot, g, "R", tg + lag) @ X["R"]
        pre = [t for t in range(0, ev["right_close"]) if lid_ok[t]]
        u = np.median(lid_uv[pre], axis=0)
        ray = cam.R @ np.linalg.solve(cam.K, np.r_[u, 1.0])
        s_ = (T_att[2, 3] - cam.t[2]) / ray[2]
        table = T_att.copy()
        table[:3, 3] = cam.t + s_ * ray
        uproj, _ = cam.project(T_att[:3, 3][None])
        res["recovery"] = {"lag_frames": lag, "lag_rms_by_lag": lagfit["rms_by_lag"],
                           "fit_R": {k: fits["R"][k] for k in ("p", "rms_px", "median_px", "p90_px", "n", "std_m", "axis", "mean_iou")},
                           "fit_L": {k: fits["L"][k] for k in ("p", "rms_px", "median_px", "p90_px", "n", "std_m", "axis", "mean_iou")},
                           "table_lid_world": table[:3, 3],
                           "grasp_vs_table_lid_px": float(np.linalg.norm(uproj[0] - u)),
                           "note": "3D only where a hand holds the can (structure from known hand motion); the standing "
                                   "can uses its lid ray at the recovered lid height"}

    # ---------------------------------------------------------------- 7-8 plans and candidate actions
    holder_orig = lambda t: "table" if t < ev["right_close"] + 8 else ("R" if t < ev["right_open"] else "L")  # noqa: E731
    plans = {}
    with timed("plans"):
        for key, kind in (("geometry_only", "geometry"), ("contact_constrained", "contact")):
            p = pl.make_plan(robot, g, ev, X, table, ep.fps, kind, cfg["delta_m"], seed=cfg["seed"])
            p.action = pl.actions(g, ga, ep.action, p, lead["k_best"])
            p.checks["actions"] = pl.action_checks(robot, p.action, p.q, lead["k_best"])
            plans[key] = p
            st = rb.Robot.merge(p.q["R"], p.q["L"], ep.state[:, rb.NAMES.index("RJ8.pos")], ep.state[:, rb.NAMES.index("LJ8.pos")])
            np.savez_compressed(OUT / f"candidate_{key}.npz", state=st, action=p.action, names=np.array(rb.NAMES),
                                offsets_R=p.offsets["R"], offsets_L=p.offsets["L"], source_state=ep.state,
                                source_action=ep.action)
            import csv
            with open(OUT / f"candidate_actions_{key}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["frame"] + rb.NAMES)
                for t in range(T):
                    w.writerow([t] + [f"{v:.6f}" for v in p.action[t]])
        res["original_action_checks"] = pl.action_checks(robot, ep.action, {"R": g["R"], "L": g["L"]}, lead["k_best"])

    # ---------------------------------------------------------------- background plate
    with timed("plate"):
        occ = (masks["R"] > 0) | (masks["L"] > 0) | (masks["can"] > 0)
        plate, never = rd.background_plate(F, occ)
        cv2.imwrite(str(OUT / "background_plate.png"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
        res["plate"] = {"never_free_frac": float(never.mean())}

    # ---------------------------------------------------------------- 9-10 rendering
    can_old = can_video(robot, gv, holder_orig, X, table, lag, T)
    clips = {}
    for key, p in plans.items():
        with timed(f"render_{key}"):
            qv = video_q(g, p.q, lag, T)
            hold = (lambda t, p=p: p.can_holder[int(np.clip(t, 0, T - 1))])
            can_new = can_video(robot, qv, hold, X, table, lag, T)
            gen, newm, infos = [], [], []
            for t in range(T):
                fr, nm, info = rd.render_frame(t, F[t], plate, robot, cam, {s: gv[s][t] for s in "RL"},
                                               {s: qv[s][t] for s in "RL"}, {s: grip[s][t] for s in "RL"},
                                               {k: masks[k][t] for k in masks}, can_old[t], can_new[t], [0, 0, 1.0])
                gen.append(fr)
                newm.append(nm)
                infos.append(info)
            gen = np.stack(gen)
            kpv = []
            for t in range(T):
                kp = rd.keypoints_3d(robot, qv["R"][t], qv["L"][t], grip["R"][t], grip["L"][t], can_new[t])
                kpv.append(rd.draw_keypoints((plate * 0.35).astype(np.uint8), cam, kp))
            kpv = np.stack(kpv)
            rd.write_mp4(OUT / "clips" / f"{key}_generated.mp4", gen)
            rd.write_mp4(OUT / "clips" / f"{key}_keypoint_control.mp4", kpv)
            sbs = [np.concatenate([rd.label(F[t], f"original ep{E} t={t}"), rd.label(kpv[t], "keypoint control (target)"),
                                   rd.label(gen[t], f"generated: {key}")], 1) for t in range(T)]
            rd.write_mp4(OUT / "clips" / f"{key}_side_by_side.mp4", sbs)
            clips[key] = {"gen": gen, "newmask": np.stack(newm), "infos": infos, "qv": qv, "can_new": can_new}
            np.savez_compressed(OUT / "cache" / f"gen_{key}.npz", gen=gen)
    # 4-panel comparison
    comp = [np.concatenate([np.concatenate([rd.label(F[t], "original"), rd.label(clips["geometry_only"]["gen"][t], "geometry-only")], 1),
                            np.concatenate([rd.label(clips["contact_constrained"]["gen"][t], "contact-constrained"),
                                            rd.label(np.stack([rd.draw_keypoints(F[t] // 2, cam, rd.keypoints_3d(
                                                robot, clips["contact_constrained"]["qv"]["R"][t], clips["contact_constrained"]["qv"]["L"][t],
                                                grip["R"][t], grip["L"][t], clips["contact_constrained"]["can_new"][t]))])[0],
                                                "contact-constrained target on original")], 1)], 0) for t in range(T)]
    rd.write_mp4(OUT / "clips" / "comparison_original_geometry_contact.mp4", comp)

    # ---------------------------------------------------------------- 11 metrics
    ev_metrics = {}
    with timed("flows"):
        flows = {"original": trk.video_flows(F, dev)}
        for key in plans:
            flows[key] = trk.video_flows(clips[key]["gen"], dev)
    with timed("metrics"):
        static = ~(occ | np.stack([clips[k]["newmask"] for k in plans]).any(0))
        static = np.stack([cv2.erode(s.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0 for s in static])
        ev_metrics["temporal"] = {k: {kk: vv for kk, vv in M.temporal_warp_error(
            F if k == "original" else clips[k]["gen"], flows[k], static).items() if kk != "warp_err_per_frame"}
            for k in flows}
        # restricted to the robot/can region (original masks + where the edited layers landed), handover window
        dyn = np.stack([cv2.dilate(((occ[t]) | np.stack([clips[k]["newmask"][t] for k in plans]).any(0)).astype(np.uint8),
                                   np.ones((15, 15), np.uint8)) > 0 for t in range(T)])
        hw = slice(ev["right_close"], min(T, ev["right_open"] + 40))
        for k in flows:
            vid = F if k == "original" else clips[k]["gen"]
            sub = {"fwd": flows[k]["fwd"][hw.start:hw.stop - 1], "bwd": flows[k]["bwd"][hw.start:hw.stop - 1]}
            r = M.temporal_warp_error(vid[hw], sub, dyn[hw])
            ev_metrics["temporal"][k]["handover_window_warp_err_mean"] = r["warp_err_mean"]
            ev_metrics["temporal"][k]["handover_window_dynamic_region_frame_diff"] = r.get("static_flicker_mean")
        # keypoint following: query points on the right hand/link7, left hand/link7 and the can at a reference frame
        ref = int(cfg["kp_ref_frame"])
        labels = rd.link_label_map(robot, cam, gv["R"][ref], gv["L"][ref], grip["R"][ref], grip["L"][ref],
                                   {k: masks[k][ref] for k in masks})
        rng = np.random.default_rng(0)
        names = [n for n, s, P in robot.link_points("hull")]
        sides = [s for n, s, P in robot.link_points("hull")]
        q_pts, q_layer = [], []
        for side in "RL":
            for i, (n, s) in enumerate(zip(names, sides)):
                if s == side and n.split("_", 1)[1] in ("hand", "link7", "left_finger", "right_finger"):
                    yy, xx = np.nonzero(cv2.erode((labels[side] == i).astype(np.uint8), np.ones((7, 7), np.uint8)))
                    for j in rng.choice(len(xx), size=min(8, len(xx)), replace=False) if len(xx) else []:
                        q_pts.append([xx[j], yy[j]]); q_layer.append(f"{side}_{n}")
        yy, xx = np.nonzero(cv2.erode(masks["can"][ref], np.ones((9, 9), np.uint8)))
        for j in rng.choice(len(xx), size=min(16, len(xx)), replace=False):
            q_pts.append([xx[j], yy[j]]); q_layer.append("can")
        q_pts = np.array(q_pts, float)
        uv_o, vis_o = trk.track_points(F, ref, q_pts, dev, flows["original"])
        ev_metrics["keypoints"] = {"n_query": len(q_pts), "ref_frame": ref, "layers": sorted(set(q_layer)),
                                   "original_track_visible_frac": float(vis_o.mean())}
        for key in plans:
            infos = clips[key]["infos"]
            tgt = np.full_like(uv_o, np.nan)
            for t in range(T):
                for i, ly in enumerate(q_layer):
                    Hm = np.array(infos[t]["homographies"][ly]) if ly in infos[t]["homographies"] else np.eye(3)
                    p = Hm @ np.r_[uv_o[t, i], 1.0]
                    tgt[t, i] = p[:2] / p[2]
            uv_g, vis_g = trk.track_points(clips[key]["gen"], ref, tgt[ref], dev, flows[key])
            clips[key]["kp_uv"] = uv_g
            fe = M.follow_error(uv_g, vis_g, tgt, vis_o)
            fw = M.window_follow(uv_g, vis_g, tgt, vis_o, ev["right_close"], min(T, ev["right_open"] + 40))
            disp_w = np.linalg.norm(tgt - uv_o, axis=-1)[ev["right_close"]:ev["right_open"] + 40]
            per = {}
            for ly in ("can", "R", "L"):
                idx = [i for i, l in enumerate(q_layer) if l == ly or l.startswith(ly + "_")]
                per[ly] = M.follow_error(uv_g[:, idx], vis_g[:, idx], tgt[:, idx], vis_o[:, idx])
            ev_metrics["keypoints"][key] = {"all": fe, "by_object": per, "handover_window_follow": fw,
                                            "target_displacement_px_median_handover": float(np.nanmedian(disp_w)),
                                            "target_displacement_px_max": float(np.nanmax(np.linalg.norm(tgt - uv_o, axis=-1)))}
        # handover-window keypoint following (the edited part of the clip)
        w0, w1 = ev["right_close"], min(T, ev["right_open"] + 40)
        for key in plans:
            ev_metrics["keypoints"][key]["handover_window"] = {"frames": [w0, w1]}
        # robot/image alignment, prompt-free: plate-difference foreground vs projected commanded / alternative state
        proj_o = projected_masks(robot, cam, gv, grip, can_old, T)
        edited = list(range(w0, w1, 3))
        ev_metrics["pose_alignment"] = {}
        for key in plans:
            proj_n = projected_masks(robot, cam, clips[key]["qv"], grip, clips[key]["can_new"], T)
            ev_metrics["pose_alignment"][key] = {
                "generated_clip": M.pose_discrimination(clips[key]["gen"], plate, proj_n, proj_o, edited),
                "original_clip_control": M.pose_discrimination(F, plate, proj_o, proj_n, edited)}
    for key in plans:
        with timed(f"lid_{key}"):
            sys.path.insert(0, str(HERE.parents[1] / "research/contact_aware_augmentation/gate1"))
            import lid_tracker as lt
            uv0, _ = cam.project(clips[key]["can_new"][0][:3, 3][None])
            tr = lt.track(clips[key]["gen"], uv0[0])
            gl_uv, gl_ok = tr.uv, tr.ok()
            p = plans[key]
            giv_g = [t for t in range(ev["right_close"] + 10, ev["left_close"] - 3) if gl_ok[t]]
            rec_g = [t for t in range(ev["right_open"] + 8, T) if gl_ok[t]]
            att = {}
            for s, fr in (("R", giv_g), ("L", rec_g)):
                att[s] = rc.fit_point(robot, cam, p.q, s, fr, gl_uv, lag) if len(fr) >= 8 else {"n": len(fr)}
                att[s] = {k: v for k, v in att[s].items() if k in ("rms_px", "median_px", "p90_px", "n")}
            ev_metrics.setdefault("attachment_in_video", {})[key] = {
                "lid_ok_frac": float(gl_ok.mean()), "giver_fit": att["R"], "receiver_fit": att["L"],
                "lid_jumps": M.lid_jumps(gl_uv, gl_ok)}
            clips[key]["lid"] = (gl_uv, gl_ok)
    # receiver fingers vs can during the transfer, measured in each clip (tracked left-finger points vs lid track)
    lf = [i for i, l in enumerate(q_layer) if l.startswith("L_") and "finger" in l]
    win = range(ev["left_close"], min(T, ev["right_open"] + 6))
    def finger_lid(uvf, lid_uv_, lid_ok_):
        d = [np.linalg.norm(np.nanmean(uvf[t, lf], axis=0) - lid_uv_[t]) for t in win if np.isfinite(lid_uv_[t]).all()]
        return float(np.median(d)) if d else None
    ev_metrics["transfer_receiver_finger_to_lid_px"] = {"original": finger_lid(uv_o, lid_uv, lid_ok)}
    for key in plans:
        uv_g = clips[key]["kp_uv"]
        ev_metrics["transfer_receiver_finger_to_lid_px"][key] = finger_lid(uv_g, *clips[key]["lid"])
    ev_metrics.setdefault("attachment_in_video", {})["original"] = {
        "lid_ok_frac": float(lid_ok.mean()), "giver_fit": {k: fits["R"][k] for k in ("rms_px", "median_px", "p90_px", "n")},
        "receiver_fit": {k: fits["L"][k] for k in ("rms_px", "median_px", "p90_px", "n")},
        "lid_jumps": M.lid_jumps(lid_uv, lid_ok)}
    res["plans"] = {k: {"params": p.params, "checks": p.checks} for k, p in plans.items()}
    res["evidence"] = ev_metrics
    res["runtime_s"] = TIMES
    res["render_info"] = {k: {"median_layers": float(np.median([i["n_layers"] for i in clips[k]["infos"]])),
                              "median_moved_px": float(np.median([i["moved_px"] for i in clips[k]["infos"]]))} for k in clips}
    (OUT / "results.json").write_text(json.dumps(jsonable(res), indent=1))
    # stills for the report
    for t in cfg["still_frames"]:
        row = np.concatenate([rd.label(F[t], f"original t={t}"), rd.label(clips["geometry_only"]["gen"][t], "geometry-only"),
                              rd.label(clips["contact_constrained"]["gen"][t], "contact-constrained")], 1)
        cv2.imwrite(str(OUT / f"still_t{t:03d}.jpg"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print(json.dumps(jsonable({"runtime_s": TIMES, "checks": {k: plans[k].checks for k in plans},
                               "evidence": ev_metrics}), indent=1)[:6000])


if __name__ == "__main__":
    main()
