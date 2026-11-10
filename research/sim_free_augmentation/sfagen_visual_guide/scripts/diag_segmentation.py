"""Stage 1 (segmentation) figures, MP4 and statistics for one source episode. Diagnostic only.

Uses the EXISTING cached SAM2 masks (configs/work/<task>/cache/sam2_ep<E>.npz) for all figures and statistics.
New diagnostic computation (recorded in diagnostics/run_seg_<task>.json):
  * URDF mask prompts re-projected (deterministic; sfagen.segment.arm_prompt) at the video/state lag the cached masks
    were prompted with (the pre-photometric lag; see notes);
  * the frame-0 "what disappears when picked up" object segment (replica of Recon._segment_object's first step);
  * one SAM2 re-run (sfagen.segment.segment_arms, same model/settings) ONLY to measure latency / GPU memory and
    run-to-run reproducibility against the cache. The cache is not overwritten.
usage: diag_segmentation.py pringles|battery [--no-rerun]"""
import json, sys, time
import cv2, numpy as np, torch
from _common import GUIDE, TASKS, Run
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen.segment import arm_prompt, segment_arms, MODEL

task = sys.argv[1]; T_ = TASKS[task]; OUT = GUIDE / "stage_01_segmentation"; OUT.mkdir(exist_ok=True)
E = T_["source_episode"]; cache = T_["work"] / f"cache/sam2_ep{E}.npz"; rcache = T_["work"] / f"cache/recon_ep{E}.npz"
COL = {"R": (235, 104, 52), "L": (42, 120, 214), "obj": (60, 180, 75)}            # right arm orange, left arm blue, objects green


def distal_sides(rec, j, excl):
    """per-pixel arm label of the z-buffered URDF silhouette using ONLY the links SAM2 is prompted with (link0-3 excluded),
    at the final lag and calibrated camera."""
    from sfagen import mesh
    q = {s: rec.gv[s][j] for s in rec.arms}; g = {s: float(rec.grip_v[s][j]) for s in rec.grip_v}
    _, lab, _ = mesh.splat(rec.G.world(rec.G.placements(q, g)), rec.G.gid, rec.cam, dev=rec.dev)
    names = [o.name for o in rec.robot.vis.geometryObjects]
    side = np.array([("X" if any(x in n for x in excl) else s) for n, s in zip(names, rec.G.side)] + ["X"])
    sd = side[lab.cpu().numpy()]; out = np.full(sd.shape, "X", dtype=object)
    for a in rec.arms:                                   # close the 3-mm point-splat speckle (near-camera links are sparse)
        out[cv2.morphologyEx((sd == a).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0] = a
    return out


def iou(a, b):
    a, b = a > 0, b > 0; u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


with Run(f"seg_{task}", inputs=[T_["config"], cache, rcache]) as run:
    ctx = Context(T_["config"]); rec = Recon(ctx, E, "cuda:0")
    info = json.loads(str(np.load(rcache)["info"])); rc = ctx.cfg["reconstruction"]
    sc = info["lag_scan"]; base = round(float(rc["lag_default"]), 3); best = max(sc, key=sc.get)
    lag_sam = float(best) if sc[best] > sc.get(base, sc.get(str(base), -1)) + 0.01 else float(base)   # Recon.__init__ rule
    lag_final = float(rec.lag)
    sam = np.load(cache); SAM = {s: sam[s] for s in rec.arms}
    run.note(sam2_cache=str(cache), sam2_model=MODEL, keyframe_every=rc["sam2_keyframe_every"], chunk=150,
             exclude_links=rc["sam2_exclude_links"], lag_used_for_sam2_prompts=lag_sam, lag_final_photometric=lag_final,
             cache_has_sam2_meta="sam2" in info, sam2_meta_in_cache=info.get("sam2"))

    # ---------- per-frame statistics from the cached masks (no SAM2 run)
    T = rec.T; stats = {s: {"area": [], "iou_mesh": [], "iou_prev": []} for s in rec.arms}; objarea = []
    for j in range(T):
        sd = distal_sides(rec, j, tuple(rc["sam2_exclude_links"]))   # distal-link mesh silhouette, FINAL lag
        for s in rec.arms:
            stats[s]["area"].append(int((SAM[s][j] > 0).sum()))
            stats[s]["iou_mesh"].append(iou(SAM[s][j], sd == s) if (sd == s).sum() > 300 else np.nan)
            stats[s]["iou_prev"].append(iou(SAM[s][j], SAM[s][j - 1]) if j else np.nan)
        objarea.append([int((rec.masks[f"obj{k}"][j] > 0).sum()) for k in range(len(rec.objects))])
    objarea = np.array(objarea)
    summ = {}
    for s in rec.arms:
        a = np.array(stats[s]["area"]); im = np.array(stats[s]["iou_mesh"]); ip = np.array(stats[s]["iou_prev"])
        vis = ~np.isnan(im)
        summ[s] = {"frames": T, "frames_mesh_visible_gt300px": int(vis.sum()), "frames_sam2_nonempty": int((a > 0).sum()),
                   "coverage_sam2_nonempty_when_mesh_visible": float(((a > 0) & vis).sum() / max(1, vis.sum())),
                   "iou_sam2_vs_mesh_silhouette": {"median": float(np.nanmedian(im)), "p10": float(np.nanpercentile(im, 10)), "min": float(np.nanmin(im)),
                                                   "argmin_frame": int(np.nanargmin(im))},
                   "frame_to_frame_iou": {"median": float(np.nanmedian(ip)), "p5": float(np.nanpercentile(ip, 5)), "min": float(np.nanmin(ip)),
                                          "argmin_frame": int(np.nanargmin(ip))},
                   "frames_with_area_jump_gt_30pct": int((np.abs(np.diff(a)) > 0.3 * np.maximum(a[:-1], 1)).sum())}
    summ["objects"] = {o["name"]: {"frames_nonempty": int((objarea[:, k] > 0).sum()), "median_px_when_nonempty": float(np.median(objarea[objarea[:, k] > 0, k])) if (objarea[:, k] > 0).any() else 0}
                       for k, o in enumerate(rec.objects)}
    run.note(statistics=summ, statistics_caveat="IoU vs the projected URDF mesh silhouette is AGREEMENT between two automatic estimates, not accuracy; "
             "no manual ground-truth masks exist for these episodes, so no mIoU/Dice/accuracy is reported.")

    # ---------- prompts at the lag SAM2 was prompted with
    rec.lag = lag_sam; rec._video_states()
    chunks = list(range(0, T, 150)); keyframes = [c0 + k for c0 in chunks for k in range(0, min(150, T - c0), 15)]
    prompts = {j: {s: arm_prompt(rec, j, s, tuple(rc["sam2_exclude_links"])) for s in rec.arms} for j in keyframes}
    rec.lag = lag_final; rec._video_states()

    # ---------- "what disappears when picked up" object segment (frame 0), replica of Recon._segment_object step 1
    dis = {}
    for k, o in enumerate(rec.objects):
        oc = ctx.cfg["objects"][o["name"]]; arm = oc["grasped_by"]; tg = rec.ev[arm]["closes"][oc.get("grasp_index", 0)]
        r = cv2.dilate(rec.robot_sil(0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        ref, valid = rec._after_ref(tg + 20)
        fg = (np.abs(rec.F[0].astype(np.int16) - ref.astype(np.int16)).max(2) > rec.fg_thr) & valid
        fg = (cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0) & ~r
        dis[o["name"]] = {"fg": fg, "ref": ref, "grasp_frame": int(tg)}

    # ---------- figure A-F for two keyframes
    pick = {"pringles": [90, 150], "battery": [270, 450]}[task]
    fig, axs = plt.subplots(len(pick) + 1, 6, figsize=(24, 4.1 * (len(pick) + 1)), constrained_layout=True)
    def show(ax, im, title):
        ax.imshow(im); ax.set_title(title, fontsize=9); ax.axis("off")
    for r_, j in enumerate(pick):
        F = rec.F[j]; P = prompts[j]
        pv = F.copy() // 2
        for s in rec.arms:
            cnt, _ = cv2.findContours(P[s].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(pv, cnt, -1, COL[s], 2)
            pv[P[s] > 0] = (0.6 * pv[P[s] > 0] + 0.4 * np.array(COL[s])).astype(np.uint8)
        def mview(m, c):
            v = (F * 0.35).astype(np.uint8); v[m > 0] = (0.35 * F[m > 0] + 0.65 * np.array(c)).astype(np.uint8); return v
        om = np.any([rec.masks[f"obj{k}"][j] > 0 for k in range(len(rec.objects))], 0)
        ov = F.copy()
        for s in rec.arms:
            m = rec.masks[s][j] > 0; ov[m] = (0.5 * ov[m] + 0.5 * np.array(COL[s])).astype(np.uint8)
        ov[om] = (0.4 * ov[om] + 0.6 * np.array(COL["obj"])).astype(np.uint8)
        for s, name in (("R", "right arm"), ("L", "left arm")):
            ys, xs = np.nonzero(rec.masks[s][j])
            if len(xs):
                cv2.putText(ov, name, (int(np.median(xs)) - 30, int(np.median(ys))), 0, 0.6, (255, 255, 255), 2)
        for k, o in enumerate(rec.objects):
            ys, xs = np.nonzero(rec.masks[f"obj{k}"][j])
            if len(xs):
                cv2.putText(ov, o["name"], (int(np.median(xs)) - 20, int(np.min(ys)) - 6), 0, 0.6, (255, 255, 255), 2)
        show(axs[r_, 0], F, f"A  real body_cam, source ep {E}, frame {j}")
        show(axs[r_, 1], pv, f"B  URDF mask prompts (keyframe; lag {lag_sam:+.2f})\norange = right arm, blue = left arm")
        show(axs[r_, 2], mview(SAM["L"][j], COL["L"]), "C  SAM2.1 left-arm mask (cached)")
        show(axs[r_, 3], mview(SAM["R"][j], COL["R"]), "D  SAM2.1 right-arm mask (cached)")
        show(axs[r_, 4], mview(om, COL["obj"]), "E  object mask (NOT SAM2):\nplate difference inside projected proxy, minus robot")
        show(axs[r_, 5], ov, "F  final overlay used by the renderer\n(arm masks minus object pixels)")
    # bottom row: object initialization ("what disappears")
    nm = list(dis)[0]; d = dis[nm]
    show(axs[-1, 0], rec.F[0], f"frame 0 (object resting on the table)")
    show(axs[-1, 1], d["ref"], f"robot-free median after grasp+20 (frame {d['grasp_frame'] + 20}+)")
    v = (rec.F[0] * 0.35).astype(np.uint8); v[d["fg"]] = (0.35 * rec.F[0][d["fg"]] + 0.65 * np.array(COL["obj"])).astype(np.uint8)
    show(axs[-1, 2], v, f"'what disappears when picked up': {nm}\n|frame0 - after| > {rec.fg_thr}, minus robot (before connected-component filter)")
    for k, o in enumerate(rec.objects):
        sil = rec.obj_sil(k, o["table"]); vv = rec.F[0].copy(); cnt, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vv, cnt, -1, (60, 255, 75), 2)
        oi = info["objects"][o["name"]]
        show(axs[-1, 3 + k], vv, f"fitted {o['proxy']} proxy of '{o['name']}' at rest\n" + (f"silhouette IoU {oi['silhouette_iou']:.2f} ({oi['source']})" if "silhouette_iou" in oi else f"source: {oi['source']}"))
    for kk in range(3 + len(rec.objects), 6):
        axs[-1, kk].axis("off")
    fig.suptitle(f"Stage 1 — segmentation, {task} source episode {E} (cached SAM2.1-hiera-small masks; prompts re-projected for display)", fontweight="bold")
    fig.savefig(run.out(OUT / f"{task}_segmentation_panels.png"), dpi=80); plt.close(fig)

    # ---------- statistics figure
    t = np.arange(T)
    fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    for s in rec.arms:
        c = np.array(COL[s]) / 255
        axs[0].plot(t, stats[s]["area"], color=c, label=f"{s} arm SAM2 area (px)")
        axs[1].plot(t, stats[s]["iou_mesh"], color=c, label=f"{s}: SAM2 vs projected URDF silhouette (prompted links only)")
        axs[2].plot(t, stats[s]["iou_prev"], color=c, label=f"{s}: SAM2 frame t vs t-1")
    for ax in axs:
        for c0 in chunks[1:]:
            ax.axvline(c0, color="k", lw=0.8, ls="--")
        for j in keyframes:
            ax.axvline(j, color="0.8", lw=0.4, zorder=0)
        ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("pixels"); axs[1].set_ylabel("IoU (agreement)"); axs[2].set_ylabel("IoU"); axs[2].set_xlabel("frame (dashed = 150-frame chunk boundary, grey = prompt keyframes)")
    axs[1].set_ylim(0, 1); axs[2].set_ylim(0, 1)
    fig.suptitle(f"{task} ep {E}: SAM2 mask coverage and temporal stability (no manual ground truth: agreement, not accuracy)", fontweight="bold")
    fig.savefig(run.out(OUT / f"{task}_segmentation_stability.png"), dpi=90); plt.close(fig)

    # ---------- failure examples: lowest agreement with the mesh silhouette and lowest frame-to-frame IoU
    fails = []
    for s in rec.arms:
        fails.append((s, "lowest SAM2-vs-mesh IoU", summ[s]["iou_sam2_vs_mesh_silhouette"]["argmin_frame"], summ[s]["iou_sam2_vs_mesh_silhouette"]["min"]))
        fails.append((s, "lowest frame-to-frame IoU", summ[s]["frame_to_frame_iou"]["argmin_frame"], summ[s]["frame_to_frame_iou"]["min"]))
    fig, axs = plt.subplots(len(fails), 3, figsize=(13, 3.4 * len(fails)), constrained_layout=True)
    for r_, (s, why, j, val) in enumerate(fails):
        sd = distal_sides(rec, j, tuple(rc["sam2_exclude_links"])) == s; F = rec.F[j]
        a = (F * 0.4).astype(np.uint8); a[SAM[s][j] > 0] = (0.4 * F[SAM[s][j] > 0] + 0.6 * np.array(COL[s])).astype(np.uint8)
        b = (F * 0.4).astype(np.uint8); b[sd] = (0.4 * F[sd] + 0.6 * np.array([200, 200, 200])).astype(np.uint8)
        show(axs[r_, 0], F, f"frame {j}"); show(axs[r_, 1], a, f"{s} arm SAM2 mask — {why} ({val:.2f})")
        show(axs[r_, 2], b, f"{s} arm projected URDF silhouette, links SAM2 is prompted with\n(calibrated camera, final lag)")
    fig.suptitle(f"{task} ep {E}: worst segmentation frames (automatic selection)", fontweight="bold")
    fig.savefig(run.out(OUT / f"{task}_segmentation_failures.png"), dpi=70); plt.close(fig)

    # ---------- overlay MP4 (every frame for Pringles; every 2nd frame for battery = 2x time-lapse)
    step = 1 if T < 400 else 2
    vw = cv2.VideoWriter(str(OUT / f"{task}_segmentation_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480))
    for j in range(0, T, step):
        F = rec.F[j]; ov = F.copy()
        for s in rec.arms:
            m = rec.masks[s][j] > 0; ov[m] = (0.5 * ov[m] + 0.5 * np.array(COL[s])).astype(np.uint8)
        for k in range(len(rec.objects)):
            m = rec.masks[f"obj{k}"][j] > 0; ov[m] = (0.4 * ov[m] + 0.6 * np.array(COL["obj"])).astype(np.uint8)
        if j in prompts:
            for s in rec.arms:
                cnt, _ = cv2.findContours(prompts[j][s].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov, cnt, -1, (255, 255, 255), 2)
        a = F.copy(); cv2.putText(a, f"real ep{E} t={j}", (10, 26), 0, 0.8, (255, 255, 0), 2)
        cv2.putText(ov, "SAM2 R (orange) L (blue), object (green)" + ("  [URDF prompt]" if j in prompts else ""), (10, 26), 0, 0.6, (255, 255, 255), 2)
        vw.write(cv2.cvtColor(np.concatenate([a, ov], 1), cv2.COLOR_RGB2BGR))
    vw.release(); run.out(OUT / f"{task}_segmentation_overlay.mp4")

    # ---------- SAM2 re-run: latency, memory, reproducibility vs cache (masks kept in diagnostics/, gitignored)
    if "--no-rerun" not in sys.argv:
        rec.lag = lag_sam; rec._video_states(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        new, meta = segment_arms(rec, "cuda:0", rc["sam2_keyframe_every"], tuple(rc["sam2_exclude_links"]))
        torch.cuda.synchronize(); dt = time.time() - t0; peak = torch.cuda.max_memory_allocated() / 2 ** 30
        rep = {s: [iou(new[s][j], SAM[s][j]) for j in range(T) if (new[s][j] > 0).any() or (SAM[s][j] > 0).any()] for s in rec.arms}
        run.note(sam2_rerun={"wall_s_incl_model_load": round(dt, 2), "frames": T, "ms_per_frame": round(1000 * dt / T, 1), "peak_torch_gib": round(peak, 3),
                             "prompts": meta["prompts"], "dtype": "bfloat16", "lag_used": lag_sam,
                             "iou_rerun_vs_cache": {s: {"median": float(np.median(v)), "min": float(np.min(v)), "frames_lt_0.9": int((np.array(v) < 0.9).sum())} for s, v in rep.items()}})
        np.savez_compressed(GUIDE / f"diagnostics/sam2_rerun_{task}_ep{E}.npz", **new)
    print(json.dumps(run.notes, indent=1, default=str)[:3000])
