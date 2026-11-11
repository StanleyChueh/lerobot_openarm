"""Stage 2 (historical): RAFT-large optical flow + flow-chained keypoint tracking exactly as in the frozen v1 code
(research/sim_free_augmentation/sfa/track.py @ e8abc26, imported unchanged), replayed on the UNMODIFIED Pringles source
episode 0 body_cam (the episode v1 itself used). Diagnostic only; RAFT is NOT part of the current sfagen path.

Query points follow the v1/v2 rule (scripts/run_pipeline.py / v2/scripts/run_episode.py::query_points): reference frame
ref = right_close + 2/3 (left_close - right_close); up to 8 random pixels (seed 0) on each of the hand, link7 and finger
links of each arm from the projected URDF label map (eroded 7x7) and 16 on the can (eroded 9x9). Re-implemented with
sfagen's z-buffer (v1's own label map needs the v1 camera/recon objects); differences are documented in the guide.

Outputs: flow colour images, tracking MP4 with trails, survival / forward-backward-error plots, a failure example and a
geometry-consistency plot (RAFT track vs FK projection of the same mesh point: agreement, NOT accuracy)."""
import json, sys
import cv2, numpy as np, torch
from _common import GUIDE, TASKS, REPO, Run
sys.path.insert(0, str(REPO / "research/sim_free_augmentation"))
from sfa import track as trk                                  # frozen v1 module, unchanged
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen import mesh

OUT = GUIDE / "stage_02_tracking"; OUT.mkdir(exist_ok=True); T_ = TASKS["pringles"]; E = 0
with Run("raft_pringles", inputs=[T_["config"], REPO / "research/sim_free_augmentation/sfa/track.py"]) as run:
    ctx = Context(T_["config"]); rec = Recon(ctx, E, "cuda:0"); F = rec.F; T = rec.T
    ev = rec.ev; rc_, lc_ = ev["R"]["closes"][0], [c for c in ev["L"]["closes"] if c > ev["R"]["closes"][0]][0]
    ref = int(rc_ + (2 * (lc_ - rc_)) // 3)
    # ---- query points
    q = {s: rec.gv[s][ref] for s in rec.arms}; g = {s: float(rec.grip_v[s][ref]) for s in rec.grip_v}
    G = rec.G; Ms = G.placements(q, g); _, lab, pt = mesh.splat(G.world(Ms), G.gid, rec.cam, dev="cuda:0")
    lab, pt = lab.cpu().numpy(), pt.cpu().numpy(); names = [o.name for o in rec.robot.vis.geometryObjects]
    rng = np.random.default_rng(0); pts, lay, pid = [], [], []
    for side in "RL":
        for i, (n, s) in enumerate(zip(names, G.side)):
            short = n.split("_", 2)[-1] if n.startswith("openarm_") else n
            if s == side and any(k in n for k in ("hand", "link7", "finger")):
                yy, xx = np.nonzero(cv2.erode((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8)))
                for j in (rng.choice(len(xx), size=min(8, len(xx)), replace=False) if len(xx) else []):
                    pts.append([xx[j], yy[j]]); lay.append(f"{side}:{short}"); pid.append(int(pt[yy[j], xx[j]]))
    yy, xx = np.nonzero(cv2.erode(rec.masks["obj0"][ref], np.ones((9, 9), np.uint8)))
    for j in rng.choice(len(xx), size=min(16, len(xx)), replace=False):
        pts.append([xx[j], yy[j]]); lay.append("can"); pid.append(-1)
    pts = np.array(pts, float); N = len(pts)
    # ---- RAFT flows + tracking (frozen v1 code)
    import time; torch.cuda.synchronize(); t0 = time.time()
    flows = trk.video_flows(F, "cuda:0"); torch.cuda.synchronize(); t_flow = time.time() - t0
    t0 = time.time(); uv, vis = trk.track_points(F, ref, pts, "cuda:0", flows); t_track = time.time() - t0
    # forward-backward error per frame (re-derived with the same flows, for plotting; v1 stores only the boolean)
    fb = np.full((T, N), np.nan)
    for t in range(T):
        if t == ref:
            fb[t] = 0; continue
        prev = uv[t - 1] if t > ref else uv[t + 1]
        k = t - 1 if t > ref else t
        fw = flows["fwd"][k] if t > ref else flows["bwd"][k]; bw = flows["bwd"][k] if t > ref else flows["fwd"][k]
        p = np.clip(prev, 0, [rec.W - 1, rec.H - 1]); d = trk.sample(fw, p); qq = p + d; back = qq + trk.sample(bw, qq)
        fb[t] = np.linalg.norm(back - p, axis=1)
    # continuously-tracked: visible on every frame between ref and t
    cont = np.zeros_like(vis)
    for n in range(N):
        a = True
        for t in range(ref, T):
            a &= vis[t, n]; cont[t, n] = a
        a = True
        for t in range(ref, -1, -1):
            a &= vis[t, n]; cont[t, n] = a
    # geometry consistency for robot points: FK projection of the same mesh point (final lag, calibrated camera)
    geo = np.full((T, N, 2), np.nan)
    rob = [n for n in range(N) if pid[n] >= 0]
    loc = G.local[torch.tensor([pid[n] for n in rob], device="cuda:0")]; gidp = G.gid[torch.tensor([pid[n] for n in rob], device="cuda:0")]
    for t in range(T):
        Mt = torch.tensor(G.placements({s: rec.gv[s][t] for s in rec.arms}, {s: float(rec.grip_v[s][t]) for s in rec.grip_v}), dtype=torch.float32, device="cuda:0")[gidp]
        Pw = (torch.einsum("nij,nj->ni", Mt[:, :3, :3], loc) + Mt[:, :3, 3]).cpu().numpy(); u_, _ = rec.cam.project(Pw); geo[t, rob] = u_
    dgeo = np.linalg.norm(uv - geo, axis=2)
    summary = {"n_query": N, "ref_frame": ref, "layers": sorted(set(lay)), "frames": T,
               "visible_frac_all": float(vis.mean()), "visible_frac_v1_saved_for_comparison": 0.9541552197802198,
               "continuously_tracked_to_ends": {"forward_to_last": float(cont[-1].mean()), "backward_to_first": float(cont[0].mean())},
               "fb_error_px": {"median": float(np.nanmedian(fb)), "p95": float(np.nanpercentile(fb, 95)), "fb_tol_px": trk.FB_TOL},
               "geometry_disagreement_px_robot_points_visible": {"median": float(np.nanmedian(dgeo[:, rob][vis[:, rob]])), "p90": float(np.nanpercentile(dgeo[:, rob][vis[:, rob]], 90))},
               "runtime": {"raft_flows_fwd_and_bwd_s": round(t_flow, 2), "flow_pairs": 2 * (T - 1), "track_s": round(t_track, 2)},
               "model": "torchvision raft_large, weights Raft_Large_Weights.C_T_SKHT_V2, num_flow_updates=12, batch 8, full 480x640 frames",
               "caveat": "no manual ground truth exists (gate1/gate1r annotation_eval.json = PENDING); visibility and FB error are self-consistency; "
                         "geometry disagreement mixes tracker error, camera-calibration error and video/state lag"}
    run.note(**summary)
    # ---- flow colour figure
    fr = [ref - 40, ref, ref + 25, min(T - 2, ref + 60)]
    fig, axs = plt.subplots(2, len(fr), figsize=(4.2 * len(fr), 6.6), constrained_layout=True)
    for i, t in enumerate(fr):
        im = flow_to_image(torch.tensor(flows["fwd"][t]).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        axs[0, i].imshow(F[t]); axs[0, i].set_title(f"real frame {t}", fontsize=9); axs[0, i].axis("off")
        mag = np.linalg.norm(flows["fwd"][t], axis=2)
        axs[1, i].imshow(im); axs[1, i].set_title(f"RAFT flow {t}->{t + 1} (max {mag.max():.1f} px)", fontsize=9); axs[1, i].axis("off")
    fig.suptitle("HISTORICAL v1/v2 stage — RAFT-large dense optical flow on Pringles source ep 0 (colour = direction, saturation = magnitude)", fontweight="bold")
    fig.savefig(run.out(OUT / "raft_flow_colour.png"), dpi=80); plt.close(fig)
    # ---- survival and FB plots
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.3), constrained_layout=True); t = np.arange(T)
    groups = {"right arm": [n for n in range(N) if lay[n].startswith("R")], "left arm": [n for n in range(N) if lay[n].startswith("L")], "can": [n for n in range(N) if lay[n] == "can"]}
    for nm, ix in groups.items():
        axs[0].plot(t, vis[:, ix].mean(1), label=f"{nm}: visible (FB<{trk.FB_TOL:g}px, in image)")
        axs[0].plot(t, cont[:, ix].mean(1), ls="--", label=f"{nm}: continuously tracked since ref")
    axs[0].axvline(ref, color="k", lw=0.8); axs[0].set_xlabel("frame"); axs[0].set_ylabel("fraction of query points"); axs[0].legend(fontsize=7); axs[0].set_title("tracking survival")
    f_ = fb[np.isfinite(fb)]; axs[1].hist(np.clip(f_, 0, 20), bins=80, color="#2a78d6"); axs[1].axvline(trk.FB_TOL, color="r", label=f"tolerance {trk.FB_TOL:g} px")
    axs[1].set_yscale("log"); axs[1].set_xlabel("forward-backward error (px, clipped at 20)"); axs[1].legend(); axs[1].set_title("forward-backward consistency")
    for nm, ix in groups.items():
        ix = [n for n in ix if pid[n] >= 0]
        if ix:
            axs[2].plot(t, np.nanmedian(np.where(vis[:, ix], dgeo[:, ix], np.nan), 1), label=nm)
    axs[2].axvline(ref, color="k", lw=0.8); axs[2].set_xlabel("frame"); axs[2].set_ylabel("px"); axs[2].legend(fontsize=8)
    axs[2].set_title("RAFT track vs FK projection of same mesh point\n(agreement only: includes calibration + lag error)", fontsize=9)
    fig.savefig(run.out(OUT / "raft_survival_fb.png"), dpi=90); plt.close(fig)
    # ---- failure example: robot point with the largest FB error while marked tracked before
    score = np.where(np.isfinite(fb), fb, 0); n_bad = int(np.argmax(score.max(0))); t_bad = int(np.argmax(score[:, n_bad]))
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), constrained_layout=True)
    for i, tt in enumerate([max(0, t_bad - 6), max(0, t_bad - 1), t_bad, min(T - 1, t_bad + 6)]):
        im = F[tt].copy(); u = uv[tt, n_bad]
        if np.isfinite(u).all():
            cv2.circle(im, (int(u[0]), int(u[1])), 8, (255, 0, 0) if vis[tt, n_bad] else (255, 255, 0), 2)
        if np.isfinite(geo[tt, n_bad]).all():
            cv2.drawMarker(im, tuple(int(x) for x in geo[tt, n_bad]), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        axs[i].imshow(im); axs[i].axis("off"); axs[i].set_title(f"frame {tt}: FB {fb[tt, n_bad]:.1f}px, {'tracked' if vis[tt, n_bad] else 'LOST'}", fontsize=9)
    fig.suptitle(f"Failure example — query point {n_bad} ({lay[n_bad]}): circle = RAFT track (red tracked / yellow lost), green cross = FK projection", fontweight="bold", fontsize=10)
    fig.savefig(run.out(OUT / "raft_failure_example.png"), dpi=80); plt.close(fig)
    run.note(failure_example={"point": n_bad, "layer": lay[n_bad], "frame": t_bad, "fb_px": float(fb[t_bad, n_bad])})
    # ---- tracking MP4 with trails
    cmap = {"R": (235, 104, 52), "L": (42, 120, 214), "c": (60, 200, 75)}
    vw = cv2.VideoWriter(str(OUT / "raft_tracking_trails.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480))
    for tt in range(T):
        a = F[tt].copy(); b = flow_to_image(torch.tensor(flows["fwd"][min(tt, T - 2)]).permute(2, 0, 1)).permute(1, 2, 0).numpy().copy()
        for n in range(N):
            c = cmap[lay[n][0]]; lo = max(0, tt - 15) if tt >= ref else tt; hi = tt if tt >= ref else min(T - 1, tt + 15)
            tr = uv[lo:hi + 1, n]; tr = tr[np.isfinite(tr).all(1)].astype(int)
            if len(tr) > 1:
                cv2.polylines(a, [tr.reshape(-1, 1, 2)], False, c, 1)
            if np.isfinite(uv[tt, n]).all():
                p = tuple(uv[tt, n].astype(int)); cv2.circle(a, p, 4, c, -1 if vis[tt, n] else 1)
        cv2.putText(a, f"RAFT tracks t={tt} (ref {ref}) filled=tracked hollow=lost", (8, 24), 0, 0.55, (255, 255, 255), 2)
        cv2.putText(b, "RAFT flow t->t+1", (8, 24), 0, 0.6, (0, 0, 0), 2)
        vw.write(cv2.cvtColor(np.concatenate([a, b], 1), cv2.COLOR_RGB2BGR))
    vw.release(); run.out(OUT / "raft_tracking_trails.mp4")
    np.savez_compressed(GUIDE / "diagnostics/raft_tracks_pringles_ep0.npz", uv=uv, vis=vis, fb=fb, geo=geo, pts=pts, layers=np.array(lay), ref=ref)
    print(json.dumps(run.notes, indent=1, default=str))
