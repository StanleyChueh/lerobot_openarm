"""Stage 3 (camera calibration, FK, table, object proxies, holders, video/state lag) figures for one source episode.
Diagnostic only. Inputs are the saved calibration.json / table_height.json / recon cache; the only new computations
are projections (deterministic) and the per-frame silhouette agreement curves.
usage: diag_reconstruction.py pringles|battery"""
import json, sys
import cv2, numpy as np, torch
from _common import GUIDE, TASKS, Run
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen import mesh

task = sys.argv[1]; T_ = TASKS[task]; E = T_["source_episode"]; OUT = GUIDE / "stage_03_reconstruction"; OUT.mkdir(exist_ok=True)
W = T_["work"]; COL = {"R": (235, 104, 52), "L": (42, 120, 214)}
with Run(f"recon_{task}", inputs=[T_["config"], W / "calibration.json", W / "table_height.json", W / f"cache/recon_ep{E}.npz"]) as run:
    ctx = Context(T_["config"]); rec = Recon(ctx, E, "cuda:0"); cal = json.loads((W / "calibration.json").read_text())
    th = json.loads((W / "table_height.json").read_text()); info = json.loads(str(np.load(W / f"cache/recon_ep{E}.npz")["info"]))
    G = mesh.Geometry(ctx.robot, device="cuda:0", spacing=0.003)

    def sil(cam, j, lag_frames=None):
        ts = j + (rec.lag if lag_frames is None else lag_frames)
        from sfagen.recon import interp_rows
        q = {s: interp_rows(rec.sp[s], np.array([ts]))[0] for s in rec.arms}; g = {s: float(interp_rows(rec.sp["g" + s][:, None], np.array([ts]))[0, 0]) for s in rec.arms}
        _, lab, _ = mesh.splat(G.world(G.placements(q, g)), G.gid, cam, dev="cuda:0"); lab = lab.cpu().numpy()
        side = np.array(G.side + ["X"])[lab]
        return cv2.morphologyEx(np.isin(side, rec.arms).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0

    # ---------- A/B: initial vs calibrated silhouette on a near-static frame (same rule as calibrate.load_frames)
    med = np.median(rec.F[::3], axis=0).astype(np.int16)
    qd = np.abs(np.gradient(np.concatenate([rec.sp[s] for s in rec.arms], 1), axis=0)).max(1) * ctx.src.fps
    cand = np.flatnonzero(qd < 0.05); cand = cand[(cand > 5) & (cand < rec.T - 5)]; j0 = int(cand[len(cand) // 2]) if len(cand) else 0
    fg = cv2.morphologyEx((np.abs(rec.F[j0].astype(np.int16) - med).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    def frame_iou(m):
        roi = cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0; a = fg & roi
        return float((a & m).sum() / max(1, (a | m).sum()))
    s0, s1 = sil(ctx.cam0, j0, 0.0), sil(ctx.cam, j0, 0.0)          # calibration uses state time = video time (static frames)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, m, c, title in ((axs[0], s0, (255, 60, 60), f"A  INITIAL camera (config guess)\nframe {j0}: IoU {frame_iou(s0):.3f}"),
                            (axs[1], s1, (60, 255, 60), f"B  CALIBRATED camera\nframe {j0}: IoU {frame_iou(s1):.3f}")):
        v = rec.F[j0].copy(); cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(v, cnt, -1, c, 2)
        ax.imshow(v); ax.set_title(title, fontsize=10); ax.axis("off")
    v = (rec.F[j0] * 0.5).astype(np.uint8); v[fg] = [255, 255, 255]; axs[2].imshow(v); axs[2].axis("off")
    axs[2].set_title("calibration target: camera-independent foreground\n|frame - temporal median| > 30 (white)", fontsize=10)
    p = np.array(cal["params7"]); ang = np.degrees(np.linalg.norm(p[:3]))
    fig.suptitle(f"{task}: render-and-compare camera calibration (Powell on 7 parameters). Correction: rotation {ang:.2f}°, translation "
                 f"{1000 * np.linalg.norm(p[3:6]):.1f} mm, focal x{np.exp(p[6]):.4f}. IoU is a silhouette score, NOT metric accuracy.", fontweight="bold", fontsize=10)
    fig.savefig(run.out(OUT / f"{task}_calibration_before_after.png"), dpi=75); plt.close(fig)
    run.note(calibration_json=cal, calibration_correction={"rotation_deg": float(ang), "translation_mm": float(1000 * np.linalg.norm(p[3:6])), "focal_scale": float(np.exp(p[6]))},
             example_frame=j0, example_iou={"initial": frame_iou(s0), "calibrated": frame_iou(s1)})

    # ---------- D: FK skeleton + TCP axes overlay
    def skeleton(j):
        q = {s: rec.gv[s][j] for s in rec.arms}; g = {s: float(rec.grip_v[s][j]) for s in rec.grip_v}; ctx.robot.fk(q, g)
        out = {}
        for s in rec.arms:
            jn = ctx.cfg["robot"]["arms"][s]["joints"]; P = [ctx.robot.d.oMi[ctx.robot.m.getJointId(n)].translation.copy() for n in jn]
            T = ctx.robot.frame(ctx.robot.tcp[s]); P.append(T[:3, 3]); out[s] = (np.array(P), T)
        return out
    def draw_skel(img, j):
        for s, (P, T) in skeleton(j).items():
            uv, z = rec.cam.project(P); uv = uv.astype(int)
            for a, b in zip(uv[:-1], uv[1:]):
                cv2.line(img, tuple(a), tuple(b), COL[s], 3)
            for u in uv:
                cv2.circle(img, tuple(u), 5, (255, 255, 255), -1)
            o, _ = rec.cam.project(T[:3, 3][None]); o = o[0].astype(int)
            for k, c in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                e, _ = rec.cam.project((T[:3, 3] + 0.06 * T[:3, k])[None]); cv2.arrowedLine(img, tuple(o), tuple(e[0].astype(int)), c, 2, tipLength=0.25)
        return img
    pick = np.linspace(10, rec.T - 10, 4).astype(int)
    fig, axs = plt.subplots(1, 4, figsize=(22, 4.6), constrained_layout=True)
    for ax, j in zip(axs, pick):
        ax.imshow(draw_skel(rec.F[j].copy(), j)); ax.axis("off"); ax.set_title(f"frame {j} (state time {j + rec.lag:+.2f})", fontsize=9)
    fig.suptitle(f"D  {task}: Pinocchio forward kinematics of the RECORDED joints projected with the calibrated camera "
                 "(joint origins = dots; TCP axes x red, y green, z blue, 6 cm)", fontweight="bold", fontsize=10)
    fig.savefig(run.out(OUT / f"{task}_fk_overlay.png"), dpi=75); plt.close(fig)

    # ---------- E: 3D frames diagram + animation
    def scene(ax, j, title=True):
        ax.cla(); zt = rec.z_table
        xs, ys = np.meshgrid(np.linspace(-0.05, 0.55, 2), np.linspace(-0.35, 0.35, 2)); ax.plot_surface(xs, ys, np.full_like(xs, zt), alpha=0.15, color="grey")
        for k, c in enumerate("rgb"):
            ax.quiver(0, 0, 0, *(0.1 * np.eye(3)[k]), color=c, lw=2)
        ax.text(0, 0, -0.03, "world (URDF base)", fontsize=7)
        C = rec.cam; ax.scatter(*C.t, color="k", s=30); ax.text(*C.t, " body camera", fontsize=7)
        for k, c in enumerate("rgb"):
            ax.quiver(*C.t, *(0.08 * C.R[:, k]), color=c, lw=1.5)
        Kinv = np.linalg.inv(C.K); corners = np.array([[0, 0, 1], [C.W, 0, 1], [C.W, C.H, 1], [0, C.H, 1]]) @ Kinv.T
        for cc in corners:
            d = C.R @ cc; lam = (zt - C.t[2]) / d[2]; P = C.t + lam * d; ax.plot(*np.stack([C.t, P]).T, color="k", lw=0.5, alpha=0.5)
        for s, (P, T) in skeleton(j).items():
            ax.plot(*P.T, "-o", color=np.array(COL[s]) / 255, ms=3, lw=2)
            for k, c in enumerate("rgb"):
                ax.quiver(*T[:3, 3], *(0.05 * T[:3, k]), color=c, lw=1)
        for k, o in enumerate(rec.objects):
            M = rec.obj_pose(k, j + rec.lag); P = (M[:3, :3] @ o["points"][::40].T).T + M[:3, 3]
            ax.scatter(*P.T, s=1, color="green"); ax.text(*M[:3, 3], f" {o['name']} ({rec.holder_at(k, j + rec.lag)})", fontsize=7)
        ax.set_xlim(-0.1, 0.6); ax.set_ylim(-0.4, 0.4); ax.set_zlim(0, 0.9); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=-140)
        if title:
            ax.set_title(f"{task} ep {E}, video frame {j}", fontsize=9)
    fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    for i, j in enumerate(pick[[0, 1, 2]]):
        scene(fig.add_subplot(1, 3, i + 1, projection="3d"), int(j))
    fig.suptitle(f"E  {task}: world / table / camera / TCP / object frames (metres; URDF base frame; table z = {rec.z_table:.3f} m)", fontweight="bold")
    fig.savefig(run.out(OUT / f"{task}_3d_frames.png"), dpi=80); plt.close(fig)
    fig = plt.figure(figsize=(10, 5)); ax3 = fig.add_subplot(1, 2, 1, projection="3d"); ax2 = fig.add_subplot(1, 2, 2)
    vw = cv2.VideoWriter(str(OUT / f"{task}_3d_fk_animation.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1000, 500))
    for j in range(0, rec.T, 3):
        scene(ax3, j); ax2.cla(); ax2.imshow(draw_skel(rec.F[j].copy(), j)); ax2.axis("off"); ax2.set_title("FK projected into body_cam", fontsize=9)
        fig.canvas.draw(); im = np.asarray(fig.canvas.buffer_rgba())[..., :3]; vw.write(cv2.cvtColor(cv2.resize(im, (1000, 500)), cv2.COLOR_RGB2BGR))
    vw.release(); plt.close(fig); run.out(OUT / f"{task}_3d_fk_animation.mp4")

    # ---------- F: object position over time + silhouette residual; lag scans; table fits
    t = np.arange(rec.T); fig, axs = plt.subplots(2, 2, figsize=(16, 8.5), constrained_layout=True)
    obj_res = {}
    for k, o in enumerate(rec.objects):
        P = np.stack([rec.obj_pose(k, j + rec.lag)[:3, 3] for j in t])
        for d, ls in zip(range(3), ("-", "--", ":")):
            axs[0, 0].plot(t, P[:, d], ls, label=f"{o['name']} {'xyz'[d]}")
        r = []
        for j in t:                       # proxy silhouette vs plate-difference foreground minus SAM2 robot, within 60 px of the proxy
            ps = rec.obj_sil(k, rec.obj_pose(k, j + rec.lag)); roi = cv2.dilate(ps.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
            f = rec._fg(j, rec.plate) & ~np.any([rec.masks[s][j] > 0 for s in rec.arms], 0) & roi
            r.append(float((f & ps).sum() / max(1, (f | ps).sum())))
        axs[0, 1].plot(t, r, label=o["name"]); obj_res[o["name"]] = {"median_iou": float(np.median(r)), "p10": float(np.percentile(r, 10))}
        for (h, a, b) in o["holder"]:
            axs[0, 0].axvspan(a, min(b, rec.T), alpha=0.06, color={"table": "grey", "R": "orange", "L": "blue", "placed": "green"}.get(h, "k"))
            axs[0, 0].text(a, axs[0, 0].get_ylim()[1] if False else P[:, 2].max() + 0.02, f"{o['name']}:{h}", fontsize=7)
    for s in rec.ev:
        for kind, c in (("closes", "k"), ("opens", "r")):
            for e in rec.ev[s][kind]:
                axs[0, 0].axvline(e - rec.lag, color=c, lw=0.6, ls=":")
    axs[0, 0].set_title("object position (world, m) over video frames; shading = holder (table / R hand / L hand / placed);\ndotted = gripper close (black) / open (red)", fontsize=9)
    axs[0, 0].legend(fontsize=7, ncol=3); axs[0, 0].set_xlabel("video frame")
    axs[0, 1].set_title("object proxy silhouette vs plate-difference foreground (robot removed)\nIoU per frame: a render-and-compare RESIDUAL, not accuracy", fontsize=9)
    axs[0, 1].set_ylim(0, 1); axs[0, 1].legend(); axs[0, 1].set_xlabel("video frame")
    sc = {float(k): v for k, v in info["lag_scan"].items()}; ph = {float(k): v for k, v in info["lag_photometric"].items()}
    a1 = axs[1, 0]; a1.plot(list(sc), list(sc.values()), "o-", color="grey", label="silhouette IoU scan (rejected: flat)")
    a2 = a1.twinx(); a2.plot(list(ph), list(ph.values()), "o-", color="#eb6834", label="photometric re-render PSNR (dB)")
    a2.axvline(rec.lag, color="#eb6834", ls="--"); a1.set_xlabel("video/state lag (frames; state time = video frame + lag)"); a1.set_ylabel("IoU"); a2.set_ylabel("PSNR dB")
    a1.set_title(f"video/state synchronisation: chosen lag {rec.lag:+.2f} frames (photometric)", fontsize=9); a1.legend(loc="lower left", fontsize=8); a2.legend(loc="lower right", fontsize=8)
    fits = th["fits"]; axs[1, 1].scatter([f["episode"] for f in fits], [f["z"] for f in fits], c=[f["iou"] for f in fits], cmap="viridis", vmin=0, vmax=1, s=60)
    for f in fits:
        axs[1, 1].text(f["episode"], f["z"] + 0.003, f"{f['iou']:.2f}", fontsize=7, ha="center")
    axs[1, 1].axhline(th["z"], color="k", ls="--", label=f"adopted table z = {th['z']:.4f} m"); axs[1, 1].axhline(th["lowest_arm_z"], color="r", ls=":", label=f"lowest arm point {th['lowest_arm_z']:.4f} m")
    axs[1, 1].set_title(f"table height per source episode (colour/number = silhouette IoU)\nrule: {th['rule']}", fontsize=9); axs[1, 1].legend(fontsize=8); axs[1, 1].set_xlabel("source episode"); axs[1, 1].set_ylabel("z (m)")
    fig.suptitle(f"F  {task} source ep {E}: object trajectory, residuals, synchronisation and table height", fontweight="bold")
    fig.savefig(run.out(OUT / f"{task}_object_lag_table.png"), dpi=80); plt.close(fig)
    run.note(object_silhouette_residual=obj_res, lag_final=rec.lag, lag_scan_range=[min(sc), max(sc), max(sc.values()) - min(sc.values())],
             photometric_range_db=max(ph.values()) - min(ph.values()), table=th["z"], table_rule=th["rule"], objects_info=info["objects"],
             holders={o["name"]: o["holder"] for o in rec.objects}, attachments={o["name"]: [np.round(x, 4).tolist() for x in o["X"]] for o in rec.objects})
    print(json.dumps({k: v for k, v in run.notes.items() if k not in ("calibration_json", "attachments")}, indent=1, default=str)[:2500])
