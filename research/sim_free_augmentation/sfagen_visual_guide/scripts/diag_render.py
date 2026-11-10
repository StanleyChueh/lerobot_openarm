"""Stage 5 (mesh-guided forward warping) intermediate panels A-H, failure frames, annotated comparison video and the
three-camera view for the representative published pair of one task. Diagnostic only.

`render_capture` is a line-by-line copy of sfagen.render.Renderer.render (commit 50832b3) that additionally RETURNS its
intermediate images; its final output is compared with the published rendered frame (configs/work/<task>/body/*.npy)
so the panels are shown to belong to the real computation. sfagen itself is not modified.
usage: diag_render.py pringles|battery"""
import glob, json, sys
import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as Fnn
from _common import GUIDE, TASKS, Run
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen.render import Renderer, candidate_frames, BORROW, _nearest
from sfagen.generate import load_candidate

task = sys.argv[1]; T_ = TASKS[task]; E, key, X = T_["source_episode"], T_["variation"], T_["export_episode"]; W = T_["work"]
OUT = GUIDE / "stage_05_rendering"; OUT.mkdir(exist_ok=True); OUT6 = GUIDE / "stage_06_cameras"; OUT6.mkdir(exist_ok=True)


@torch.no_grad()
def render_capture(self, j, q, g, objs, exclude=None):
    cap = {}
    H, W_, dev = self.H, self.W, self.dev
    S = self.source_state(j); Mt = self.placements(q, g, objs)
    Mtt = torch.tensor(Mt, dtype=torch.float32, device=dev); Mst = torch.tensor(S["Ms"], dtype=torch.float32, device=dev)
    rel = Mtt @ torch.linalg.inv(Mst)
    sub = torch.arange(0, len(self.local), 50, device=dev)
    uo, _ = self.project(self.world(S["Ms"], sub)); un, _ = self.project(self.world(Mt, sub))
    disp = torch.zeros(len(Mt), device=dev).scatter_reduce(0, self.gid[sub], (un - uo).norm(dim=1), reduce="amax")
    moved = (disp >= 0.5) & self.movable
    part = S["part"]; img_s = self.F[j].to(dev).float(); out = img_s.clone()
    mv_all = (part >= 0) & moved[part.clamp(min=0)]; static = (part >= 0) & ~moved[part.clamp(min=0)]
    rm = torch.tensor(cv2.dilate(mv_all.cpu().numpy().astype(np.uint8), np.ones((7, 7), np.uint8)) > 0, device=dev) & ~static
    plate_used = self.adapted_plate(img_s, (part >= 0) | rm) if self.cfg["plate_adapt"] else self.plate
    out[rm] = plate_used[rm]
    cap.update(cat=S["cat"].cpu().numpy(), moved_mask=mv_all.cpu().numpy(), static_mask=static.cpu().numpy(), removed=rm.cpu().numpy(),
               plate_adapted=plate_used.clamp(0, 255).byte().cpu().numpy(), after_removal=out.clamp(0, 255).byte().cpu().numpy())
    zb = torch.where(static, S["sdepth"], torch.full_like(S["sdepth"], float("inf")))
    ys, xs = mv_all.nonzero(as_tuple=True); info = {"moved_parts": int(moved.sum()), "warped_px": int(len(ys))}
    hit = torch.zeros((H, W_), dtype=torch.bool, device=dev); warp_only = torch.zeros((H, W_, 3), device=dev)
    if len(ys):
        Rm = rel[part[ys, xs]]; Pn = torch.einsum("nij,nj->ni", Rm[:, :3, :3], S["P"][ys, xs]) + Rm[:, :3, 3]
        uv, z = self.project(Pn); col = img_s[ys, xs]; cp, cz, cc = [], [], []
        for du in (0, 1):
            for dv in (0, 1):
                uu = torch.floor(uv[:, 0]).long() + du; vv = torch.floor(uv[:, 1]).long() + dv
                ok = (uu >= 0) & (uu < W_) & (vv >= 0) & (vv < H) & (z > 0.05)
                cp.append((vv * W_ + uu)[ok]); cz.append(z[ok]); cc.append(col[ok])
        pix, zz, cc = torch.cat(cp), torch.cat(cz), torch.cat(cc)
        front = torch.full((H * W_,), float("inf"), device=dev); front.scatter_reduce_(0, pix, zz, reduce="amin")
        vis = (zz <= front[pix] + 1e-6) & (zz < zb.view(-1)[pix] - 0.005)
        acc = torch.zeros((H * W_, 3), device=dev); cnt = torch.zeros(H * W_, device=dev)
        acc.index_add_(0, pix[vis], cc[vis]); cnt.index_add_(0, pix[vis], torch.ones_like(zz[vis]))
        hit0 = (cnt.view(1, 1, H, W_) > 0).float(); k5 = torch.ones(1, 1, 5, 5, device=dev)
        region = -Fnn.max_pool2d(-Fnn.max_pool2d(hit0, 5, 1, 2), 5, 1, 2)
        avg = Fnn.conv2d(acc.T.reshape(3, 1, H, W_), k5, padding=2).view(3, -1).T; wsum = Fnn.conv2d(cnt.view(1, 1, H, W_), k5, padding=2).view(-1)
        fm = ((region > 0) & (hit0 == 0)).view(-1) & (wsum > 0); h = cnt > 0
        out.view(-1, 3)[h] = acc[h] / cnt[h, None]; out.view(-1, 3)[fm] = avg[fm] / wsum[fm, None]
        warp_only.view(-1, 3)[h] = acc[h] / cnt[h, None]; warp_only.view(-1, 3)[fm] = avg[fm] / wsum[fm, None]
        hit = (h | fm).view(H, W_)
        cap.update(occluded_by_static=int((~vis).sum()), pinholes_closed=int(fm.sum()))
    cap.update(warped=warp_only.clamp(0, 255).byte().cpu().numpy(), hit=hit.cpu().numpy(), after_warp=out.clamp(0, 255).byte().cpu().numpy())
    dt, lt, pt = self.zbuf(Mt)
    tmask = (lt >= 0) & moved[lt.clamp(min=0)] & (dt < zb - 0.005)
    tm = cv2.erode(tmask.cpu().numpy().astype(np.uint8), np.ones((self.cfg["tm_erode"],) * 2, np.uint8)) > 0
    hn = tm & ~hit.cpu().numpy(); hitn = hit.cpu().numpy()
    small = hn & (cv2.dilate(hitn.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (cv2.blur(hitn.astype(np.float32), (5, 5)) > 0.6)
    big = torch.tensor(hn & ~small, device=dev); pixb = big.nonzero(); filled_n = 0
    cap.update(target_sil=tmask.cpu().numpy(), target_sil_eroded=tm, holes_new_surface=hn.copy(), holes_small=small.copy())
    borrowed_mask = np.zeros((H, W_), bool)
    if len(pixb):
        pid = pt[pixb[:, 0], pixb[:, 1]]; labs = lt[pixb[:, 0], pixb[:, 1]]; loc = self.local[pid]
        col = torch.zeros((len(pixb), 3), device=dev); filled = torch.zeros(len(pixb), dtype=torch.bool, device=dev)
        for d in [0] + BORROW:
            if filled.all():
                break
            jj = j + d
            if jj < 0 or jj >= self.rec.T or (exclude is not None and exclude[0] <= jj <= exclude[1]):
                continue
            ri = (~filled).nonzero()[:, 0]
            fn = lambda Ms, ri=ri: torch.einsum("nij,nj->ni", torch.tensor(Ms, dtype=torch.float32, device=dev)[labs[ri]][:, :3, :3], loc[ri]) + torch.tensor(Ms, dtype=torch.float32, device=dev)[labs[ri]][:, :3, 3]  # noqa: E731
            rgb, v = self.lookup(jj, fn, labs[ri]); col[ri[v]] = rgb[v]; filled[ri[v]] = True
        out[pixb[filled][:, 0], pixb[filled][:, 1]] = col[filled]; filled_n = int(filled.sum()); big[pixb[filled][:, 0], pixb[filled][:, 1]] = False
        pb = pixb[filled].cpu().numpy(); borrowed_mask[pb[:, 0], pb[:, 1]] = True
    info["borrowed_px"] = filled_n
    img = out.clamp(0, 255).byte().cpu().numpy(); rest = small | big.cpu().numpy()
    cap.update(borrowed=borrowed_mask, after_borrow=img.copy(), telea_mask=rest.copy())
    if self.cfg["part_fill"]:
        raise NotImplementedError("published config has part_fill: false")
    if rest.any():
        img = cv2.inpaint(img, rest.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    cap["after_telea"] = img.copy()
    info["inpainted_px"] = int(rest.sum()); wm = (hitn | rest).astype(np.uint8)
    if self.cfg["despeckle"]:
        reg = cv2.dilate(wm, np.ones((5, 5), np.uint8)) > 0; med = cv2.medianBlur(img, 3)
        sel = reg & (np.abs(img.astype(np.int16) - med.astype(np.int16)).max(2) > 40); img[sel] = med[sel]; cap["despeckled_px"] = int(sel.sum())
    edge = (cv2.dilate(wm, np.ones((3, 3), np.uint8)) - cv2.erode(wm, np.ones((3, 3), np.uint8))) > 0
    bl = cv2.GaussianBlur(img, (3, 3), 0.7); img[edge] = bl[edge]; cap["antialiased_px"] = int(edge.sum())
    return img, info, cap


def tint(F, m, c, a=0.6):
    v = (F * 0.45).astype(np.uint8); v[m] = ((1 - a) * F[m] + a * np.array(c)).astype(np.uint8); return v


with Run(f"render_{task}", inputs=[T_["config"], W / f"body/{E}_{key}.npy", W / f"candidates/ep{E}/{key}.npz"]) as run:
    ctx = Context(T_["config"]); rec = Recon(ctx, E, "cuda:0"); R = Renderer(rec, "cuda:0", ctx.cfg.get("render", {}))
    c = load_candidate(ctx, rec, key); frames = candidate_frames(rec, c); pub = np.load(W / f"body/{E}_{key}.npy", mmap_mode="r")
    pinfo = json.loads((W / f"body/{E}_{key}.json").read_text())
    rendered = [i for i, x in enumerate(pinfo) if not x.get("copied")]
    pick = {"pringles": [130, 180, 235], "battery": [150, 262, 290]}[task]
    caps, eq = {}, {}
    for i in pick:
        _, q, g, objs = frames[i]; img, inf, cap = render_capture(R, i, q, g, objs); caps[i] = (img, inf, cap)
        d = np.abs(img.astype(np.int16) - np.asarray(pub[i]).astype(np.int16)).max(2)
        eq[i] = {"identical_px_frac": float((d == 0).mean()), "max_abs_diff": int(d.max()), "info": inf, "published_info": pinfo[i]}
    run.note(replica_vs_published=eq, rendered_frames=len(rendered), copied_frames=len(pinfo) - len(rendered))
    cats = R.cats
    for i in pick:
        img, inf, cap = caps[i]; F = rec.F[i]
        catv = (F * 0.45).astype(np.uint8)
        for k_, cn in enumerate(cats):
            col = {"R": (235, 104, 52), "L": (42, 120, 214)}.get(cn, (60, 200, 75)); m = cap["cat"] == k_
            catv[m] = (0.4 * F[m] + 0.6 * np.array(col)).astype(np.uint8)
        for k_, o in enumerate(rec.objects):
            cnt, _ = cv2.findContours(rec.obj_sil(k_, rec.obj_pose(k_, i + rec.lag)).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(catv, cnt, -1, (255, 255, 255), 1)
        holes = np.zeros_like(F); holes[cap["hit"]] = (60, 60, 60); holes[cap["holes_new_surface"]] = (255, 200, 0); holes[cap["borrowed"]] = (0, 200, 255); holes[cap["telea_mask"]] = (255, 0, 255)
        panels = [("A  original real frame", F),
                  ("B  robust masks (SAM2 arms + proxy/plate fg) + object proxy outline", catv),
                  ("C  clean plate relit to this frame (log-ratio, sigma 10)", cap["plate_adapted"]),
                  (f"D  moved parts removed -> plate ({int(cap['removed'].sum())} px)", cap["after_removal"]),
                  (f"E  3-D forward-warped real pixels ({inf['warped_px']} px warped)", cap["warped"]),
                  (f"F  holes: yellow=new surface, cyan=borrowed ({inf['borrowed_px']} px), magenta=Telea ({inf['inpainted_px']} px)", holes),
                  ("G  after borrowing + Telea (before despeckle/anti-alias)", cap["after_telea"]),
                  (f"H  final synthetic frame (== published: {eq[i]['identical_px_frac'] * 100:.2f}% px identical)", img)]
        fig, axs = plt.subplots(2, 4, figsize=(22, 8.6), constrained_layout=True)
        for ax, (t_, im) in zip(axs.ravel(), panels):
            ax.imshow(im); ax.set_title(t_, fontsize=9); ax.axis("off")
        fig.suptitle(f"Stage 5 — mesh-guided forward warping, {task} '{key}' output frame {i} (source frame {i}, target state time {i + rec.lag:+.2f}, w={c['w'][min(rec.T - 1, int(round(i + rec.lag)))]:.2f})", fontweight="bold")
        fig.savefig(run.out(OUT / f"{task}_render_panels_frame{i}.png"), dpi=72); plt.close(fig)
    # ---- failure frames: largest Telea-inpainted area among published renders of this task
    rows = []
    for f in glob.glob(str(W / "body/*.json")):
        nm = f.split("/")[-1][:-5]; inf = json.loads(open(f).read())
        for i, x in enumerate(inf):
            if not x.get("copied"):
                rows.append((x.get("inpainted_px", 0), x.get("borrowed_px", 0), nm, i))
    rows.sort(reverse=True); seen, worst = set(), []
    for r_ in rows:
        if r_[2] not in seen:
            worst.append(r_); seen.add(r_[2])
        if len(worst) == 3:
            break
    ip = np.array([r_[0] for r_ in rows]); run.note(inpainted_px_over_all_rendered_frames={"frames": len(rows), "median": float(np.median(ip)), "p95": float(np.percentile(ip, 95)), "max": int(ip.max())},
                                                   worst_frames=[{"render": r_[2], "frame": r_[3], "inpainted_px": r_[0], "borrowed_px": r_[1]} for r_ in worst])
    fig, axs = plt.subplots(len(worst), 2, figsize=(12, 4.6 * len(worst)), constrained_layout=True)
    srcF = {}
    for r_, (ipx, bpx, nm, i) in enumerate(worst):
        e = int(nm.split("_")[0]); kk = nm.split("_", 1)[1]
        if e not in srcF:
            srcF[e] = ctx.src.decode(ctx.src.body, e)
        syn = np.load(W / f"body/{nm}.npy", mmap_mode="r")[i]
        axs[r_, 0].imshow(srcF[e][i]); axs[r_, 0].set_title(f"real source ep {e}, frame {i}", fontsize=9); axs[r_, 0].axis("off")
        axs[r_, 1].imshow(syn); axs[r_, 1].set_title(f"published synthetic '{kk}' frame {i}: {ipx} px Telea-inpainted, {bpx} px borrowed", fontsize=9); axs[r_, 1].axis("off")
    fig.suptitle(f"{task}: three published frames with the largest inpainted area (automatic selection; visual defects)", fontweight="bold")
    fig.savefig(run.out(OUT / f"{task}_failure_frames.png"), dpi=70); plt.close(fig)
    # ---- annotated comparison video for the pair: real | synthetic | |difference| with per-frame renderer stats
    step = 1 if rec.T < 400 else 2
    vw = cv2.VideoWriter(str(OUT / f"{task}_pair_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 480))
    for i in range(0, rec.T, step):
        a = rec.F[i].copy(); b = np.asarray(pub[i]).copy(); dd = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2).astype(np.uint8)
        hm = cv2.applyColorMap(np.clip(dd * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].copy()
        x = pinfo[i]
        cv2.putText(a, f"REAL source ep{E} t={i}", (8, 24), 0, 0.7, (255, 255, 0), 2)
        cv2.putText(b, f"SYNTHETIC ep{X} '{key}'", (8, 24), 0, 0.6, (255, 160, 60), 2)
        cv2.putText(b, "copied (unchanged)" if x.get("copied") else f"warped {x['warped_px']} borrowed {x['borrowed_px']} telea {x['inpainted_px']}", (8, 470), 0, 0.5, (255, 255, 255), 1)
        cv2.putText(hm, "|synthetic - real| x3", (8, 24), 0, 0.7, (255, 255, 255), 2)
        vw.write(cv2.cvtColor(np.concatenate([a, b, hm], 1), cv2.COLOR_RGB2BGR))
    vw.release(); run.out(OUT / f"{task}_pair_annotated.mp4")
    # ---- three-camera view from the PUBLISHED dataset (local copy; identical hashes on the Hub)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(T_["repo"], root=W / "dataset"); ep = ds.meta.episodes; s0 = int(ep["dataset_from_index"][X]); n = int(ep["length"][X])
    df = pd.read_parquet(sorted(glob.glob(str(W / "dataset/data/*/*.parquet")))); d = df[df.episode_index == X].sort_values("frame_index")
    val = np.stack(d["wrist_view_valid"].to_numpy()); off = np.stack(d["wrist_view_pose_offset"].to_numpy())
    wr = {w_: ctx.src.decode(w_, E) for w_ in ctx.src.wrists}
    tt = pick
    fig, axs = plt.subplots(len(tt), 6, figsize=(30, 4.0 * len(tt)), constrained_layout=True)
    for r_, i in enumerate(tt):
        x = ds[s0 + i]; im = lambda k_: (x[k_].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # noqa: E731
        for k_, (nm, a) in enumerate((("REAL body_cam", rec.F[i]), ("SYNTHETIC body_cam (published)", im("observation.images.body_cam")),
                                      ("REAL wrist_cam (left)", wr["observation.images.wrist_cam"][i]), ("published wrist_cam = REAL frame reused", im("observation.images.wrist_cam")),
                                      ("REAL right_wrist_cam", wr["observation.images.right_wrist_cam"][i]), ("published right_wrist_cam = REAL frame reused", im("observation.images.right_wrist_cam")))):
            axs[r_, k_].imshow(a); axs[r_, k_].axis("off")
            extra = f"\nwrist_view_valid={val[i].tolist()}, pose offset L {100 * off[i, 0]:.1f} cm / {np.degrees(off[i, 1]):.1f}°, R {100 * off[i, 2]:.1f} cm / {np.degrees(off[i, 3]):.1f}°" if k_ == 3 else ""
            axs[r_, k_].set_title(f"{nm}, frame {i}{extra}", fontsize=9)
    fig.suptitle(f"{task} published ep {X}: three cameras. Only body_cam is re-rendered; both wrist videos are the recorded real frames even where the synthetic arm pose differs.", fontweight="bold")
    fig.savefig(run.out(OUT6 / f"{task}_three_cameras.png"), dpi=60); plt.close(fig)
    wr_diff = {w_: float(np.mean([np.abs((ds[s0 + i][w_].permute(1, 2, 0).numpy() * 255) - wr[w_][i].astype(float)).mean() for i in range(0, n, 25)])) for w_ in ctx.src.wrists}
    fig, axs = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True, constrained_layout=True); t = np.arange(n)
    axs[0].plot(t, val[:, 0], label="wrist_view_valid[wrist_cam (L)]"); axs[0].plot(t, val[:, 1] + 0.02, label="wrist_view_valid[right_wrist_cam (R)] (+0.02 offset for visibility)")
    axs[0].set_ylim(-0.1, 1.15); axs[0].legend(fontsize=8); axs[0].set_title("1 = the whole scene equals the recording at this frame; 0 = the wrist image no longer matches the (synthetic) robot state", fontsize=9)
    axs[1].plot(t, 100 * off[:, 0], label="L wrist translation offset (cm)"); axs[1].plot(t, 100 * off[:, 2], label="R wrist translation offset (cm)")
    axs[1].plot(t, np.degrees(off[:, 1]), ls="--", label="L rotation offset (deg)"); axs[1].plot(t, np.degrees(off[:, 3]), ls="--", label="R rotation offset (deg)")
    axs[1].legend(fontsize=8); axs[1].set_xlabel("frame"); axs[1].set_title("wrist_view_pose_offset: how far each synthetic wrist camera is from where the reused real image was taken", fontsize=9)
    fig.suptitle(f"{task} published ep {X} '{key}': wrist-camera flags", fontweight="bold")
    fig.savefig(run.out(OUT6 / f"{task}_wrist_flags.png"), dpi=80); plt.close(fig)
    run.note(wrist_valid_fraction={w_: float(val[:, k_].mean()) for k_, w_ in enumerate(ctx.src.wrists)}, wrist_offset_max_cm={"L": float(100 * off[:, 0].max()), "R": float(100 * off[:, 2].max())},
             wrist_offset_max_deg={"L": float(np.degrees(off[:, 1].max())), "R": float(np.degrees(off[:, 3].max()))},
             published_wrist_vs_real_source_mean_abs_diff_8bit=wr_diff, frames_with_modified_state_but_reused_wrist=int((val[:, 0] == 0).sum()))
    print(json.dumps(run.notes, indent=1, default=str)[:3500])
