"""Stage 4 (new trajectories: F1/F2 transformation + IK) for the representative published pair of one task.
Inputs: the saved candidate (configs/work/<task>/candidates/ep<E>/<key>.npz + checks.json) and the PUBLISHED episode
(local copy of the built dataset, Hub v3.0 files have identical hashes). New diagnostic computation: one re-plan + re-check
of the same candidate (sfagen.plan / sfagen.checks, unchanged) to (a) time it and (b) confirm bit-identical reproduction.
usage: diag_trajectories.py pringles|battery"""
import json, sys, time, glob
import cv2, numpy as np, pandas as pd
from _common import GUIDE, TASKS, Run
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen import plan
from sfagen.checks import Checker

task = sys.argv[1]; T_ = TASKS[task]; E, key, X = T_["source_episode"], T_["variation"], T_["export_episode"]
OUT = GUIDE / "stage_04_trajectories"; OUT.mkdir(exist_ok=True); W = T_["work"]
npz = W / f"candidates/ep{E}/{key}.npz"; ckp = W / f"candidates/ep{E}/checks.json"
with Run(f"traj_{task}", inputs=[T_["config"], npz, ckp]) as run:
    ctx = Context(T_["config"]); rec = Recon(ctx, E, "cuda:0"); rob = ctx.robot
    z = np.load(npz); ck = json.loads(ckp.read_text())[key]; S0, A0 = rec.state, rec.action
    Sn, An = z["state"].astype(np.float64), z["action"].astype(np.float64); w = z["w"]; ik_err = z["ik_err"]
    # published episode == candidate?
    df = pd.read_parquet(sorted(glob.glob(str(W / "dataset/data/*/*.parquet")))); d = df[df.episode_index == X].sort_values("frame_index")
    Sp, Ap = np.stack(d["observation.state"].to_numpy()), np.stack(d["action"].to_numpy())
    pub_ok = bool(np.array_equal(Sp, z["state"]) and np.array_equal(Ap, z["action"]))
    # ---- re-plan + re-check (timed), compare with saved
    fam = ck["family"]; d_ = np.array(ck["shift_m"]); il = ctx.cfg["variations"].get("interaction_location", {})
    t0 = time.time()
    c = plan.f1_plan(rec, [o["name"] for o in rec.objects].index(ck["object"]), d_) if fam == "object_position" else plan.f2_plan(rec, d_, il.get("d_interact_m", 0.2), il.get("ramp_frames", 45))
    t_plan = time.time() - t0
    chk = Checker(ctx); t0 = time.time(); chk.envelope(ctx.episodes()); t_env = time.time() - t0
    t0 = time.time(); ck2 = chk.check(rec, c); t_check = time.time() - t0
    same = bool(np.array_equal(c["state"].astype(np.float32), z["state"]) and np.array_equal(c["action"].astype(np.float32), z["action"]))
    keys_cmp = ["ik_pos_err_mm_max", "ik_rot_err_deg_max", "speed_ratio_to_real_max", "accel_ratio_to_real_max", "min_mesh_distance_mm",
                "source_min_mesh_distance_same_frames_mm", "min_hand_distance_mm", "source_min_hand_distance_same_frames_mm", "lowest_arm_z_m", "pass"]
    run.note(published_episode=X, published_equals_candidate=pub_ok, replan_bit_identical=same, replan_s=round(t_plan, 2), envelope_s=round(t_env, 2), check_s=round(t_check, 2),
             checks_saved={k: ck.get(k) for k in keys_cmp}, checks_rerun={k: ck2.get(k) for k in keys_cmp}, marks=ck["marks"], lead_frames=ck["lead"],
             placement=ck.get("placement"), moved_arms=c["moved_arms"], shift_m=ck["shift_m"], family=fam)
    # ---- trajectories
    arms = c["moved_arms"]; t = np.arange(rec.T)
    P0 = {s: rob.tcp_from_state(S0, s) for s in rob.arms}; Pn = {s: rob.tcp_from_state(Sn, s) for s in rob.arms}
    des = {s: P0[s][:, :3, 3] + w[:, None] * d_[None] for s in arms}
    ev = rob.gripper_events(A0); m = ck["marks"]
    def marks(ax):
        for s in ev:
            for e in ev[s]["closes"]:
                ax.axvline(e, color="k", ls=":", lw=0.8)
            for e in ev[s]["opens"]:
                ax.axvline(e, color="r", ls=":", lw=0.8)
        full = np.flatnonzero(w >= 0.999)
        if len(full):
            ax.axvspan(full[0], full[-1], color="gold", alpha=0.12)
    fig, axs = plt.subplots(len(arms) * 3 + 2, 1, figsize=(13, 2.0 * (len(arms) * 3 + 2)), sharex=True, constrained_layout=True)
    i = 0
    for s in arms:
        for k in range(3):
            ax = axs[i]; i += 1
            ax.plot(t, P0[s][:, k, 3], color="grey", lw=2, label="recorded TCP")
            ax.plot(t, des[s][:, k], color="#2a78d6", ls="--", lw=1.5, label="desired = recorded + w(t)·d")
            ax.plot(t, Pn[s][:, k, 3], color="#eb6834", lw=1, label="IK solution (FK of synthetic state)")
            ax.set_ylabel(f"{s} TCP {'xyz'[k]} (m)"); marks(ax)
            if i == 1:
                ax.legend(fontsize=7, ncol=3, loc="upper right")
    axs[i].plot(t, w, color="k"); axs[i].set_ylabel("w(t)"); marks(axs[i]); i += 1
    axs[i].semilogy(t, np.maximum(1000 * ik_err[:, 0], 1e-4), label="position residual (mm)"); axs[i].semilogy(t, np.maximum(ik_err[:, 1], 1e-4), label="rotation residual (deg)")
    for y, c_, l in ((1.0, "g", "contact limit 1 mm"), (0.5, "g", None), (5.0, "r", "blend limit 5 mm"), (2.0, "r", None)):
        axs[i].axhline(y, color=c_, ls="--", lw=0.7, label=l)
    axs[i].legend(fontsize=7, ncol=4); axs[i].set_ylabel("|FK - desired|"); axs[i].set_xlabel("frame (state time); dotted black/red = gripper close/open command; gold = full offset (contact) frames"); marks(axs[i])
    fig.suptitle(f"{task}: source ep {E} -> published ep {X} '{key}' ({fam}, d = {np.round(100 * d_, 1).tolist()} cm)", fontweight="bold")
    fig.savefig(run.out(OUT / f"{task}_tcp_desired_vs_ik.png"), dpi=80); plt.close(fig)
    # ---- joints and actions
    for name, A, B in (("joints", S0, Sn), ("actions", A0, An)):
        fig, axs = plt.subplots(len(arms), 8, figsize=(26, 2.8 * len(arms)), sharex=True, constrained_layout=True, squeeze=False)
        for r_, s in enumerate(arms):
            idx = rob.sidx[s] + [rob.grip[s]["state_index"]]
            for k, j in enumerate(idx):
                ax = axs[r_, k]; ax.plot(t, A[:, j], color="grey", lw=2, label="real"); ax.plot(t, B[:, j], color="#eb6834", lw=1, label="synthetic")
                ax.set_title(f"{s} {'J' + str(k + 1) if k < 7 else 'gripper'} (rad / raw)", fontsize=8); marks(ax)
                if k == 0 and r_ == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f"{task}: {'q_real(t) vs q_synthetic(t)' if name == 'joints' else 'a_real(t) vs a_synthetic(t)  (a′ = a + Δq(t + lead), lead = ' + str(ck['lead']) + ' frames)'}"
                     f" — moved arm(s) {arms}; grippers are bit-identical", fontweight="bold")
        fig.savefig(run.out(OUT / f"{task}_{name}_real_vs_synthetic.png"), dpi=70); plt.close(fig)
    # ---- top-down XY
    fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)
    for s in rob.arms:
        ax.plot(P0[s][:, 0, 3], P0[s][:, 1, 3], color="grey", lw=2, label=f"{s} recorded" if s == rob.arms[0] else None)
        ax.plot(Pn[s][:, 0, 3], Pn[s][:, 1, 3], color="#eb6834" if s == "R" else "#2a78d6", lw=1, label=f"{s} synthetic")
    for k, o in enumerate(rec.objects):
        p0 = o["table"][:3, 3]; p1 = z["objects"][k][0][:3, 3]
        ax.scatter(*p0[:2], marker="s", s=80, facecolor="none", edgecolor="k", label=f"{o['name']} recorded rest")
        if not np.allclose(p0, p1):
            ax.scatter(*p1[:2], marker="s", s=80, color="green", label=f"{o['name']} new rest"); ax.annotate("", p1[:2], p0[:2], arrowprops=dict(arrowstyle="->", color="green"))
    if fam == "interaction_location":
        tm = int((m["t_window_start"] + m["t_window_end"]) / 2)
        for s in rob.arms:
            ax.scatter(*P0[s][tm, :2, 3], marker="o", s=70, facecolor="none", edgecolor="k"); ax.scatter(*Pn[s][tm, :2, 3], marker="o", s=70, color="purple")
        ax.annotate("interaction point moved", Pn["R"][tm, :2, 3], P0["R"][tm, :2, 3], arrowprops=dict(arrowstyle="->", color="purple"), fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"top view (world): recorded vs synthetic TCP paths\n{task} '{key}'", fontsize=10)
    fig.savefig(run.out(OUT / f"{task}_topdown_paths.png"), dpi=90); plt.close(fig)
    # ---- 3D animation: recorded (grey ghost) vs synthetic (colour) skeletons + TCP trails + objects
    def skel(S, j):
        sp = rob.split(S[j:j + 1]); rob.fk({s: sp[s][0] for s in rob.arms}, {s: sp["g" + s][0] for s in rob.arms}); out = {}
        for s in rob.arms:
            P = [rob.d.oMi[rob.m.getJointId(n)].translation.copy() for n in ctx.cfg["robot"]["arms"][s]["joints"]] + [rob.frame(rob.tcp[s])[:3, 3]]
            out[s] = np.array(P)
        return out
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(projection="3d")
    vw = cv2.VideoWriter(str(OUT / f"{task}_3d_recorded_vs_synthetic.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (800, 700))
    src_obj = [np.stack([rec.obj_pose(k, j) for j in range(rec.T)]) for k in range(len(rec.objects))]
    for j in range(0, rec.T, 3):
        ax.cla(); a, b = skel(S0, j), skel(Sn, j)
        for s in rob.arms:
            ax.plot(*a[s].T, "-o", color="0.6", ms=2, lw=3, alpha=0.6); ax.plot(*b[s].T, "-o", color="#eb6834" if s == "R" else "#2a78d6", ms=2, lw=1.5)
            ax.plot(*P0[s][max(0, j - 30):j + 1, :3, 3].T, color="0.5", lw=1); ax.plot(*Pn[s][max(0, j - 30):j + 1, :3, 3].T, color="purple", lw=1)
        for k, o in enumerate(rec.objects):
            for M, cc in ((src_obj[k][j], "0.5"), (z["objects"][k][j], "green")):
                P = (M[:3, :3] @ o["points"][::60].T).T + M[:3, 3]; ax.scatter(*P.T, s=1, color=cc)
        ax.set_xlim(0.0, 0.5); ax.set_ylim(-0.3, 0.3); ax.set_zlim(0.15, 0.65); ax.view_init(elev=28, azim=-150)
        ax.set_title(f"{task} '{key}'  frame {j}  w={w[j]:.2f}\ngrey = recorded, colour = synthetic, green = object (synthetic)", fontsize=9)
        fig.canvas.draw(); im = np.asarray(fig.canvas.buffer_rgba())[..., :3]; vw.write(cv2.cvtColor(cv2.resize(im, (800, 700)), cv2.COLOR_RGB2BGR))
    vw.release(); plt.close(fig); run.out(OUT / f"{task}_3d_recorded_vs_synthetic.mp4")
    # ---- whole-dataset IK/check statistics from saved checks.json (all candidates of all sources)
    allc = []
    for e in ctx.episodes():
        for k_, v in json.loads((W / f"candidates/ep{e}/checks.json").read_text()).items():
            allc.append({"e": e, "key": k_, **{kk: v.get(kk) for kk in ("pass", "family", "ik_pos_err_mm_max", "ik_rot_err_deg_max", "ik_pos_err_mm_max_blend", "speed_ratio_to_real_max", "accel_ratio_to_real_max", "min_mesh_distance_mm", "source_min_mesh_distance_same_frames_mm")}, "error": v.get("error")})
    import collections; from sfagen.why import reasons
    rs = collections.Counter()
    for e in ctx.episodes():
        for k_, v in json.loads((W / f"candidates/ep{e}/checks.json").read_text()).items():
            if not v.get("pass"):
                for r_ in (reasons(v, ctx.cfg.get("checks", {}), 0) if "error" not in v else ["no_schedule"]):
                    rs[r_] += 1
    dfc = pd.DataFrame(allc)
    run.note(all_candidates={"total": len(dfc), "passing": int(dfc["pass"].sum()), "by_family": dfc.groupby("family")["pass"].agg(["sum", "count"]).to_dict(),
                             "failure_reason_counts": dict(rs),
                             "ik_pos_err_mm_passing": {"median": float(dfc[dfc["pass"] == True].ik_pos_err_mm_max.median()), "max": float(dfc[dfc["pass"] == True].ik_pos_err_mm_max.max())},
                             "source_clearance_zero_mm_candidates": int((dfc.source_min_mesh_distance_same_frames_mm.fillna(1) <= 0).sum())})
    print(json.dumps(run.notes, indent=1, default=str)[:4000])
