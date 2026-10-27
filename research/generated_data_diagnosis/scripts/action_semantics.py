"""Phase 2: action-label semantics of manual vs generated data, and of both policies' evaluation rollouts.

All four datasets store, per frame, the measured joints before the step (observation.state) and the joint
target of the step (action), in the same 16D interleaved RJ/LJ motor-radian order (asserted in common.load).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ARM, GRIP, DATA, NAMES, OPEN_RAW, load, openness, savefig

SETS = ["manual", "generated", "eval_manual", "eval_generated"]
LABEL = {"manual": "manual sim_10 (train)", "generated": "generated_500 (train)",
         "eval_manual": "eval: manual-data policy", "eval_generated": "eval: generated-data policy"}
COLOR = {"manual": "#2a78d6", "generated": "#eb6834", "eval_manual": "#2a78d6", "eval_generated": "#eb6834"}
KS = range(0, 9)
D = {k: load(k) for k in SETS}
out = {}


def per_ep(d, fn):
    return [fn(d["state"][a:b], d["action"][a:b]) for a, b in d["episodes"]]


def pct(x, qs=(50, 95, 99)):
    x = np.asarray(x)
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} | {"max": float(x.max()), "n": int(x.size)}


for name, d in D.items():
    r = {"episodes": len(d["episodes"]), "frames": int(len(d["state"])),
         "length": pct([b - a for a, b in d["episodes"]], (5, 50, 95)) | {"min": int(min(b - a for a, b in d["episodes"]))}}
    # 1. lead: mean |a[t] - s[t+k]| over arm joints and frames, per k
    lead = {}
    for k in KS:
        errs = per_ep(d, lambda S, A, k=k: np.abs(A[:len(S) - k, ARM] - S[k:, ARM]).mean(1) if len(S) > k else np.zeros(0))
        lead[k] = float(np.concatenate(errs).mean())
    r["lead_mean_abs_err"] = lead
    r["best_k"] = int(min(lead, key=lead.get))
    # 2. request-vs-measured gap (safety: 0.5 rad tracking hold) and per-tick request jump (0.25 rad hold)
    gap = np.abs(d["action"][:, ARM] - d["state"][:, ARM]).max(1)
    jump = np.concatenate(per_ep(d, lambda S, A: np.abs(np.diff(A[:, ARM], axis=0)).max(1)))
    r["gap_max_joint"] = pct(gap) | {"frac_gt_0.5": float((gap > 0.5).mean()), "frac_gt_0.25": float((gap > 0.25).mean())}
    r["action_jump_max_joint"] = pct(jump) | {f"frac_gt_{t}": float((jump > t).mean()) for t in (0.0833, 0.125, 0.25)}
    r["episodes_with_gap_gt_0.5"] = int(sum(np.any(np.abs(A[:, ARM] - S[:, ARM]).max(1) > 0.5) for S, A in
                                        [(d["state"][a:b], d["action"][a:b]) for a, b in d["episodes"]]))
    r["episodes_with_jump_gt_0.25"] = int(sum(np.any(np.abs(np.diff(A[:, ARM], axis=0)).max(1) > 0.25) for A in
                                          [d["action"][a:b] for a, b in d["episodes"]]))
    # 3. smoothness of the executed motion (measured state), 30 Hz finite differences, rad/s, rad/s^2, rad/s^3
    vel, acc, jerk, ep_jerk = [], [], [], []
    for a, b in d["episodes"]:
        S = d["state"][a:b, :][:, ARM]
        v = np.diff(S, axis=0) * 30; ac = np.diff(v, axis=0) * 30; j = np.diff(ac, axis=0) * 30
        vel.append(np.abs(v).max(1)); acc.append(np.abs(ac).max(1)); jerk.append(np.abs(j).max(1))
        ep_jerk.append(np.percentile(np.abs(j).max(1), 95))
    r["speed_max_joint_rad_s"] = pct(np.concatenate(vel))
    r["acc_max_joint"] = pct(np.concatenate(acc))
    r["jerk_max_joint"] = pct(np.concatenate(jerk))
    r["episode_p95_jerk"] = pct(ep_jerk, (5, 50, 95))
    # action smoothness too (what the policy is trained to output)
    ajerk = np.concatenate(per_ep(d, lambda S, A: np.abs(np.diff(A[:, ARM], n=3, axis=0) * 30 ** 3).max(1)))
    r["action_jerk_max_joint"] = pct(ajerk)
    # 4. gripper action encoding
    g = {}
    for side in "RL":
        o = openness(d["action"][:, GRIP[side]], side)
        g[side] = {"frac_binary(<2% or >98%)": float(((o < 0.02) | (o > 0.98)).mean()),
                   "frac_intermediate(5-95%)": float(((o > 0.05) & (o < 0.95)).mean()),
                   "unique_values": int(len(np.unique(np.round(d["action"][:, GRIP[side]], 5))))}
    r["gripper_action"] = g
    out[name] = r
    print(f"== {name}: eps {r['episodes']} frames {r['frames']} best_k {r['best_k']}  lead {[round(v, 4) for v in lead.values()]}")
    print(f"   gap p95 {r['gap_max_joint']['p95']:.3f} max {r['gap_max_joint']['max']:.3f} >0.5 {r['gap_max_joint']['frac_gt_0.5']:.4f} (eps {r['episodes_with_gap_gt_0.5']})"
          f" | jump p99 {r['action_jump_max_joint']['p99']:.3f} max {r['action_jump_max_joint']['max']:.3f} >0.125 {r['action_jump_max_joint']['frac_gt_0.125']:.4f} >0.25 eps {r['episodes_with_jump_gt_0.25']}")
    print(f"   speed p95 {r['speed_max_joint_rad_s']['p95']:.2f} jerk p95 {r['jerk_max_joint']['p95']:.0f} p99 {r['jerk_max_joint']['p99']:.0f} | action jerk p95 {r['action_jerk_max_joint']['p95']:.0f}")
    print(f"   gripper {g}")

# 5. dynamics consistency with the evaluation controller.
# The manual demos were recorded THROUGH the evaluation's control path (sim server: one joint target held for
# the 4 physics substeps of a 1/30 s step). Fit, per arm joint, ds[t] = alpha*(a[t]-s[t]) + beta*(s[t]-s[t-1]) on
# manual episodes, then measure how well it predicts each set's transitions.
def design(d, j):
    X, y, ep = [], [], []
    for e, (a, b) in enumerate(d["episodes"]):
        S, A = d["state"][a:b, j], d["action"][a:b, j]
        X.append(np.c_[A[1:-1] - S[1:-1], S[1:-1] - S[:-2]]); y.append(S[2:] - S[1:-1]); ep.append(np.full(len(S) - 2, e))
    return np.concatenate(X), np.concatenate(y), np.concatenate(ep)


dyn = {}
for j in ARM:
    Xm, ym, em = design(D["manual"], j)
    # leave-one-episode-out on manual for an honest in-domain error
    pred = np.zeros_like(ym)
    for e in np.unique(em):
        tr = em != e
        coef, *_ = np.linalg.lstsq(Xm[tr], ym[tr], rcond=None)
        pred[~tr] = Xm[~tr] @ coef
    coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    row = {"alpha_beta": coef.tolist(), "manual_loeo_rmse": float(np.sqrt(np.mean((pred - ym) ** 2)))}
    for name in ["generated", "eval_manual", "eval_generated"]:
        X, y, _ = design(D[name], j)
        row[f"{name}_rmse"] = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
        c2, *_ = np.linalg.lstsq(X, y, rcond=None)
        row[f"{name}_own_alpha_beta"] = c2.tolist()
    dyn[NAMES[j]] = row
out["dynamics_fit_on_manual"] = dyn
print("\n== dynamics: per joint  alpha,beta(manual) | rmse manual(LOEO) gen eval_m eval_g | own alpha gen")
for k, r in dyn.items():
    print(f"  {k:8s} {r['alpha_beta'][0]:.3f},{r['alpha_beta'][1]:.3f} | {r['manual_loeo_rmse']:.4f} {r['generated_rmse']:.4f} {r['eval_manual_rmse']:.4f} {r['eval_generated_rmse']:.4f} | gen {r['generated_own_alpha_beta'][0]:.3f},{r['generated_own_alpha_beta'][1]:.3f}")

json.dump(out, open(DATA / "action_semantics.json", "w"), indent=1)

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for name in SETS:
    ls = "-" if name in ("manual", "generated") else "--"
    ax[0].plot(list(KS), list(out[name]["lead_mean_abs_err"].values()), ls, marker="o", ms=4, color=COLOR[name], label=LABEL[name])
ax[0].set_xlabel("k (frames)"); ax[0].set_ylabel("mean |action[t] − state[t+k]| (rad, arm)")
ax[0].set_title("Action → state lead"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
for name in ["manual", "generated"]:
    d = D[name]; gap = np.abs(d["action"][:, ARM] - d["state"][:, ARM]).max(1)
    ax[1].hist(gap, bins=np.linspace(0, 0.6, 61), density=True, histtype="step", lw=1.6, color=COLOR[name], label=LABEL[name])
ax[1].axvline(0.5, color="k", ls=":", lw=1); ax[1].text(0.49, ax[1].get_ylim()[1] * 0.9, "safety hold 0.5", ha="right", fontsize=8)
ax[1].set_xlabel("max over arm joints |action − state| (rad)"); ax[1].set_ylabel("density"); ax[1].set_yscale("log")
ax[1].set_title("Command lead over measured state"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
savefig(fig, "action_state_lag.png")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for name in SETS:
    d = D[name]; ls = "-" if name in ("manual", "generated") else "--"
    sp, jk, aj = [], [], []
    for a, b in d["episodes"]:
        S = d["state"][a:b][:, ARM]; A = d["action"][a:b][:, ARM]
        sp.append(np.abs(np.diff(S, axis=0) * 30).max(1)); jk.append(np.abs(np.diff(S, n=3, axis=0) * 30 ** 3).max(1))
        aj.append(np.abs(np.diff(A, n=3, axis=0) * 30 ** 3).max(1))
    for i, (x, bins) in enumerate([(np.concatenate(sp), np.linspace(0, 4, 81)), (np.concatenate(jk), np.logspace(0, 5, 81)), (np.concatenate(aj), np.logspace(0, 5, 81))]):
        ax[i].hist(x, bins=bins, density=True, histtype="step", lw=1.5, ls=ls, color=COLOR[name], label=LABEL[name])
ax[0].set_xlabel("max joint speed (rad/s), measured"); ax[1].set_xlabel("max joint jerk (rad/s³), measured"); ax[2].set_xlabel("max joint jerk (rad/s³), ACTION")
for a in ax[1:]: a.set_xscale("log")
for a in ax: a.set_yscale("log"); a.grid(alpha=.3)
ax[0].legend(fontsize=7); fig.suptitle("Per-frame speed and jerk (arm joints, 30 Hz finite differences)")
savefig(fig, "speed_jerk_distributions.png")
