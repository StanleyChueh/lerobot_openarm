"""Phase 4: stage-wise failure analysis of both 100-episode evaluations, and success vs initial can position.

The rollouts have no can pose, so stages come from the grippers: a hand HOLDS the can when it is commanded
(at least partly) closed but is held open well beyond its command -- the can is in the way. The rule is validated
on the 510 training episodes, where the can pose is known (can within 0.11 m of that hand's EEF and off the pad).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from common import DATA, GRIP, load, load_hdf5, openness, savefig

BLOCK = 0.15      # measured openness exceeds the command by this much = blocked by the can
CMD_CLOSED = 0.6  # commanded openness below this = (partly) closing
HOLD_STEPS = 30   # HANDOVER_RECEIVER_HOLD_STEPS at 30 Hz
REST_Z = 0.380


def holding(S, A, side):
    cmd, meas = openness(A[:, GRIP[side]], side), openness(S[:, GRIP[side]], side)
    return (cmd < CMD_CLOSED) & (meas - cmd > BLOCK)


def runs(mask):
    """Longest run of True."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else None


def stages(S, A):
    hR, hL = holding(S, A, "R"), holding(S, A, "L")
    tR = first(hR)
    tL = first(hL)
    # stable hand-over (the success term's stage 3): left holds, right not holding, for HOLD_STEPS running
    passed = hL & ~hR
    t_done = None
    cur = 0
    for t, p in enumerate(passed):
        cur = cur + 1 if p else 0
        if cur >= HOLD_STEPS and tR is not None and tL is not None and t > tR:
            t_done = t
            break
    if tR is None:
        stage = "approach/grasp"          # right hand never held the can
    elif tL is None:
        stage = "transfer"                # right held it, left never did
    elif t_done is None:
        stage = "release/hold"            # both held it, no stable left-only hold of 1 s
    else:
        stage = "complete"
    end_state = ("left" if hL[-10:].all() and not hR[-10:].any() else
                 "both" if hL[-10:].any() and hR[-10:].any() else
                 "right" if hR[-10:].all() else "none")
    return {"t_R_hold": tR, "t_L_hold": tL, "t_stable_handover": t_done, "stage_reached": stage,
            "R_hold_frames": int(hR.sum()), "L_hold_frames": int(hL.sum()), "L_hold_longest": runs(hL & ~hR),
            "end_holder": end_state}


# ---------------------------------------------------------------- validate on training episodes (truth from HDF5)
h = load_hdf5()
val_rows = []
for name, eps_h in [("manual", h["source"]["episodes"]), ("generated", h["generated"]["episodes"])]:
    d = load(name)
    ann = h["annotated"]["episodes"]
    for i, (a, b) in enumerate(d["episodes"]):
        S, A = d["state"][a:b], d["action"][a:b]
        e = eps_h[i]
        can = np.r_[e["initial_state/rigid_object/can/root_pose"][:, :3], e["states/rigid_object/can/root_pose"][:-1, :3]]
        pR = (ann[i] if name == "manual" else e)["obs/right_eef_pos"]
        pL = (ann[i] if name == "manual" else e)["obs/eef_pos"]
        up = can[:, 2] > REST_Z + 0.005
        tR = (np.linalg.norm(pR - can, axis=1) < 0.11) & up & (openness(A[:, GRIP["R"]], "R") < 0.5)
        tL = (np.linalg.norm(pL - can, axis=1) < 0.11) & up & (openness(A[:, GRIP["L"]], "L") < 0.5)
        for side, truth in (("R", tR), ("L", tL)):
            pred = holding(S, A, side)
            val_rows.append({"set": name, "side": side, "tp": int((pred & truth).sum()), "fp": int((pred & ~truth).sum()),
                             "fn": int((~pred & truth).sum()), "tn": int((~pred & ~truth).sum())})
        st = stages(S, A)
        val_rows.append({"set": name, "side": "episode", "complete": st["stage_reached"] == "complete"})
V = pd.DataFrame(val_rows)
valid = {}
for (s, side), g in V[V.side != "episode"].groupby(["set", "side"]):
    tp, fp, fn, tn = g.tp.sum(), g.fp.sum(), g.fn.sum(), g.tn.sum()
    valid[f"{s}_{side}"] = {"precision": float(tp / (tp + fp)), "recall": float(tp / (tp + fn)), "frames": int(tp + fp + fn + tn)}
for s, g in V[V.side == "episode"].groupby("set"):
    valid[f"{s}_episodes_classified_complete"] = f"{int(g.complete.sum())}/{len(g)} (all are successful demos)"
print("stage classifier validation:", json.dumps(valid, indent=1))

# ---------------------------------------------------------------- evaluation episodes
pos = pd.read_csv(DATA / "start_positions.csv")
X_FIT = (pos[pos.set == "sim_20"].x_true.min(), pos[pos.set == "sim_20"].x_true.max())
Y_FIT = (pos[pos.set == "sim_20"].y_true.min(), pos[pos.set == "sim_20"].y_true.max())
rows = []
for name, policy in [("eval_manual", "manual-data policy"), ("eval_generated", "generated-data policy")]:
    d = load(name)
    ev = json.loads((d["info"] and (DATA.parent.parent.parent / "x").exists()) or "{}") if False else None
    meta = json.load(open(str(__import__("common").DATASETS[name] / "meta/openarm_eval.json")))
    for i, (a, b) in enumerate(d["episodes"]):
        S, A = d["state"][a:b], d["action"][a:b]
        m = meta["episodes"][i]
        assert m["episode_index"] == i and m["actions"] == b - a, "openarm_eval.json does not match the frames"
        p = pos[(pos.set == name) & (pos.episode == i)].iloc[0]
        r = {"set": name, "policy": policy, "episode": i, "success": bool(m["success"]), "sim_steps": b - a,
             "wall_s": m["seconds"], "x_est": p.x_est, "y_est": p.y_est,
             "extrapolated": not (X_FIT[0] - 0.01 <= p.x_est <= X_FIT[1] + 0.01 and Y_FIT[0] - 0.01 <= p.y_est <= Y_FIT[1] + 0.01),
             **stages(S, A)}
        # gripper command shape
        for side in "RL":
            o = openness(A[:, GRIP[side]], side)
            r[f"{side}_cmd_intermediate_frac"] = float(((o > 0.1) & (o < 0.9)).mean())
        rows.append(r)
E = pd.DataFrame(rows)
E["margin_steps"] = E.sim_steps - E.t_stable_handover
E.to_csv(DATA / "eval_episodes.csv", index=False)

# consistency of the stage classifier with the simulator's success label
cons = pd.crosstab([E.policy, E.success], E.stage_reached)
print(cons)
summary = {"classifier_validation": valid, "crosstab_success_vs_stage": {f"{k[0]}|success={k[1]}": v for k, v in cons.T.to_dict().items()}}
for policy, g in E.groupby("policy"):
    s = g[g.success]; f = g[~g.success]
    summary[policy] = {
        "success": f"{int(g.success.sum())}/{len(g)}",
        "sim_steps_success_median": float(s.sim_steps.median()), "sim_steps_failure_median": float(f.sim_steps.median()),
        "failure_stage_counts": f.stage_reached.value_counts().to_dict(),
        "failure_end_holder": f.end_holder.value_counts().to_dict(),
        "success_end_holder": s.end_holder.value_counts().to_dict(),
        "t_R_hold_median_all": float(g.t_R_hold.median()), "t_L_hold_median_all": float(g.t_L_hold.median()),
        "t_stable_handover_median_success": float(s.t_stable_handover.median()),
        "t_stable_handover_max_success": float(s.t_stable_handover.max()),
        "margin_steps_success_min": float(s.margin_steps.min()),
        "success_without_stable_handover_in_recorded_frames": int(s.t_stable_handover.isna().sum()),
        "gripper_cmd_intermediate_frac_median": {k: float(g[f"{k}_cmd_intermediate_frac"].median()) for k in "RL"},
    }
# success vs position: regions on x (generation range starts at 0.203) and y
E["x_region"] = pd.cut(E.x_est, [0, 0.203, 0.27, 0.5], labels=["x<0.203 (no generated starts)", "0.203-0.27", "x>0.27"])
E["y_region"] = pd.cut(E.y_est, [-1, -0.01, 0.03, 1], labels=["y<-0.01", "-0.01..0.03", "y>0.03"])
reg = {}
for col in ["x_region", "y_region"]:
    t = E.groupby([col, "policy"], observed=False).success.agg(["sum", "count"]).unstack("policy")
    reg[col] = {str(k): {p: f"{int(t.loc[k, ('sum', p)])}/{int(t.loc[k, ('count', p)])}" for p in E.policy.unique()} for k in t.index}
    for k in t.index:
        a, n = int(t.loc[k, ("sum", "manual-data policy")]), int(t.loc[k, ("count", "manual-data policy")])
        b, m = int(t.loc[k, ("sum", "generated-data policy")]), int(t.loc[k, ("count", "generated-data policy")])
        if n and m:
            reg[col][str(k)]["fisher_p"] = float(fisher_exact([[a, n - a], [b, m - b]])[1])
summary["success_by_region"] = reg
# overall test
a, b = int(E[E.policy == "manual-data policy"].success.sum()), int(E[E.policy == "generated-data policy"].success.sum())
summary["overall_fisher_p"] = float(fisher_exact([[a, 100 - a], [b, 100 - b]])[1])


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - w), float(c + w)]


summary["wilson95"] = {"manual-data policy": wilson(a, 100), "generated-data policy": wilson(b, 100)}
# position-matched comparison: logistic model success ~ policy + x + y (+ x^2) via statsmodels if available
try:
    import statsmodels.formula.api as smf
    E["gen"] = (E.policy == "generated-data policy").astype(int)
    fit = smf.logit("success ~ gen + x_est + I(x_est**2) + y_est", data=E.assign(success=E.success.astype(int))).fit(disp=0)
    summary["logit_success~policy+position"] = {k: [float(fit.params[k]), float(fit.pvalues[k])] for k in fit.params.index}
    fit2 = smf.logit("success ~ gen * x_est + y_est", data=E.assign(success=E.success.astype(int))).fit(disp=0)
    summary["logit_interaction_policy_x"] = {k: [float(fit2.params[k]), float(fit2.pvalues[k])] for k in fit2.params.index}
except Exception as ex:  # noqa: BLE001
    summary["logit"] = f"skipped: {ex!r}"
json.dump(summary, open(DATA / "eval_analysis.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in summary.items() if k != "classifier_validation"}, indent=1, default=str))

# ---------------------------------------------------------------- figures
BLUE, ORANGE = "#2a78d6", "#eb6834"
order = ["approach/grasp", "transfer", "release/hold", "complete"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
w = 0.38
for k, (policy, c) in enumerate([("manual-data policy", BLUE), ("generated-data policy", ORANGE)]):
    g = E[E.policy == policy]
    for j, succ in enumerate([False, True]):
        cnt = [int(((g.stage_reached == s) & (g.success == succ)).sum()) for s in order]
        ax[j].bar(np.arange(4) + (k - 0.5) * w, cnt, w, color=c, label=policy)
        for x, v in zip(np.arange(4) + (k - 0.5) * w, cnt):
            ax[j].text(x, v + 0.5, str(v), ha="center", fontsize=8)
for j, t in enumerate(["FAILED episodes: furthest stage reached", "SUCCESSFUL episodes: stage seen in recorded frames"]):
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels(order); ax[j].set_title(t, fontsize=10); ax[j].set_ylabel("episodes")
ax[0].legend(fontsize=8)
savefig(fig, "stagewise_failures.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for a, (policy, c) in zip(ax, [("manual-data policy", BLUE), ("generated-data policy", ORANGE)]):
    g = E[E.policy == policy]
    a.scatter(g[g.success].x_est, g[g.success].y_est, s=34, c=c, edgecolor="white", lw=0.6, label="success")
    a.scatter(g[~g.success].x_est, g[~g.success].y_est, s=34, marker="x", c="#444444", label="failure")
    a.axvline(0.2029, color=ORANGE, ls="--", lw=1); a.text(0.205, 0.075, "generation range →", color=ORANGE, fontsize=8)
    a.set_title(f"{policy}: {int(g.success.sum())}/100", fontsize=10); a.set_xlabel("estimated initial can x (m)")
    a.set_aspect("equal"); a.grid(alpha=0.3)
ax[0].set_ylabel("estimated initial can y (m)"); ax[0].legend(fontsize=8)
savefig(fig, "success_vs_position.png")
