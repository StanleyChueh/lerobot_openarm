"""Stable vs transient success, failure sub-classes, time budget, and a position-adjusted policy comparison.

Stable success = the simulator's (any-step) success AND the left gripper still closed on the can in the last 10
recorded frames (commanded closed, measured blocked open). Release-after-success = success, then the left gripper
commanded open. Visual spot-check of the manual policy: figures/manual_policy_success_end.png (model review, not human).
"""
import json

import numpy as np
import pandas as pd

from common import DATA, GRIP, load, openness

E = pd.read_csv(DATA / "eval_episodes.csv")
rows = []
for name in ["eval_manual", "eval_generated"]:
    d = load(name)
    for _, e in E[E.set == name].iterrows():
        a, b = d["episodes"][int(e.episode)]
        S, A = d["state"][a:b], d["action"][a:b]
        cL, mL = openness(A[:, GRIP["L"]], "L"), openness(S[:, GRIP["L"]], "L")
        cR, mR = openness(A[:, GRIP["R"]], "R"), openness(S[:, GRIP["R"]], "R")
        held_end = bool(((cL[-10:] < 0.5) & (mL[-10:] - cL[-10:] > 0.15)).all())
        t_open_after = None
        if not np.isnan(e.t_stable_handover):
            later = np.flatnonzero(cL[int(e.t_stable_handover):] > 0.5)
            t_open_after = int(later[0] + e.t_stable_handover) if len(later) else None
        # failure sub-class
        hL = (cL < 0.6) & (mL - cL > 0.15); hR = (cR < 0.6) & (mR - cR > 0.15)
        sub = None
        if not e.success:
            tR_rel = None
            if not np.isnan(e.t_R_hold):
                idx = np.flatnonzero(~hR[int(e.t_R_hold):]); tR_rel = int(idx[0] + e.t_R_hold) if len(idx) else None
            if np.isnan(e.t_L_hold):
                sub = "left never took it"
            elif hL[-10:].any() and hR[-10:].any():
                sub = "still passing at time limit"
            elif e.L_hold_longest < 30 and not hL[-10:].any():
                sub = "dropped during/after transfer (left lost it within 1 s)"
            else:
                sub = "other"
        rows.append({"set": name, "episode": int(e.episode), "held_by_left_at_end": held_end,
                     "t_left_opened_after_success": t_open_after, "failure_subclass": sub})
X = E.merge(pd.DataFrame(rows), on=["set", "episode"])
X["stable_success"] = X.success & X.held_by_left_at_end
X.to_csv(DATA / "eval_episodes.csv", index=False)

out = {}
for policy, g in X.groupby("policy"):
    s = g[g.success]
    out[policy] = {"success_any_step": f"{int(g.success.sum())}/100",
                   "stable_success_held_at_end": f"{int(g.stable_success.sum())}/100",
                   "successes_released_afterwards": int(s.t_left_opened_after_success.notna().sum()),
                   "release_step_after_success_median": float((s.t_left_opened_after_success - s.t_stable_handover).median()),
                   "failure_subclasses": g[~g.success].failure_subclass.value_counts().to_dict(),
                   "L_hold_longest_failures_median": float(g[~g.success].L_hold_longest.median()),
                   "sim_steps_median": float(g.sim_steps.median()),
                   "stable_handover_step_success": [float(s.t_stable_handover.quantile(q)) for q in (0, .5, .9, 1)],
                   "successes_within_last_20_steps_of_budget": int((s.margin_steps < 20).sum())}

# manual training-data horizon vs eval horizon
for name in ["manual", "generated"]:
    d = load(name); L = np.array([b - a for a, b in d["episodes"]])
    out[f"{name}_train_episode_length"] = [int(L.min()), int(np.median(L)), int(L.max())]


# position-adjusted comparison: logistic regression by IRLS (numpy), success ~ 1 + gen + x + x^2 + y (+ gen*x)
def logit(Xm, y, iters=50):
    w = np.zeros(Xm.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xm @ w)); W = p * (1 - p)
        H = Xm.T @ (Xm * W[:, None]) + 1e-6 * np.eye(len(w))
        w = w + np.linalg.solve(H, Xm.T @ (y - p))
    p = 1 / (1 + np.exp(-Xm @ w)); W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.inv(Xm.T @ (Xm * W[:, None]))))
    from scipy.stats import norm
    return w, se, 2 * (1 - norm.cdf(np.abs(w / se)))


Z = X.dropna(subset=["x_est"])
x = (Z.x_est - 0.235) / 0.07; yv = (Z.y_est - 0.01) / 0.04; g = (Z.policy == "generated-data policy").astype(float)
for label, cols, names in [
        ("success ~ gen + x + x^2 + y", [np.ones(len(Z)), g, x, x * x, yv], ["1", "gen", "x", "x^2", "y"]),
        ("success ~ gen*x + gen*y", [np.ones(len(Z)), g, x, yv, g * x, g * yv], ["1", "gen", "x", "y", "gen:x", "gen:y"])]:
    for target in ["success", "stable_success"]:
        w, se, p = logit(np.c_[tuple(cols)], Z[target].astype(float).to_numpy())
        out[f"logit[{target}] {label}"] = {n: {"coef": round(float(c), 3), "p": round(float(pp), 4)} for n, c, pp in zip(names, w, p)}
json.dump(out, open(DATA / "eval_stability.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
