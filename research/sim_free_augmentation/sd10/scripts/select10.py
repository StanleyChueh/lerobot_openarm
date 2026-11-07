"""STEP 3 - deterministic selection of exactly 10 source episodes (protocol §4) plus controls and comparison metrics.

    env -u PYTHONPATH -u LD_LIBRARY_PATH ../../../.venv/bin/python scripts/select10.py

Reads full50_episode_features.csv; writes selected_10_episodes.json and outputs/selection_details.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sd10 import common as C  # noqa: E402

K = 10
POS = ("lid_u", "lid_v")
INTER = ("left_receive_x_m", "left_receive_y_m", "left_receive_z_m", "handover_dt_s")
W_POS, W_INTER = 1.0, 0.25
SEED_RANDOM, SEED_NULL, N_NULL = 20261107, 20261108, 10_000
EPS_Q, TOL = 1e-4, 1e-9


def read():
    rows = list(csv.DictReader(open(HERE / "full50_episode_features.csv")))
    for r in rows:
        r["episode"] = int(r["episode"])
        r["eligible"] = r["eligible"] == "True"
    return rows


def fnum(r, k):
    return float(r[k]) if r[k] not in ("", "None") else np.nan


def metric_space(el):
    X = np.array([[fnum(r, k) for k in POS + INTER] for r in el])
    sd = X.std(0)                                            # population std over eligible
    Z = X / sd
    w = np.array([W_POS] * len(POS) + [W_INTER] * len(INTER))
    D = np.sqrt((((Z[:, None, :] - Z[None, :, :]) ** 2) * w).sum(-1))
    return D, sd


def setcover_feasible(D, r):
    n = len(D)
    A = (D <= r + TOL).astype(float)
    res = milp(c=np.ones(n), constraints=LinearConstraint(A, lb=1, ub=np.inf), integrality=np.ones(n),
               bounds=Bounds(0, 1), options={"presolve": True})
    return res.status == 0 and round(res.fun) <= K


def kcenter_radius(D):
    cand = np.unique(np.round(D[np.triu_indices(len(D), 1)], 12))
    cand = np.concatenate([[0.0], cand])
    lo, hi = 0, len(cand) - 1
    while lo < hi:                                           # smallest feasible radius
        mid = (lo + hi) // 2
        if setcover_feasible(D, cand[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo])


def pmedian_at_radius(D, R, q, cuts=()):
    """min sum_e min_s d(e,s) - EPS_Q*sum q_s  s.t. |S| = K and every e assigned within R. Variables: x_s, y_es."""
    n = len(D)
    pairs = [(e, s) for e in range(n) for s in range(n) if D[e, s] <= R + TOL]
    nv = n + len(pairs)
    c = np.zeros(nv)
    c[:n] = -EPS_Q * q
    for j, (e, s) in enumerate(pairs):
        c[n + j] = D[e, s]
    rows, lb, ub = [], [], []
    A = lil_matrix((n + len(pairs) + 1 + len(cuts), nv))
    i = 0
    for e in range(n):                                       # each e assigned once
        for j, (ee, s) in enumerate(pairs):
            if ee == e:
                A[i, n + j] = 1
        lb.append(1); ub.append(1); i += 1
    for j, (e, s) in enumerate(pairs):                       # y_es <= x_s
        A[i, n + j] = 1; A[i, s] = -1
        lb.append(-np.inf); ub.append(0); i += 1
    A[i, :n] = 1; lb.append(K); ub.append(K); i += 1         # |S| = K
    for S in cuts:                                           # no-good cuts
        for s in S:
            A[i, s] = 1
        lb.append(-np.inf); ub.append(K - 1); i += 1
    res = milp(c=c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.ones(nv), bounds=Bounds(0, 1))
    if res.status != 0:
        return None, None
    S = tuple(sorted(int(s) for s in np.flatnonzero(res.x[:n] > 0.5)))
    return S, float(res.fun)


def assign_cost(D, S):
    return float(D[:, list(S)].min(1).sum())


def coverage(P, targets, S):
    """P: (n,2) px positions of the pool (indexed by episode); S: source episodes; targets: target episodes."""
    src = P[list(S)]
    T = P[list(targets)]
    d = np.sqrt(((T[:, None] - src[None]) ** 2).sum(-1)).min(1)
    pd = np.sqrt(((src[:, None] - src[None]) ** 2).sum(-1))
    pd[np.diag_indices(len(S))] = np.inf
    return {"covering_radius_px": float(d.max()), "mean_nearest_px": float(d.mean()),
            "frac_within_20px": float((d <= 20).mean()), "frac_within_40px": float((d <= 40).mean()),
            "min_pairwise_spacing_px": float(pd.min()), "hull_area_px2": float(ConvexHull(src).volume)}


def main():
    rows = read()
    el = [r for r in rows if r["eligible"]]
    ids = [r["episode"] for r in el]
    D, sd = metric_space(el)
    q = np.array([fnum(r, "quality_score_q") for r in el])
    n = len(el)
    assert n >= K, "INFEASIBLE"
    # Stage 1
    R = kcenter_radius(D)
    # Stage 2
    S, obj = pmedian_at_radius(D, R, q)
    radius_S = float(D[:, list(S)].min(1).max())
    # Stage 3: tie check with no-good cuts
    ties, cuts = [S], [S]
    while True:
        S2, obj2 = pmedian_at_radius(D, R, q, cuts)
        if S2 is None or obj2 > obj + TOL:
            break
        ties.append(S2); cuts.append(S2)
    best = min(ties, key=lambda s: tuple(ids[i] for i in s))
    sel = sorted(ids[i] for i in best)
    runner_up = {"subset": [ids[i] for i in S2] if S2 else None, "objective": obj2}
    # controls
    rng = np.random.default_rng(SEED_RANDOM)
    rand10 = sorted(int(x) for x in rng.choice(ids, K, replace=False))
    # greedy farthest point (diagnostic): start at the eligible medoid
    g = [int(np.argmin(D.sum(1)))]
    while len(g) < K:
        g.append(int(np.argmax(D[:, g].min(1))))
    fps = sorted(ids[i] for i in g)
    orig = list(range(10))
    # metrics
    P = np.array([[fnum(r, "lid_u"), fnum(r, "lid_v")] for r in rows])
    hull_el = ConvexHull(P[ids])
    on_hull = {ids[i] for i in hull_el.vertices}
    pool_hull = ConvexHull(P).volume
    idx = {e: i for i, e in enumerate(ids)}

    def comp_metrics(S):
        si = [idx[e] for e in S if e in idx]
        m = {"pool50": coverage(P, range(50), S), "v4a_locked6": coverage(P, C.V4A_LOCKED, S)}
        m["pool50"]["hull_frac_of_pool_hull"] = m["pool50"]["hull_area_px2"] / pool_hull
        m["metric_space_eligible"] = {"covering_radius": float(D[:, si].min(1).max()) if len(si) == len(S) else None,
                                      "sum_nearest": assign_cost(D, si) if len(si) == len(S) else None}
        m["n_on_eligible_hull"] = int(sum(e in on_hull for e in S))
        sub = [r for r in rows if r["episode"] in S]
        m["interaction"] = {k: {"min": float(np.min([fnum(r, k) for r in sub])), "max": float(np.max([fnum(r, k) for r in sub])),
                                "std": float(np.std([fnum(r, k) for r in sub]))}
                            for k in INTER + ("right_grasp_x_m", "right_grasp_y_m", "right_grasp_yaw_deg", "left_receive_yaw_deg",
                                              "tcp_separation_lc_m", "meas_dual_hold_frames", "cmd_overlap_frames", "length")}
        qs = [fnum(r, "quality_score_q") for r in sub]
        m["mean_quality_q"] = float(np.nanmean(qs))
        m["all_pass_quality"] = all(r["passes_quality"] == "True" for r in sub)
        return m

    sets = {"original_ten": orig, "sd10_selected": sel, "random_ten_control": rand10, "greedy_fps_diagnostic": fps}
    M = {k: comp_metrics(v) for k, v in sets.items()}
    # null distribution
    rng2 = np.random.default_rng(SEED_NULL)
    keys = ["covering_radius_px", "mean_nearest_px", "frac_within_20px", "frac_within_40px", "min_pairwise_spacing_px", "hull_area_px2"]
    null = {t: {k: [] for k in keys} for t in ("pool50", "v4a_locked6")}
    null_ms = {"covering_radius": [], "sum_nearest": []}
    for _ in range(N_NULL):
        S_ = sorted(int(x) for x in rng2.choice(ids, K, replace=False))
        for t, tg in (("pool50", range(50)), ("v4a_locked6", C.V4A_LOCKED)):
            cv = coverage(P, tg, S_)
            for k in keys:
                null[t][k].append(cv[k])
        si = [idx[e] for e in S_]
        null_ms["covering_radius"].append(float(D[:, si].min(1).max()))
        null_ms["sum_nearest"].append(assign_cost(D, si))
    lower_better = {"covering_radius_px", "mean_nearest_px", "covering_radius", "sum_nearest"}
    for k_set, m in M.items():
        m["null_percentile_better_than"] = {}
        for t in ("pool50", "v4a_locked6"):
            for k in keys:
                arr = np.array(null[t][k])
                x = m[t][k]
                pct = float((arr > x).mean() if k in lower_better else (arr < x).mean())
                m["null_percentile_better_than"][f"{t}.{k}"] = pct
        for k in null_ms:
            arr = np.array(null_ms[k])
            x = m["metric_space_eligible"][k]
            if x is not None:
                m["null_percentile_better_than"][f"metric_space.{k}"] = float((arr > x).mean())
    null_summary = {t: {k: {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
                        for k, v in null[t].items()} for t in null}
    null_summary["metric_space"] = {k: {"median": float(np.median(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
                                    for k, v in null_ms.items()}
    details = {"protocol": "ten_source_selection_protocol.md (+ Amendment A1)", "eligible": ids, "n_eligible": n,
               "excluded": {r["episode"]: r["exclusion_reason"].strip() for r in rows if not r["eligible"]},
               "feature_std_eligible": dict(zip(POS + INTER, sd.tolist())), "weights": {"position": W_POS, "interaction": W_INTER},
               "stage1_kcenter_radius": R, "stage2_objective": obj, "stage2_subset_radius": radius_S,
               "stage3_ties": [[ids[i] for i in t] for t in ties], "next_best_after_cut": runner_up,
               "sets": sets, "metrics": M, "null_random_ten": null_summary, "seeds": {"random_ten": SEED_RANDOM, "null": SEED_NULL, "n_null": N_NULL}}
    C.dump(details, C.OUT / "selection_details.json")
    sel_rows = [r for r in rows if r["episode"] in sel]
    out = {"study": "SFA-SD10 source-diversity ten (PROPOSED; source curation only)",
           "status": "Selection executed — pending independent Codex review",
           "source_dataset": {"repo": C.POOL_REPO, "revision": C.POOL_REV},
           "original_ten_dataset": {"repo": C.TEN_REPO, "revision": C.TEN_REV, "identical_to_pool_episodes": list(range(10))},
           "selected_episodes": sel,
           "overlap_with_original_ten": sorted(set(sel) & set(orig)),
           "v4a_locked_excluded": list(C.V4A_LOCKED),
           "random_ten_control": rand10, "random_seed": SEED_RANDOM,
           "selection_rule": {"stage1": "exact k-center (k=10) on eligible episodes", "stage2": "radius-constrained p-median, -1e-4*q tie-break",
                              "stage3": "no-good-cut tie check; lexicographic", "solver": "scipy.optimize.milp (HiGHS)",
                              "features": {"position": list(POS), "interaction": list(INTER)},
                              "weights_on_squared_normalised_diffs": {"position": W_POS, "interaction": W_INTER}},
           "selected_details": [{"episode": r["episode"], "lid_uv_px": [round(fnum(r, "lid_u"), 1), round(fnum(r, "lid_v"), 1)],
                                 "left_receive_xyz_m": [round(fnum(r, f"left_receive_{a}_m"), 4) for a in "xyz"],
                                 "handover_dt_s": round(fnum(r, "handover_dt_s"), 3), "length": int(r["length"]),
                                 "quality_score_q": round(fnum(r, "quality_score_q"), 3), "on_eligible_hull": r["episode"] in on_hull}
                                for r in sel_rows],
           "provenance": {"code": ["scripts/select10.py", "scripts/build_features.py"], "features": "full50_episode_features.csv"},
           "use_restrictions": ["Selected episodes are training sources for a separate SD10 study; never evaluation data for it.",
                                "Recordings 10-49 are development data; not an untouched independent test.",
                                "Do not use VLA outcomes to amend this selection."]}
    C.dump(out, HERE / "selected_10_episodes.json")
    print("eligible", n, "R*", round(R, 4), "obj", round(obj, 4), "ties", len(ties))
    print("SD10:", sel, " original:", orig, " random:", rand10, " fps:", fps)
    for k, m in M.items():
        print(k, {kk: round(v, 3) for kk, v in m["pool50"].items()}, "hull", m["n_on_eligible_hull"], "q", round(m["mean_quality_q"], 3))


if __name__ == "__main__":
    main()
