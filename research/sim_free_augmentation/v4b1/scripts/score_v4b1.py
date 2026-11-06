"""v4B.1 scorer: implements V4B1_EXPERIMENT_SPEC.md section 7 (frozen). Reuses the v4B per-recording scoring
(v4/scripts/score_v4b.py: score_run, train_std, bins, rel_ci) and the v4B prediction archives of the frozen references.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/score_v4b1.py [--pred_dir P]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V4 = HERE.parent / "v4"
sys.path.insert(0, str(V4 / "scripts"))
import score_v4b as sv  # noqa: E402

SC = json.loads((HERE / "configs/v4b1.json").read_text())["scoring"]
FROZEN = ["R", "RpBoth", "RpPos"]
METRICS = ["approach", "grasp", "carry", "handover", "release", "pb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=str(HERE / "outputs/predictions"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(SC["bootstrap_seed"])
    vpred = V4 / "outputs/v4b/predictions"
    meta = json.loads((vpred / "eval_meta.json").read_text())
    eps = [int(e) for e in meta["episodes"]]
    tg = np.load(vpred / "targets.npz")
    sd = sv.train_std()
    B = sv.bins(eps)
    conds = SC["conditions"]
    S = {}
    for p in sorted(vpred.glob("*.npz")):
        if p.stem.rsplit("_s", 1)[0] in FROZEN:
            S[p.stem] = sv.score_run(np.load(p), tg, sd, eps, conds)
    for p in sorted(Path(a.pred_dir).glob("*.npz")):
        if p.stem != "targets":
            S[p.stem] = sv.score_run(np.load(p), tg, sd, eps, conds)
    arms = sorted({r.rsplit("_s", 1)[0] for r in S})
    runs = {x: [f"{x}_s{s}" for s in (0, 1, 2) if f"{x}_s{s}" in S] for x in arms}
    sets = {"all40": eps, "near": [e for e in eps if B[e]["bin"] == "near"], "far": [e for e in eps if B[e]["bin"] == "far"]}

    def rec(x, cond, e, m, seeds=None):
        rs = runs[x] if seeds is None else [f"{x}_s{s}" for s in seeds if f"{x}_s{s}" in S]
        v = [S[r][cond]["by_recording"][e][m] for r in rs]
        return None if any(q is None for q in v) else float(np.mean(v))

    def contrast(x, ref, sn, m, cond="original", seeds=None):
        xs = [(rec(x, cond, e, m, seeds), rec(ref, cond, e, m, seeds)) for e in sets[sn]]
        xs = [(p, q) for p, q in xs if p is not None and q is not None]
        if len(xs) < 2:
            return None
        return sv.rel_ci([p for p, _ in xs], [q for _, q in xs], rng, SC["bootstrap_n"])

    table = {}
    for x in arms:
        if not runs[x]:
            continue
        t = {sn: {m: float(np.mean([v for v in [rec(x, "original", e, m) for e in E] if v is not None])) for m in METRICS} for sn, E in sets.items()}
        t["gripper_timing_frames"] = float(np.mean([S[r]["original"]["gripper_timing_frames"] for r in runs[x]]))
        t["per_seed"] = {r: {"far_grasp": float(np.mean([S[r]["original"]["by_recording"][e]["grasp"] for e in sets["far"]])),
                             "all40_handover": float(np.mean([S[r]["original"]["by_recording"][e]["handover"] for e in eps])),
                             "all40_pb": float(np.mean([S[r]["original"]["by_recording"][e]["pb"] for e in eps]))} for r in runs[x]}
        table[x] = t

    def full(x, ref, cond="original"):
        out = {f"{sn}/{m}": contrast(x, ref, sn, m, cond) for sn in sets for m in METRICS}
        common = [s for s in (0, 1, 2) if f"{x}_s{s}" in S and f"{ref}_s{s}" in S]
        out["per_seed"] = {f"s{s}": {k: contrast(x, ref, sn, m, cond, [s])[0] for k, (sn, m) in
                                     {"far/grasp": ("far", "grasp"), "all40/handover": ("all40", "handover"), "all40/pb": ("all40", "pb")}.items()}
                           for s in common}
        if cond == "original":
            out["gripper_timing_delta_frames"] = table[x]["gripper_timing_frames"] - table[ref]["gripper_timing_frames"]
        return out

    C = {f"{x}__vs__{ref}": full(x, ref) for x, ref in SC["contrasts"] if runs.get(x) and runs.get(ref)}
    # inference-time image conditions (same condition for both models; not a training intervention)
    inf = {}
    for x, ref in SC["inference_condition_contrasts"]:
        if runs.get(x) and runs.get(ref):
            inf[f"{x}__vs__{ref}"] = {c: {k: contrast(x, ref, sn, m, c) for k, (sn, m) in
                                          {"far/grasp": ("far", "grasp"), "all40/grasp": ("all40", "grasp"), "all40/handover": ("all40", "handover")}.items()}
                                      for c in conds}
    # v4B screening rule (frozen in v4B) applied to each new arm vs R
    screen = {}
    for x in arms:
        if x in ("R",) or not runs[x] or not runs.get("R"):
            continue
        g, h = contrast(x, "R", "far", "grasp"), contrast(x, "R", "all40", "handover")
        screen[x] = {"far_grasp_rel": g[0], "far_grasp_ci": g[1], "handover_degradation": -h[0],
                     "PASS": bool(g[0] >= 0.05 and g[1][0] > 0 and -h[0] <= 0.02)}
    # retained-gain fraction (H1): (R - NoRGB) / (R - Both), recording bootstrap
    def retained(x_full, x_part, sn="far", m="grasp"):
        r = np.array([rec("R", "original", e, m) for e in sets[sn]])
        f = np.array([rec(x_full, "original", e, m) for e in sets[sn]])
        p = np.array([rec(x_part, "original", e, m) for e in sets[sn]])
        pt = (r.mean() - p.mean()) / (r.mean() - f.mean())
        idx = rng.integers(0, len(r), (SC["bootstrap_n"], len(r)))
        bs = (r[idx].mean(1) - p[idx].mean(1)) / (r[idx].mean(1) - f[idx].mean(1))
        return [float(pt), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    dec = decisions(C, screen, retained if runs.get("RpBoth_NoSynthRGB") else None)
    res = {"arms": table, "runs": runs, "contrasts": C, "inference_condition_contrasts": inf, "v4b_screen_rule": screen,
           "decisions": dec, "sets": {k: len(v) for k, v in sets.items()}, "scoring": SC}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    summary(res, Path(a.out).with_name("summary.md"))


def decisions(C, screen, retained):
    th = SC["thresholds"]
    d = {}
    k = "RpBoth__vs__RpBoth_NoSynthRGB"
    if k in C:
        v, ci = C[k]["far/grasp"]
        seeds = [v2["far/grasp"] for v2 in C[k]["per_seed"].values()]
        noise = C.get("RpBoth_rerun__vs__RpBoth", {}).get("far/grasp")
        noise_pt = abs(noise[0]) if noise else None
        useful = v >= th["rgb_min_rel"] and ci[0] > 0 and all(s > 0 for s in seeds) and (noise_pt is None or abs(v) > noise_pt)
        harmful = v <= -th["rgb_min_rel"] and ci[1] < 0
        nr = C.get("RpBoth_NoSynthRGB__vs__R", {}).get("far/grasp")
        d["H1"] = {"generated_rgb_incremental_far_grasp": v, "ci95": ci, "per_seed": seeds, "replication_noise_far_grasp": noise_pt,
                   "verdict": "GENERATED_RGB_USEFUL" if useful else "GENERATED_RGB_HARMFUL" if harmful else "INCREMENTAL_VALUE_NOT_ESTABLISHED",
                   "no_rgb_arm_vs_R_far_grasp": nr,
                   "kinematic_supervision_alone_gives_screen_gain": bool(nr and nr[0] >= 0.05 and nr[1][0] > 0),
                   "retained_gain_fraction_far_grasp": retained("RpBoth", "RpBoth_NoSynthRGB") if retained else None}
    k1, k2, k3 = "RpPos_PhaseMask__vs__RpPos", "RpPos_PhaseMask__vs__RpPos_RandMask", "RpPos_RandMask__vs__RpPos"
    if k1 in C:
        h1 = C[k1]["all40/handover"]
        p1 = bool(h1[1][0] > 0 and screen["RpPos_PhaseMask"]["handover_degradation"] <= th["handover_guardrail"])
        g = screen["RpPos_PhaseMask"]
        p2 = bool(g["far_grasp_rel"] >= 0.05 and g["far_grasp_ci"][0] > 0)
        p3 = bool(k2 in C and C[k2]["all40/handover"][1][0] > 0)
        rnd = C.get(k3, {}).get("all40/handover")
        if p1 and p3:
            v = "PHASE_SPECIFIC_INTERFERENCE_SUPPORTED"
        elif p1 and rnd is not None and rnd[1][0] > 0:
            v = "HANDOVER_REDUCED_BUT_EXPLAINED_BY_SYNTHETIC_BUDGET"
        elif p1:
            v = "HANDOVER_REDUCED_PHASE_SPECIFICITY_NOT_ESTABLISHED"
        else:
            v = "PHASE_RESTRICTION_DOES_NOT_REDUCE_HANDOVER_REGRESSION"
        d["H2"] = {"P1_handover_reduced_vs_RpPos": p1, "handover_change_vs_RpPos": h1, "P2_far_grasp_preserved_vs_R": p2,
                   "P3_better_than_budget_control": p3, "phase_vs_random_handover": C.get(k2, {}).get("all40/handover"),
                   "random_vs_RpPos_handover": rnd, "phase_mask_screen": g, "verdict": v}
    return d


def summary(res, path):
    f = lambda x: "–" if not x else f"{100 * x[0]:+.1f}% [{100 * x[1][0]:+.1f}, {100 * x[1][1]:+.1f}]"  # noqa: E731
    L = ["# v4B.1 results (scripts/score_v4b1.py)", "", "## Original images, seed-averaged error (lower is better)", "",
         "| arm | far grasp | near grasp | approach | grasp | carry | handover | release | phase-balanced | gripper timing |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x, t in res["arms"].items():
        a, n, fa = t["all40"], t["near"], t["far"]
        L.append(f"| {x} | {fa['grasp']:.4f} | {n['grasp']:.4f} | {a['approach']:.4f} | {a['grasp']:.4f} | {a['carry']:.4f} | {a['handover']:.4f} | "
                 f"{a['release']:.4f} | {a['pb']:.4f} | {t['gripper_timing_frames']:.2f} |")
    L += ["", "## Paired contrasts (x vs ref; positive = x better), recording bootstrap 95% CI", "",
          "| contrast | far grasp | near grasp | approach | handover | carry | release | pb | per-seed far grasp | per-seed handover | timing Δ |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in res["contrasts"].items():
        ps = c["per_seed"]
        L.append(f"| {k.replace('__vs__', ' vs ')} | {f(c['far/grasp'])} | {f(c['near/grasp'])} | {f(c['all40/approach'])} | {f(c['all40/handover'])} | "
                 f"{f(c['all40/carry'])} | {f(c['all40/release'])} | {f(c['all40/pb'])} | "
                 + " / ".join(f"{100 * ps[s]['far/grasp']:+.1f}" for s in ps) + " | " + " / ".join(f"{100 * ps[s]['all40/handover']:+.1f}" for s in ps)
                 + f" | {c['gripper_timing_delta_frames']:+.2f} |")
    L += ["", "## v4B screening rule vs R (≥5% far grasp, CI>0, ≤2% handover degradation)", "", "| arm | far grasp | handover degradation | PASS |", "|---|---|---|---|"]
    for x, s in res["v4b_screen_rule"].items():
        L.append(f"| {x} | {100 * s['far_grasp_rel']:+.1f}% [{100 * s['far_grasp_ci'][0]:+.1f}, {100 * s['far_grasp_ci'][1]:+.1f}] | {100 * s['handover_degradation']:.1f}% | {s['PASS']} |")
    L += ["", "## Inference-time image conditions (same condition for both models)", "", "| contrast | condition | far grasp | grasp (40) | handover (40) |", "|---|---|---|---|---|"]
    for k, v in res["inference_condition_contrasts"].items():
        for c, d in v.items():
            L.append(f"| {k.replace('__vs__', ' vs ')} | {c} | {f(d['far/grasp'])} | {f(d['all40/grasp'])} | {f(d['all40/handover'])} |")
    L += ["", "## Frozen decisions", "", "```", json.dumps(res["decisions"], indent=1, default=float), "```"]
    path.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
