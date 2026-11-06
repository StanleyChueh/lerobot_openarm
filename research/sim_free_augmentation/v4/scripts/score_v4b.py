"""v4B scorer - implements the metrics and decision rules frozen in V4B_EXPERIMENT_SPEC.md (section 6). Do not change
after results exist; any later analysis goes to a separately named exploratory function/file.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/score_v4b.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V31 = HERE.parent / "v3_1"
sys.path.insert(0, str(V31 / "scripts"))
sys.path.insert(0, str(HERE))
import vision_ablation as va  # noqa: E402
from sfa4b import mixture as mx  # noqa: E402

OUT = HERE / "outputs/v4b"
PRED = OUT / "predictions"
PHASES = va.PHASES
SP = json.loads((HERE / "configs/v4b.json").read_text())["scoring"]
ARMS = ["R", "RpPos", "RpHO", "RpBoth", "RpPosAll"]
REF = ["v3_R", "v3_RpGp", "v3_RpC"]


def train_std():
    from sfa3.recon3 import data
    A = np.concatenate([data.load(e, cams=()).action for e in range(10)])
    return A[:, va.ARM].std(0) + 1e-6


def closed(x, side):
    return np.clip(x / va.rb.GRIP_OPEN_RAW[side], 0, 1) < 0.5


def bins(eps):
    ck = va.clicks()
    src = np.array([ck[e] for e in range(10)])
    cam = va.rb.Camera()
    sel = mx.selection()
    def pool_uv(arm):
        uv = []
        for E, keys in sel["keys"][arm].items():
            for k in keys:
                z = np.load(V31 / f"outputs/ep{E}/candidates/{k}.npz")
                uv.append(cam.project(z["table_new"][:3, 3][None])[0][0])
        return np.array(uv)
    gp, ga = pool_uv("RpPos"), pool_uv("RpPosAll")
    out = {}
    for e in eps:
        d = float(np.min(np.linalg.norm(src - np.array(ck[e]), axis=1)))
        out[e] = {"dist_to_source_px": d, "bin": "near" if d <= SP["near_px"] else "far",
                  "dist_to_RpPos_generated_px": float(np.min(np.linalg.norm(gp - np.array(ck[e]), axis=1))),
                  "dist_to_RpPosAll_generated_px": float(np.min(np.linalg.norm(ga - np.array(ck[e]), axis=1)))}
        out[e]["far_covered_RpPos"] = bool(out[e]["bin"] == "far" and out[e]["dist_to_RpPos_generated_px"] <= SP["covered_px"])
        out[e]["far_covered_RpPosAll"] = bool(out[e]["bin"] == "far" and out[e]["dist_to_RpPosAll_generated_px"] <= SP["covered_px"])
    return out


def score_run(z, tg, sd, eps, conds):
    items = [tuple(map(int, x)) for x in z["items"]]
    res = {}
    for cond in conds:
        if cond not in z.files:
            continue
        P = z[cond].astype(np.float32)
        Po = z["original"].astype(np.float32)
        rec = {e: {ph: [] for ph in PHASES} for e in eps}
        dif = {e: {ph: [] for ph in PHASES} for e in eps}
        timing = []
        for n, (e, t) in enumerate(items):
            A, ph = tg[f"A_{e}"], tg[f"phase_{e}"]
            gt = A[t:t + 50]
            err = np.abs(P[n, :len(gt)][:, va.ARM] - gt[:, va.ARM]) / sd
            dd = np.abs(P[n, :len(gt)][:, va.ARM] - Po[n, :len(gt)][:, va.ARM]) / sd
            for k in range(len(gt)):
                rec[e][ph[t + k]].append(err[k].mean())
                dif[e][ph[t + k]].append(dd[k].mean())
            for side, i in (("R", va.GR), ("L", va.GL)):
                pc, gc = closed(P[n, :25, i], side), closed(gt[:25, i], side)
                if gc.any() != gc.all():
                    kg = int(np.argmax(gc != gc[0]))
                    kp = int(np.argmax(pc != pc[0])) if (pc != pc[0]).any() else 25
                    timing.append(abs(kp - kg))
        per = {e: {ph: (float(np.mean(v)) if v else None) for ph, v in rec[e].items()} for e in eps}
        for e in eps:
            per[e]["pb"] = float(np.mean([v for k, v in per[e].items() if k in PHASES and v is not None]))
        res[cond] = {"by_recording": per, "gripper_timing_frames": float(np.mean(timing)) if timing else None,
                     "pred_diff_by_recording": {e: {ph: (float(np.mean(v)) if v else None) for ph, v in dif[e].items()} for e in eps}}
    return res


def seedavg(S, runs, cond, e, metric, key="by_recording"):
    v = [S[r][cond][key][e][metric] for r in runs]
    return None if any(x is None for x in v) else float(np.mean(v))


def rel_ci(x, b, rng, n):
    """relative change 1 - mean(x)/mean(b) (positive = x better) with a paired recording bootstrap."""
    x, b = np.asarray(x), np.asarray(b)
    pt = 1 - x.mean() / b.mean()
    idx = rng.integers(0, len(x), (n, len(x)))
    bs = 1 - x[idx].mean(1) / b[idx].mean(1)
    return float(pt), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    global PRED, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=str(PRED))
    ap.add_argument("--out_dir", default=str(OUT))
    a = ap.parse_args()
    PRED, OUT = Path(a.pred_dir), Path(a.out_dir)
    rng = np.random.default_rng(SP["bootstrap_seed"])
    meta = json.loads((PRED / "eval_meta.json").read_text())
    eps = [int(e) for e in meta["episodes"]]
    tg = np.load(PRED / "targets.npz")
    sd = train_std()
    B = bins(eps)
    conds = SP["conditions"]
    S = {}
    for p in sorted(PRED.glob("*.npz")):
        if p.stem == "targets":
            continue
        S[p.stem] = score_run(np.load(p), tg, sd, eps, conds)
    runs = {a: sorted(r for r in S if r.rsplit("_s", 1)[0] == a) for a in ARMS + REF}
    sets = {"all40": eps, "near": [e for e in eps if B[e]["bin"] == "near"], "far": [e for e in eps if B[e]["bin"] == "far"],
            "far_covered_RpPos": [e for e in eps if B[e]["far_covered_RpPos"]], "far_uncovered_RpPos": [e for e in eps if B[e]["bin"] == "far" and not B[e]["far_covered_RpPos"]],
            "primary10": [e for e in va.H["primary"] if e in eps]}
    metrics = PHASES + ["pb"]
    table = {}
    for a, rs in runs.items():
        if not rs:
            continue
        table[a] = {"runs": rs, "original": {}, "per_seed": {}}
        for sn, E in sets.items():
            vals = {m: [v for v in [seedavg(S, rs, "original", e, m) for e in E] if v is not None] for m in metrics}
            table[a]["original"][sn] = {m: (float(np.mean(v)) if v else None) for m, v in vals.items()}
            table[a]["original"][sn]["n_recordings"] = len(E)
        table[a]["gripper_timing_frames"] = float(np.mean([S[r]["original"]["gripper_timing_frames"] for r in rs]))
        table[a]["per_seed"] = {r: {"far_grasp": float(np.mean([S[r]["original"]["by_recording"][e]["grasp"] for e in sets["far"]])),
                                    "all40_handover": float(np.mean([S[r]["original"]["by_recording"][e]["handover"] for e in eps])),
                                    "all40_pb": float(np.mean([S[r]["original"]["by_recording"][e]["pb"] for e in eps]))} for r in rs}
    # ---- frozen comparisons vs R (paired over recordings, seed-averaged)
    comp = {}
    for a in ["RpPos", "RpHO", "RpBoth", "RpPosAll"] + REF:
        if not runs.get(a):
            continue
        c = {}
        for sn, E in sets.items():
            for m in metrics:
                xs = [(seedavg(S, runs[a], "original", e, m), seedavg(S, runs["R"], "original", e, m)) for e in E]
                xs = [(x, b) for x, b in xs if x is not None and b is not None]
                if len(xs) >= 2:
                    c[f"{sn}/{m}"] = rel_ci([x for x, _ in xs], [b for _, b in xs], rng, SP["bootstrap_n"])
        c["gripper_timing_delta_frames"] = table[a]["gripper_timing_frames"] - table["R"]["gripper_timing_frames"]
        c["per_seed_far_grasp_rel"] = [1 - table[a]["per_seed"][ra]["far_grasp"] / table["R"]["per_seed"][rr]["far_grasp"]
                                       for ra, rr in zip(runs[a], runs["R"])] if a in ARMS else None
        if "far/grasp" not in c or "all40/handover" not in c:
            c["decision"] = {"PASS": None, "reason": "too few recordings"}
            comp[a] = c
            continue
        rel, ci = c["far/grasp"]
        ho = -c["all40/handover"][0]                       # relative handover error increase
        c["decision"] = {"far_grasp_rel_improvement": rel, "ci95": ci, "handover_rel_degradation": ho,
                         "PASS": bool(rel >= SP["min_far_grasp_rel"] and ci[0] > 0 and ho <= SP["max_handover_degradation"])}
        comp[a] = c
    # hypothesis B: handover augmentation hurts handover prediction
    hyp = {}
    if "RpHO" in comp:
        pt, ci = comp["RpHO"]["all40/handover"]
        hyp["B_handover_aug_hurts_handover"] = {"rel_change_(+=better)": pt, "ci95": ci, "established": bool(ci[1] < 0)}
    if "RpPos" in comp:
        hyp["A_position_aug_improves_far_grasp"] = comp["RpPos"]["decision"]
    # ---- B4 visual dependence (relative error increase vs the same model's original images)
    vis = {}
    for a, rs in runs.items():
        if not rs:
            continue
        vis[a] = {}
        for cond in conds:
            if cond == "original" or cond not in S[rs[0]]:
                continue
            vis[a][cond] = {}
            for sn in ("all40", "far"):
                for m in ("approach", "grasp", "handover", "pb"):
                    xs = [(seedavg(S, rs, cond, e, m), seedavg(S, rs, "original", e, m)) for e in sets[sn]]
                    xs = [(x, b) for x, b in xs if x is not None and b is not None]
                    if len(xs) < 2:
                        continue
                    pt, ci = rel_ci([x for x, _ in xs], [b for _, b in xs], rng, SP["bootstrap_n"])
                    vis[a][cond][f"{sn}/{m}"] = [-pt, [-ci[1], -ci[0]]]        # + = error increase
            vis[a][cond]["pred_diff_all40"] = {ph: float(np.mean([v for v in [seedavg(S, rs, cond, e, ph, "pred_diff_by_recording") for e in eps] if v is not None]))
                                               for ph in PHASES}
        if "object_only" in vis[a] and "background" in vis[a]:
            xs = [(seedavg(S, rs, "object_only", e, "grasp"), seedavg(S, rs, "background", e, "grasp"), seedavg(S, rs, "original", e, "grasp")) for e in eps]
            xs = [x for x in xs if None not in x]
            d = np.array([(o - b) / g for o, b, g in xs])
            idx = rng.integers(0, len(d), (SP["bootstrap_n"], len(d)))
            vis[a]["object_minus_background_grasp_all40"] = [float(d.mean()), [float(np.percentile(d[idx].mean(1), 2.5)), float(np.percentile(d[idx].mean(1), 97.5))]]
    res = {"bins": B, "sets": sets, "arms": table, "vs_R": comp, "hypotheses": hyp, "visual_dependence": vis,
           "scoring_config": SP, "normalisation": "per-joint action std of real_10 episodes 0-9"}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    write_summary(res)


def write_summary(res):
    T, C = res["arms"], res["vs_R"]
    L = ["# v4B results (generated by scripts/score_v4b.py)", "",
         "## Original images, seed-averaged (normalised per-step arm-action error; lower is better)", "",
         "| arm | far grasp | near grasp | approach | grasp | carry | handover | release | phase-balanced | far pb | gripper timing (frames) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, t in T.items():
        o = t["original"]
        L.append(f"| {a} | {o['far']['grasp']:.4f} | {o['near']['grasp']:.4f} | {o['all40']['approach']:.4f} | {o['all40']['grasp']:.4f} | {o['all40']['carry']:.4f} | "
                 f"{o['all40']['handover']:.4f} | {o['all40']['release']:.4f} | {o['all40']['pb']:.4f} | {o['far']['pb']:.4f} | {t['gripper_timing_frames']:.2f} |")
    L += ["", "## Relative change vs R (positive = better than R), recording bootstrap 95% CI", "",
          "| arm | far grasp | near grasp | approach (40) | handover (40) | carry (40) | release (40) | pb (40) | far-covered grasp | far-uncovered grasp | per-seed far grasp | timing Δ | PASS |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x: f"{100 * x[0]:+.1f}% [{100 * x[1][0]:+.1f}, {100 * x[1][1]:+.1f}]" if x else "–"  # noqa: E731
    for a, c in C.items():
        ps = c.get("per_seed_far_grasp_rel")
        L.append(f"| {a} | {f(c.get('far/grasp'))} | {f(c.get('near/grasp'))} | {f(c.get('all40/approach'))} | {f(c.get('all40/handover'))} | {f(c.get('all40/carry'))} | "
                 f"{f(c.get('all40/release'))} | {f(c.get('all40/pb'))} | {f(c.get('far_covered_RpPos/grasp'))} | {f(c.get('far_uncovered_RpPos/grasp'))} | "
                 + (" / ".join(f"{100 * x:+.1f}%" for x in ps) if ps else "–") + f" | {c['gripper_timing_delta_frames']:+.2f} | {c['decision']['PASS']} |")
    L += ["", "## Visual dependence (relative error INCREASE vs the model's own original images, all 40 recordings)", "",
          "| arm | condition | approach | grasp | handover | phase-balanced |", "|---|---|---|---|---|---|"]
    for a, v in res["visual_dependence"].items():
        for cond, d in v.items():
            if isinstance(d, dict) and "all40/grasp" in d:
                L.append(f"| {a} | {cond} | {f(d['all40/approach'])} | {f(d['all40/grasp'])} | {f(d['all40/handover'])} | {f(d['all40/pb'])} |")
        if "object_minus_background_grasp_all40" in v:
            x = v["object_minus_background_grasp_all40"]
            L.append(f"| {a} | object − background (grasp) | | {100 * x[0]:+.1f}% [{100 * x[1][0]:+.1f}, {100 * x[1][1]:+.1f}] | | |")
    L += ["", "Sets: " + ", ".join(f"{k}={len(v)}" for k, v in res["sets"].items())]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
