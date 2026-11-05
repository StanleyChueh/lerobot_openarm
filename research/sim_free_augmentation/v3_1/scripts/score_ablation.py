"""Score the Task A predictions exactly as declared in configs/vision_ablation.json (per-step phases, training-only
normalisation, recording-level bootstrap, near/far bins, gripper timing, prediction differences).

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/score_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import vision_ablation as va  # noqa: E402

OUT, PRED = va.OUT, va.PRED
ARMS = {"R": "R", "RpGp": "R+G+", "RpC": "R+C"}
rng = np.random.default_rng(0)


def train_std():
    from sfa3.recon3 import data
    A = np.concatenate([data.load(e, cams=()).action for e in range(10)])
    return A[:, va.ARM].std(0) + 1e-6


def closed(x, side):
    return np.clip(x / va.rb.GRIP_OPEN_RAW[side], 0, 1) < 0.5


def main():
    meta = json.loads((OUT / "run_meta.json").read_text())
    tg = np.load(OUT / "targets.npz")
    sd = train_std()
    eps = [int(e) for e in meta["episodes"]]
    ck = va.clicks()
    src_clicks = np.array([ck[e] for e in range(10)])
    gen_uv = []
    exp = json.loads((HERE.parent / "v3/outputs/dataset_export/export_checks.json").read_text())
    import sys as _s
    _s.path.insert(0, str(va.V3 / "scripts"))
    cam = va.rb.Camera()
    for E, vid, s, fam in exp["c_all"]:
        if fam in ("C_new_initial_position", "D_new_position_plus_handover"):
            z = np.load(va.V3 / f"outputs/ep{E}/candidates/{vid}__C__s{s}.npz")
            gen_uv.append(cam.project(z["table_new"][:3, 3][None])[0][0])
    gen_uv = np.array(gen_uv)
    near = {}
    for e in eps:
        d = float(np.min(np.linalg.norm(src_clicks - np.array(ck[e]), axis=1)))
        dg = float(np.min(np.linalg.norm(gen_uv - np.array(ck[e]), axis=1))) if len(gen_uv) else np.inf
        near[e] = {"dist_to_source_px": d, "bin": "near" if d <= 40 else "far", "far_covered_by_generation": bool(d > 40 and dg <= 20),
                   "dist_to_generated_px": dg}
    runs = sorted(p.stem for p in PRED.glob("*.npz"))
    per = {}
    for run in runs:
        z = np.load(PRED / f"{run}.npz")
        items = [tuple(map(int, x)) for x in z["items"]]
        per[run] = {}
        for cond in va.CONDS:
            P = z[cond].astype(np.float32)
            # per-step errors, per recording
            rec = {e: {ph: [] for ph in va.PHASES} for e in eps}
            start_phase = {e: {ph: [] for ph in ("approach", "carry", "handover", "retreat")} for e in eps}
            timing, gtp, gfp, gfn = [], 0, 0, 0
            diff = None
            if cond != "original":
                Po = z["original"].astype(np.float32)
                diff = {ph: [] for ph in va.PHASES}
            for n, (e, t) in enumerate(items):
                A = tg[f"A_{e}"]
                ph = tg[f"phase_{e}"]
                gt = A[t: t + 50]
                err = np.abs(P[n, :, va.ARM].T - gt[:, va.ARM]) / sd          # (50, 14)
                for k in range(len(gt)):
                    rec[e][ph[t + k]].append(err[k].mean())
                    if diff is not None:
                        diff[ph[t + k]].append((np.abs(P[n, k, va.ARM] - Po[n, k, va.ARM]) / sd).mean())
                sp = "approach" if ph[t] in ("approach",) else "carry" if ph[t] in ("grasp", "carry") else "handover" if ph[t] == "handover" else "retreat"
                start_phase[e][sp].append(err.mean())
                for side, i in (("R", va.GR), ("L", va.GL)):
                    pc, gc = closed(P[n, :25, i], side), closed(gt[:25, i], side)
                    gtp += int((pc & gc).sum()); gfp += int((pc & ~gc).sum()); gfn += int((~pc & gc).sum())
                    if gc.any() != gc.all():
                        kg = int(np.argmax(gc != gc[0]))
                        kp = int(np.argmax(pc != pc[0])) if (pc != pc[0]).any() else 25
                        timing.append(abs(kp - kg))
            pb_rec = {e: float(np.mean([np.mean(v) for v in rec[e].values() if v])) for e in eps}
            per_phase = {ph: float(np.mean([np.mean(rec[e][ph]) for e in eps if rec[e][ph]])) for ph in va.PHASES}
            per[run][cond] = {"pb_by_recording": pb_rec,
                              "pb_primary": float(np.mean([pb_rec[e] for e in va.H["primary"]])),
                              "pb_all": float(np.mean(list(pb_rec.values()))),
                              "per_phase_all": per_phase,
                              "per_phase_by_recording": {e: {ph: float(np.mean(v)) if v else None for ph, v in rec[e].items()} for e in eps},
                              "v3_start_phase_metric_primary": float(np.mean([np.mean([np.mean(v) for v in start_phase[e].values() if v]) for e in va.H["primary"]])),
                              "gripper_timing_frames": float(np.mean(timing)) if timing else None,
                              "gripper_precision": gtp / max(1, gtp + gfp), "gripper_recall": gtp / max(1, gtp + gfn),
                              "pred_diff_vs_original": {ph: float(np.mean(v)) for ph, v in diff.items() if v} if diff else None}
    # aggregate per arm (mean over 3 seeds), recording-level bootstrap of the intervention effect
    agg = {}
    for arm in ARMS:
        rs = [r for r in runs if r.rsplit("_s", 1)[0] == arm]
        agg[arm] = {}
        base = {e: np.mean([per[r]["original"]["pb_by_recording"][e] for r in rs]) for e in eps}
        for cond in va.CONDS:
            v = {e: np.mean([per[r][cond]["pb_by_recording"][e] for r in rs]) for e in eps}
            out = {}
            for setname, S in (("primary", va.H["primary"]), ("all40", eps),
                               ("near", [e for e in eps if near[e]["bin"] == "near"]), ("far", [e for e in eps if near[e]["bin"] == "far"]),
                               ("far_covered_by_generation", [e for e in eps if near[e]["far_covered_by_generation"]])):
                if not S:
                    continue
                x = np.array([v[e] for e in S])
                d = np.array([v[e] - base[e] for e in S])
                rel = d / np.array([base[e] for e in S])
                bs = [np.mean(rng.choice(d, len(d))) / np.mean([base[e] for e in S]) for _ in range(2000)]
                out[setname] = {"n_recordings": len(S), "pb_mean": float(x.mean()), "pb_se": float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else None,
                                "rel_change_vs_original": float(d.mean() / np.mean([base[e] for e in S])),
                                "rel_change_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            out["per_phase_all"] = {ph: float(np.mean([per[r][cond]["per_phase_all"][ph] for r in rs])) for ph in va.PHASES}
            out["gripper_timing_frames"] = float(np.mean([per[r][cond]["gripper_timing_frames"] for r in rs]))
            out["gripper_precision"] = float(np.mean([per[r][cond]["gripper_precision"] for r in rs]))
            out["gripper_recall"] = float(np.mean([per[r][cond]["gripper_recall"] for r in rs]))
            if cond != "original":
                out["pred_diff_vs_original"] = {ph: float(np.mean([per[r][cond]["pred_diff_vs_original"][ph] for r in rs])) for ph in va.PHASES}
            out["v3_start_phase_metric_primary"] = float(np.mean([per[r][cond]["v3_start_phase_metric_primary"] for r in rs]))
            agg[arm][cond] = out
        b, w = agg[arm]["blank"]["primary"], agg[arm]["wrong_episode"]["primary"]
        agg[arm]["image_dependence_CLEAR"] = bool(b["rel_change_vs_original"] >= 0.10 and b["rel_change_ci95"][0] > 0 and
                                                  w["rel_change_vs_original"] >= 0.10 and w["rel_change_ci95"][0] > 0)
    # arm comparison on ORIGINAL images by bin (does augmentation help where generation added coverage?)
    comp = {}
    for setname in ("primary", "all40", "near", "far", "far_covered_by_generation"):
        if setname in agg["R"]["original"]:
            comp[setname] = {a: agg[a]["original"][setname]["pb_mean"] for a in ARMS}
    res = {"per_run": per, "per_arm": agg, "near_far": near, "arm_comparison_original_images": comp,
           "normalisation": "training real episodes 0-9 action std"}
    (OUT / "scores.json").write_text(json.dumps(res, indent=1, default=float))
    # compact table
    lines = ["| arm | condition | primary (10 rec) pb err | Δ vs original [95% CI] | all 40 rec | near | far | grip timing | pred diff (approach/grasp/handover) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm, lab in ARMS.items():
        for cond in va.CONDS:
            o = agg[arm][cond]
            p = o["primary"]
            pdiff = o.get("pred_diff_vs_original")
            lines.append(f"| {lab} | {cond} | {p['pb_mean']:.4f} | {100 * p['rel_change_vs_original']:+.1f}% [{100 * p['rel_change_ci95'][0]:+.1f}, {100 * p['rel_change_ci95'][1]:+.1f}] | "
                         f"{o['all40']['pb_mean']:.4f} ({100 * o['all40']['rel_change_vs_original']:+.1f}%) | {o['near']['pb_mean']:.4f} | {o['far']['pb_mean']:.4f} | "
                         f"{o['gripper_timing_frames']:.2f} | " + (f"{pdiff['approach']:.3f}/{pdiff['grasp']:.3f}/{pdiff['handover']:.3f}" if pdiff else "–") + " |")
        lines.append(f"| {lab} | image dependence CLEAR (pre-declared rule) | {agg[arm]['image_dependence_CLEAR']} | | | | | | |")
    lines += ["", "| set | n recordings | R | R+G+ | R+C |", "|---|---|---|---|---|"]
    for s, v in comp.items():
        n = agg["R"]["original"][s]["n_recordings"]
        lines.append(f"| {s} | {n} | {v['R']:.4f} | {v['RpGp']:.4f} ({100 * (1 - v['RpGp'] / v['R']):+.1f}%) | {v['RpC']:.4f} ({100 * (1 - v['RpC'] / v['R']):+.1f}%) |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print({e: near[e]["bin"] + ("*" if near[e]["far_covered_by_generation"] else "") for e in eps})


if __name__ == "__main__":
    main()
