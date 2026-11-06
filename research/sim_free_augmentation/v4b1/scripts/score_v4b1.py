"""v4B.1 scorer: V4B1_EXPERIMENT_SPEC.md section 7 as corrected by Amendment A1 (fail-closed study contract).

Reuses the v4B per-recording scoring (v4/scripts/score_v4b.py: score_run, train_std, bins, rel_ci) and the frozen v4B
prediction archives of the references R / RpBoth / RpPos.

final mode (default): refuses to produce any output unless the study is complete and matches its contract -
  * exactly the registered seeds for every registered new arm and every frozen reference;
  * only registered new-arm archive names in --pred_dir (frozen reference names cannot be shadowed);
  * frozen archives, checkpoints and normalisers match their recorded sha256; optimizer step counters = 1000;
  * new runs verified by scripts/verify_runs.py (all_pass, same checkpoints that were evaluated);
  * evaluation metadata, targets and the ordered (recording, frame) item list identical to the frozen evaluation;
  * all six image conditions present, every prediction array finite with shape (1524, 50, 16).
smoke mode (--smoke): descriptive tables only, decisions = SMOKE_NO_SCIENTIFIC_DECISIONS; frozen names still protected.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/score_v4b1.py [--pred_dir P] [--smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V4 = HERE.parent / "v4"
sys.path.insert(0, str(V4 / "scripts"))
import score_v4b as sv  # noqa: E402

CFG = json.loads((HERE / "configs/v4b1.json").read_text())
SC = CFG["scoring"]
SEEDS = list(CFG["seeds"])
NEW = list(CFG["arms"])
FROZEN = list(CFG["frozen_references"])
METRICS = ["approach", "grasp", "carry", "handover", "release", "pb"]
VPRED = V4 / "outputs/v4b/predictions"
N_ITEMS, CHUNK, ADIM = 1524, 50, 16
SEED_KEYS = {f"s{s}" for s in SEEDS}


class StudyContractError(RuntimeError):
    """the inputs do not form the registered, complete, untampered study; no scientific output is produced."""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def parse(stem):
    m = re.fullmatch(r"(.+)_s(\d+)", stem)
    if not m:
        raise StudyContractError(f"unparseable archive name {stem!r}")
    return m.group(1), int(m.group(2))


def check_names(stems, smoke=False):
    """new-study archive names: whitelist, no frozen-reference shadowing, exact registered seed set (final)."""
    have = {}
    for st in stems:
        arm, seed = parse(st)
        if arm in FROZEN or arm.startswith("v3_") or arm in ("RpHO", "RpPosAll"):
            raise StudyContractError(f"{st}: name collides with a frozen v4B reference; new predictions may not replace it")
        if arm not in NEW:
            raise StudyContractError(f"{st}: arm {arm!r} is not registered in configs/v4b1.json arms")
        if seed not in SEEDS:
            raise StudyContractError(f"{st}: seed {seed} is not registered {SEEDS}")
        have.setdefault(arm, set()).add(seed)
    if not smoke:
        check_complete({a: have.get(a, set()) for a in NEW})
    return have


def check_complete(seeds_by_arm):
    for arm, s in seeds_by_arm.items():
        if set(s) != set(SEEDS):
            raise StudyContractError(f"{arm}: seeds {sorted(s)} != registered {SEEDS}; incomplete studies yield no verdict")


def check_archive(name, z, ref_items, conds):
    if "items" not in z.files or not np.array_equal(z["items"], ref_items):
        raise StudyContractError(f"{name}: (recording, frame) items differ from the frozen evaluation order")
    for c in conds:
        if c not in z.files:
            raise StudyContractError(f"{name}: missing image condition {c!r}")
        P = z[c]
        if P.shape != (N_ITEMS, CHUNK, ADIM):
            raise StudyContractError(f"{name}/{c}: shape {P.shape} != {(N_ITEMS, CHUNK, ADIM)}")
        if not np.isfinite(P.astype(np.float32)).all():
            raise StudyContractError(f"{name}/{c}: non-finite predictions")


def check_eval_contract(pred_dir, frozen_meta, frozen_targets):
    meta = json.loads((pred_dir / "eval_meta.json").read_text())
    for k in ("episodes", "conditions", "pool_revision", "partner"):
        if meta.get(k) != frozen_meta.get(k):
            raise StudyContractError(f"eval_meta.{k} differs from the frozen v4B evaluation")
    if list(meta["conditions"]) != list(SC["conditions"]):
        raise StudyContractError("evaluated conditions differ from the registered six")
    t = np.load(pred_dir / "targets.npz")
    if sorted(t.files) != sorted(frozen_targets.files) or any(not np.array_equal(t[k], frozen_targets[k]) for k in t.files):
        raise StudyContractError("evaluation targets differ from the frozen v4B targets")


def check_frozen_refs():
    man = json.loads((V4 / "outputs/v4b/predictions_manifest.json").read_text())
    ck = json.loads((VPRED / "checkpoint_hashes.json").read_text())
    out = {}
    for arm in FROZEN:
        for s in SEEDS:
            n = f"{arm}_s{s}"
            if sha(VPRED / f"{n}.npz") != man[f"{n}.npz"]:
                raise StudyContractError(f"frozen archive {n}.npz hash mismatch")
            root = V4 / CFG["frozen_references"][arm].format(seed=s).replace("../v4/", "") / "checkpoints/last"
            pm = root / "pretrained_model"
            if sha(pm / "model.safetensors") != ck[n]["model.safetensors"]:
                raise StudyContractError(f"frozen checkpoint {n} model hash mismatch")
            if sha(next(pm.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))) != ck[n]["normalizer"]:
                raise StudyContractError(f"frozen checkpoint {n} normalizer hash mismatch")
            step = json.loads((root / "training_state/training_step.json").read_text())["step"]
            if step != CFG["steps"]:
                raise StudyContractError(f"frozen checkpoint {n} optimizer step {step} != {CFG['steps']}")
            out[n] = {"archive": man[f"{n}.npz"], "model": ck[n]["model.safetensors"], "normalizer": ck[n]["normalizer"], "step": step}
    return out


def check_new_runs(pred_dir, verification_path):
    ver = json.loads(Path(verification_path).read_text())
    if not ver.get("all_pass") or ver.get("steps") != CFG["steps"]:
        raise StudyContractError("run verification (scripts/verify_runs.py) did not pass for the registered step count")
    ck = json.loads((pred_dir / "checkpoint_hashes.json").read_text())
    for arm in NEW:
        for s in SEEDS:
            n = f"{arm}_s{s}"
            v = ver["runs"].get(n)
            if not v or not v["pass"]:
                raise StudyContractError(f"{n}: not verified")
            if n not in ck or ck[n]["model.safetensors"] != v["model_sha256"] or ck[n]["normalizer"] != v["normalizer_sha256"]:
                raise StudyContractError(f"{n}: evaluated checkpoint differs from the verified training output")
    return {"verification_sha256": sha(verification_path)}


def load_study(pred_dir, smoke=False, verification=None):
    pred_dir = Path(pred_dir).resolve()
    if pred_dir == VPRED.resolve():
        raise StudyContractError("--pred_dir must not be the frozen v4B prediction directory")
    conds = SC["conditions"]
    frozen_meta = json.loads((VPRED / "eval_meta.json").read_text())
    tg = np.load(VPRED / "targets.npz")
    eps = [int(e) for e in frozen_meta["episodes"]]
    stems = sorted(p.stem for p in pred_dir.glob("*.npz") if p.stem != "targets")
    have = check_names(stems, smoke)
    prov = {}
    if not smoke:
        check_eval_contract(pred_dir, frozen_meta, tg)
        prov["frozen_references"] = check_frozen_refs()
        prov["new_runs"] = check_new_runs(pred_dir, verification or HERE / "outputs/run_verification.json")
    ref_items = np.load(VPRED / "R_s0.npz")["items"]
    sd = sv.train_std()
    S = {}
    for arm in FROZEN:
        for s in SEEDS:
            z = np.load(VPRED / f"{arm}_s{s}.npz")
            check_archive(f"{arm}_s{s}", z, ref_items, conds)
            S[f"{arm}_s{s}"] = sv.score_run(z, tg, sd, eps, conds)
    for st in stems:
        z = np.load(pred_dir / f"{st}.npz")
        check_archive(st, z, ref_items, conds)
        S[st] = sv.score_run(z, tg, sd, eps, conds)
    runs = {a: [f"{a}_s{s}" for s in SEEDS] for a in FROZEN} | {a: [f"{a}_s{s}" for s in sorted(v)] for a, v in have.items()}
    return S, runs, eps, prov


def analyse(S, runs, eps, smoke=False):
    rng = np.random.default_rng(SC["bootstrap_seed"])
    B = sv.bins(eps)
    sets = {"all40": eps, "near": [e for e in eps if B[e]["bin"] == "near"], "far": [e for e in eps if B[e]["bin"] == "far"]}

    def rec(x, cond, e, m, seeds=None):
        rs = runs[x] if seeds is None else [f"{x}_s{s}" for s in seeds]
        v = [S[r][cond]["by_recording"][e][m] for r in rs]
        return None if any(q is None for q in v) else float(np.mean(v))

    def contrast(x, ref, sn, m, cond="original", seeds=None):
        xs = [(rec(x, cond, e, m, seeds), rec(ref, cond, e, m, seeds)) for e in sets[sn]]
        xs = [(p, q) for p, q in xs if p is not None and q is not None]
        if len(xs) < 2:
            raise StudyContractError(f"{x} vs {ref} {sn}/{m}: fewer than two defined recordings")
        return sv.rel_ci([p for p, _ in xs], [q for _, q in xs], rng, SC["bootstrap_n"])

    table = {}
    for x, rs in runs.items():
        t = {sn: {m: float(np.mean([v for v in (rec(x, "original", e, m) for e in E) if v is not None])) for m in METRICS} for sn, E in sets.items()}
        t["gripper_timing_frames"] = float(np.mean([S[r]["original"]["gripper_timing_frames"] for r in rs]))
        t["per_seed"] = {r: {"far_grasp": float(np.mean([S[r]["original"]["by_recording"][e]["grasp"] for e in sets["far"]])),
                             "all40_handover": float(np.mean([S[r]["original"]["by_recording"][e]["handover"] for e in eps])),
                             "all40_pb": float(np.mean([S[r]["original"]["by_recording"][e]["pb"] for e in eps]))} for r in rs}
        table[x] = t

    def seeds_of(x):
        return [int(r.rsplit("_s", 1)[1]) for r in runs[x]]

    def full(x, ref, cond="original"):
        out = {f"{sn}/{m}": contrast(x, ref, sn, m, cond) for sn in sets for m in METRICS}
        common = [s for s in seeds_of(x) if s in seeds_of(ref)]
        out["per_seed"] = {f"s{s}": {k: contrast(x, ref, sn, m, cond, [s])[0] for k, (sn, m) in
                                     {"far/grasp": ("far", "grasp"), "all40/handover": ("all40", "handover"), "all40/pb": ("all40", "pb")}.items()}
                           for s in common}
        out["gripper_timing_delta_frames"] = table[x]["gripper_timing_frames"] - table[ref]["gripper_timing_frames"]
        return out

    C = {f"{x}__vs__{ref}": full(x, ref) for x, ref in SC["contrasts"] if x in runs and ref in runs}
    inf = {f"{x}__vs__{ref}": {c: {k: contrast(x, ref, sn, m, c) for k, (sn, m) in
                                    {"far/grasp": ("far", "grasp"), "all40/grasp": ("all40", "grasp"), "all40/handover": ("all40", "handover")}.items()}
                               for c in SC["conditions"]}
           for x, ref in SC["inference_condition_contrasts"] if x in runs and ref in runs}
    screen = {}
    for x in runs:
        if x == "R":
            continue
        g, h = contrast(x, "R", "far", "grasp"), contrast(x, "R", "all40", "handover")
        screen[x] = {"far_grasp_rel": g[0], "far_grasp_ci": g[1], "handover_degradation": -h[0], "handover_degradation_ci": [-h[1][1], -h[1][0]],
                     "PASS": bool(g[0] >= SC["thresholds"]["far_grasp_screen_rel"] and g[1][0] > 0 and -h[0] <= SC["thresholds"]["handover_guardrail"])}

    def retained(x_full, x_part, sn="far", m="grasp"):
        r, f, p = (np.array([rec(a, "original", e, m) for e in sets[sn]]) for a in ("R", x_full, x_part))
        pt = (r.mean() - p.mean()) / (r.mean() - f.mean())
        idx = rng.integers(0, len(r), (SC["bootstrap_n"], len(r)))
        bs = (r[idx].mean(1) - p[idx].mean(1)) / (r[idx].mean(1) - f[idx].mean(1))
        return [float(pt), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]

    if smoke:
        dec = {"status": "SMOKE_NO_SCIENTIFIC_DECISIONS"}
    else:
        dec = decisions(C, screen, retained("RpBoth", "RpBoth_NoSynthRGB"))
    return {"arms": table, "runs": runs, "contrasts": C, "inference_condition_contrasts": inf, "v4b_screen_rule": screen,
            "decisions": dec, "sets": {k: len(v) for k, v in sets.items()}, "scoring": SC}


# ----------------------------------------------------------------------------------------------- decisions (A1)
def _need(C, k, metric):
    if k not in C or metric not in C[k]:
        raise StudyContractError(f"required contrast {k} [{metric}] missing")
    ps = C[k].get("per_seed", {})
    if set(ps) != SEED_KEYS:
        raise StudyContractError(f"{k}: per-seed values for {sorted(ps)} != registered {sorted(SEED_KEYS)}")
    pt, ci = C[k][metric]
    return float(pt), [float(ci[0]), float(ci[1])], [float(ps[s][metric]) for s in sorted(ps)]


def _noise(C, name):
    r = SC["replication_noise"][name]
    pt, _, _ = _need(C, r["contrast"], r["metric"])
    return abs(pt)


def _effect(pt, ci, noise):
    """direction with interval and replication guard; 'not_resolved' never means equivalence or no effect."""
    beyond = abs(pt) > noise
    if ci[0] > 0 and beyond:
        return "improved"
    if ci[1] < 0 and beyond:
        return "worsened"
    return "not_resolved"


def decisions(C, screen, retained):
    th = SC["thresholds"]
    d = {"_doc": "Amendment A1 decision contract. 'not_resolved' / NOT_ESTABLISHED never means equivalence or no effect."}
    # ---------------- H1
    v, ci, seeds = _need(C, "RpBoth__vs__RpBoth_NoSynthRGB", "far/grasp")
    noise = _noise(C, "H1")
    reasons = []
    if -th["rgb_min_rel"] < v < th["rgb_min_rel"]:
        reasons.append(f"|effect| below the {th['rgb_min_rel']:.0%} threshold")
    if ci[0] <= 0 <= ci[1]:
        reasons.append("95% CI includes 0")
    if not (all(s > 0 for s in seeds) or all(s < 0 for s in seeds)):
        reasons.append("per-seed signs disagree")
    if abs(v) <= noise:
        reasons.append("effect within replication noise |RpBoth_rerun vs RpBoth|")
    useful = v >= th["rgb_min_rel"] and ci[0] > 0 and all(s > 0 for s in seeds) and abs(v) > noise
    harmful = v <= -th["rgb_min_rel"] and ci[1] < 0 and abs(v) > noise
    nr, nr_ci, _ = _need(C, "RpBoth_NoSynthRGB__vs__R", "far/grasp")
    sens = _need(C, "RpBoth_rerun__vs__RpBoth_NoSynthRGB", "far/grasp")
    d["H1"] = {"question": "generated synthetic RGB vs SmolVLA missing-camera treatment on synthetic rows (real RGB unchanged)",
               "generated_rgb_incremental_far_grasp": v, "ci95": ci, "per_seed": seeds, "replication_noise_far_grasp": noise,
               "exceeds_replication_noise": abs(v) > noise,
               "sensitivity_rerun_vs_NoSynthRGB_far_grasp": sens[:2],
               "verdict": "GENERATED_RGB_USEFUL" if useful else "GENERATED_RGB_HARMFUL" if harmful else "INCREMENTAL_VALUE_NOT_ESTABLISHED",
               "not_established_reasons": [] if (useful or harmful) else reasons,
               "no_rgb_arm_vs_R_far_grasp": [nr, nr_ci],
               "synthetic_state_action_without_synthetic_rgb_passes_far_grasp_screen_part": bool(nr >= th["far_grasp_screen_rel"] and nr_ci[0] > 0),
               "retained_gain_fraction_far_grasp": retained}
    # ---------------- H2
    nh, ng = _noise(C, "H2_handover"), _noise(C, "H2_grasp")
    h, h_ci, h_seeds = _need(C, "RpPos_PhaseMask__vs__RpPos", "all40/handover")
    hm, hm_ci, _ = _need(C, "RpPos_MatchedRandMask__vs__RpPos", "all40/handover")
    dd, dd_ci, dd_seeds = _need(C, "RpPos_PhaseMask__vs__RpPos_MatchedRandMask", "all40/handover")
    g, g_ci, _ = _need(C, "RpPos_PhaseMask__vs__RpPos", "far/grasp")
    hs = _need(C, "RpPos_PhaseMask__vs__RpPos_rerun", "all40/handover")
    ps = screen["RpPos_PhaseMask"]
    A = _effect(h, h_ci, nh)
    A_m = _effect(hm, hm_ci, nh)
    D = _effect(dd, dd_ci, nh)
    B_within = bool(ps["handover_degradation"] <= th["handover_guardrail"])
    C_pres = bool(ps["far_grasp_rel"] >= th["far_grasp_screen_rel"] and ps["far_grasp_ci"][0] > 0)
    G = _effect(g, g_ci, ng)
    if A == "improved" and D == "improved":
        mech = "PHASE_PLACED_SUPERVISION_REDUCES_HANDOVER_ERROR_BEYOND_MATCHED_RANDOM"
    elif A == "improved" and A_m == "improved":
        mech = "BOTH_MASKS_REDUCE_HANDOVER_ERROR_PHASE_SPECIFICITY_NOT_ESTABLISHED"
    elif A == "improved":
        mech = "HANDOVER_REDUCED_PHASE_SPECIFICITY_NOT_ESTABLISHED"
    elif A == "worsened":
        mech = "PHASE_MASK_INCREASES_HANDOVER_ERROR"
    else:
        mech = "HANDOVER_REDUCTION_NOT_ESTABLISHED"
    d["H2"] = {
        "question": "does phase-placed synthetic target supervision beat the per-row matched random control and reduce handover harm "
                    "without destroying the far-grasp gain? (estimand: target placement within matched active contexts)",
        "A_handover_improvement_vs_RpPos": {"rel_change": h, "ci95": h_ci, "per_seed": h_seeds, "status": A,
                                            "sensitivity_vs_RpPos_rerun": hs[:2],
                                            "matched_random_vs_RpPos": {"rel_change": hm, "ci95": hm_ci, "status": A_m}},
        "B_residual_handover_degradation_vs_R": {"degradation": ps["handover_degradation"], "ci95": ps["handover_degradation_ci"],
                                                 "guardrail": th["handover_guardrail"], "within_guardrail_point_estimate": B_within},
        "C_far_grasp_preservation": {"vs_R": [ps["far_grasp_rel"], ps["far_grasp_ci"]], "P2_preserved_vs_R": C_pres,
                                     "vs_RpPos": [g, g_ci], "vs_RpPos_status": G, "far_grasp_reduced_vs_RpPos": G == "worsened"},
        "D_phase_specific_increment_vs_matched_random": {"rel_change": dd, "ci95": dd_ci, "per_seed": dd_seeds, "status": D},
        "E_statistical_resolution": {"replication_noise_handover": nh, "replication_noise_far_grasp": ng,
                                     "A_exceeds_noise": abs(h) > nh, "D_exceeds_noise": abs(dd) > nh,
                                     "A_ci_width": h_ci[1] - h_ci[0], "D_ci_width": dd_ci[1] - dd_ci[0]},
        "original_P1_reduction_and_guardrail": bool(A == "improved" and B_within),
        "original_P2_far_grasp_preserved": C_pres,
        "original_P3_beats_control": D == "improved",
        "mechanism_label": mech,
        "Q2_answer": "YES" if (A == "improved" and D == "improved" and B_within and C_pres) else "NOT_SUPPORTED_BY_THIS_SCREEN",
    }
    return d


def summary(res, path):
    f = lambda x: "–" if not x else f"{100 * x[0]:+.1f}% [{100 * x[1][0]:+.1f}, {100 * x[1][1]:+.1f}]"  # noqa: E731
    L = ["# v4B.1 results (scripts/score_v4b1.py, Amendment A1 contract)", "",
         f"Mode: **{res['mode']}**. Open-loop action error on development recordings 10–49 (not robot success).", "",
         "## Original images, seed-averaged error (lower is better)", "",
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
        L.append(f"| {x} | {100 * s['far_grasp_rel']:+.1f}% [{100 * s['far_grasp_ci'][0]:+.1f}, {100 * s['far_grasp_ci'][1]:+.1f}] | "
                 f"{100 * s['handover_degradation']:.1f}% [{100 * s['handover_degradation_ci'][0]:+.1f}, {100 * s['handover_degradation_ci'][1]:+.1f}] | {s['PASS']} |")
    L += ["", "## Inference-time image conditions (same condition for both models)", "", "| contrast | condition | far grasp | grasp (40) | handover (40) |", "|---|---|---|---|---|"]
    for k, v in res["inference_condition_contrasts"].items():
        for c, d in v.items():
            L.append(f"| {k.replace('__vs__', ' vs ')} | {c} | {f(d['far/grasp'])} | {f(d['all40/grasp'])} | {f(d['all40/handover'])} |")
    L += ["", "## Decisions", "", "```", json.dumps(res["decisions"], indent=1, default=float), "```"]
    path.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=str(HERE / "outputs/predictions"))
    ap.add_argument("--smoke", action="store_true", help="descriptive only; no scientific decisions; incomplete inputs allowed")
    ap.add_argument("--verification", default=str(HERE / "outputs/run_verification.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else HERE / ("results_smoke.json" if a.smoke else "results.json")
    S, runs, eps, prov = load_study(a.pred_dir, a.smoke, a.verification)
    res = analyse(S, runs, eps, a.smoke) | {"provenance_checks": prov, "mode": "smoke" if a.smoke else "final"}
    out.write_text(json.dumps(res, indent=1, default=float))
    summary(res, out.with_name("summary_smoke.md" if a.smoke else "summary.md"))


if __name__ == "__main__":
    main()
