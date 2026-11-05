"""Dry run of the generation-feasibility pilot: builds every candidate plan of the pilot (A, B, C x 24 poses) and the
smoke stage, runs the offline checks, computes PLANNED hand-over descriptors, and validates the simulator-ground-truth
metric code on the local recorded GT episodes. NO simulation, NO GPU, NO network.

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/generation_feasibility/scripts/run_dry_run.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from gf import checks, config as C, descriptors as D, kin, plans, sim_metrics as M, sources  # noqa: E402

OUT = HERE / "outputs/dry_run"


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, cwd=C.REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def pilot_poses():
    poses = json.loads(C.EVAL_POSES.read_text())["poses"]
    pilot = poses[:C.PILOT_N]
    smoke = poses[-C.SMOKE_N:]
    assert not {p["trial"] for p in pilot} & {p["trial"] for p in smoke}
    return pilot, smoke


def desc_summary(d):
    return {"pass": d["pass"], "axial_sep": d["axial_sep"], "rel_azimuth_deg": d["rel_azimuth_deg"],
            "axial_L": d["axial_L"], "t_close_minus_arrive": d["t_close_minus_arrive"], "overlap": d["overlap"]}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plans").mkdir(exist_ok=True)
    K = kin.Kin()

    # ------------------------------------------------------------- sources
    eps = sources.load_sim10()
    S = [plans.segment_source(e, K) for e in eps]
    SD = [checks.source_descriptor(s) for s in S]
    scale = D.source_scale(SD)
    env = D.envelope(SD)
    src_table = []
    for s, d in zip(S, SD):
        src_table.append({"name": s.ep.name, "can0": s.ep.can0[:2], "t_sig": s.t_sig, "t_R_close": s.t_rc,
                          "t_L_approach": s.t_la, "t_giver_arrive": s.t_ga, "t_L_close": s.t_lc, "t_R_release": s.t_rr,
                          "T": s.ep.T, **desc_summary(d)})
    can0 = np.array([s.ep.can0[:2] for s in S])
    passes = np.array([d["pass"] for d in SD])
    source_stats = {
        "can0_std_xy_m": can0.std(0), "can0_range_xy_m": [can0.min(0), can0.max(0)],
        "pass_std_xyz_m": passes.std(0), "pass_range_xyz_m": [passes.min(0), passes.max(0)],
        "corr_can0x_passx": float(np.corrcoef(can0[:, 0], passes[:, 0])[0, 1]),
        "corr_can0y_passy": float(np.corrcoef(can0[:, 1], passes[:, 1])[0, 1]),
        "nn_scale": scale, "envelope": {k: v for k, v in env.items() if k != "pass"},
        "attachment_source": S[0].attach_src}
    # source self-capture under the nominal-attachment prediction (calibration of the plan-time capture check)
    self_capture = [checks.jaw_capture(np.linalg.inv(s.TL[s.t_lc]) @ s.can_at(s.t_lc)) for s in S]
    source_stats["receiver_capture_under_prediction"] = f"{sum(self_capture)}/{len(S)} (all 10 are GT successes)"

    # ------------------------------------------------------------- A self-replay sanity (A at each source's own pose)
    self_replay = []
    for k, s in enumerate(S):
        pA = plans.plan_A(s, s.ep.can0, idx=k)
        nv = D.novelty(checks.plan_descriptor(pA), SD, scale)
        self_replay.append({"source": s.ep.name, "pass_nn_m": nv["pass"]["nn_dist"], "path_nn_m": nv["path"]["nn_dist"],
                            "grasp_ratio": nv["grasp"]["ratio"], "inserted_steps": pA.T - s.ep.T})

    # ------------------------------------------------------------- candidates
    pilot, smoke = pilot_poses()
    rng_sens = np.random.default_rng(7)
    rows, manifest = [], []
    for stage, poses in (("pilot", pilot), ("smoke", smoke)):
        for i, pose in enumerate(poses):
            can = np.array([pose["x"], pose["y"], C.CAN_REST_Z])
            k = plans.nearest_source(S, can)
            src = S[k]
            params = plans.sample_params(np.random.default_rng(C.PILOT_PARAM_SEED + 1000 * (stage == "smoke") + i))
            cand = {"A": plans.plan_A(src, can, i), "B": plans.plan_phase(src, can, params, "B", i)}
            pC, rC, tried = checks.make_C(src, can, params, K, i)
            cand["C"] = pC
            for cond, p in cand.items():
                r = rC if cond == "C" else checks.check_plan(p, src, K)
                dsc = checks.plan_descriptor(p)
                nv = D.novelty(dsc, SD, scale)
                env_t = D.outside_envelope(dsc, env)
                sens = checks.attachment_sensitivity(p, src, rng_sens)
                tag = f"{stage}_{cond}_{i:02d}"
                np.savez_compressed(OUT / "plans" / f"{tag}.npz", target_R=p.target["R"], target_L=p.target["L"],
                                    grip_R=p.grip["R"], grip_L=p.grip["L"], can_pred=p.can_pred)
                row = {"stage": stage, "cond": cond, "i": i, "trial": pose["trial"], "region": pose["region"],
                       "can0": can[:2], "source": src.ep.name, "source_dxy_m": float(np.linalg.norm(can[:2] - src.ep.can0[:2])),
                       "status": p.status, "T": p.T, "checks": r, "descriptor": desc_summary(dsc),
                       "novelty": {f: nv[f]["ratio"] for f in nv}, "novelty_nn": {f: nv[f]["nn_dist"] for f in nv},
                       "outside_envelope": env_t, "attachment_capture_model": sens,
                       "c_tries": tried if cond == "C" else None}
                rows.append(row)
                manifest.append({"tag": tag, "stage": stage, "cond": cond, "i": i, "trial": pose["trial"],
                                 "can0": can[:2], "source": src.ep.name, "params": p.params, "marks": p.marks,
                                 "runtime": p.runtime, "status": p.status, "steps": p.T, "notes": p.notes,
                                 "plan_file": f"plans/{tag}.npz",
                                 "plan_sha256": sha256(OUT / "plans" / f"{tag}.npz")})

    # ------------------------------------------------------------- metric-code validation on recorded GT episodes
    gt = {}
    for nm in C.LOCAL_GT_HDF5:
        G = sources.load_gt_hdf5(nm)
        R = [M.rollout_metrics(e, K) for e in G]
        gt[nm] = {"n": len(G), "fps": G[0].fps,
                  **{k: sum(bool(r[k]) for r in R) for k in ("success_final", "stable_success", "slip_flag",
                                                              "lowered_to_pad_while_held", "off_table",
                                                              "arm_clearance_violation")},
                  "success_attr": sum(e.meta["success_attr"] for e in G),
                  "contract_violation_eps": sum(r["contract_violations"] > 0 for r in R),
                  "retention_steps_min": min(r["retention_steps"] for r in R),
                  "slip_receiver_m_median": float(np.median([r["slip_receiver_m"] for r in R])),
                  "slip_giver_m_median": float(np.median([r["slip_giver_m"] for r in R]))}

    # ------------------------------------------------------------- summary
    def agg(cond, stage="pilot"):
        R = [r for r in rows if r["cond"] == cond and r["stage"] == stage]
        acc = [r for r in R if r["status"] != "rejected"]
        ck = lambda key: sum(bool(r["checks"][key]) for r in R)   # noqa: E731
        return {"n": len(R), "accepted": len(acc),
                "checks_pass_all_ok": ck("all_ok"), "checks_pass_all_ok_C": ck("all_ok_C"),
                **{f"pass_{k}": ck(k) for k in ("ik_ok", "clearance_ok", "phase_order_ok", "receiver_in_grasp_cylinder",
                                               "receiver_jaw_capture", "receiver_on_can_body", "grasp_separation_ok",
                                               "receiver_after_giver_arrival")},
                "speed_violation_plans": sum(r["checks"]["speed_violations"] > 0 for r in R),
                "jump_violation_plans": sum(r["checks"]["jump_violations"] > 0 for r in R),
                "planned_novel_outside_envelope": sum(r["outside_envelope"]["novel"] for r in acc),
                "planned_novel_pass_far": sum(r["outside_envelope"]["pass_far"] for r in acc),
                "planned_novel_axial": sum(r["outside_envelope"]["axial_out"] for r in acc),
                "planned_novel_azimuth": sum(r["outside_envelope"]["azimuth_out"] for r in acc),
                "median_novelty_ratio": {f: float(np.median([r["novelty"][f] for r in acc])) if acc else None
                                         for f in D.FAMILIES},
                "attachment_capture_model_median": float(np.median([r["attachment_capture_model"] for r in R])),
                "sources_used": sorted({r["source"] for r in R}),
                "median_source_dxy_m": float(np.median([r["source_dxy_m"] for r in R])),
                "steps_median": float(np.median([r["T"] for r in R]))}

    summary = {
        "version": C.VERSION, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": git("rev-parse", "HEAD"), "isaaclab_commit": git("rev-parse", "--short", "HEAD", cwd=C.ISAACLAB),
        "isaaclab_dirty": bool(git("status", "--porcelain", cwd=C.ISAACLAB)),
        "source_dataset": f"{C.SIM10_REPO}@{C.SIM10_REV}", "n_sources": len(S),
        "pilot_trials": [p["trial"] for p in pilot], "smoke_trials": [p["trial"] for p in smoke],
        "source_stats": source_stats, "source_table": src_table, "A_self_replay": self_replay,
        "pilot": {c: agg(c) for c in "ABC"}, "smoke": {c: agg(c, "smoke") for c in "ABC"},
        "metric_validation_on_recorded_GT": gt,
        "simulation_run": False, "runtime_s": round(time.time() - t0, 1)}
    (OUT / "dry_run_summary.json").write_text(json.dumps(jsonable(summary), indent=1))
    (OUT / "candidates.jsonl").write_text("\n".join(json.dumps(jsonable(r)) for r in rows) + "\n")
    (OUT / "pilot_manifest.json").write_text(json.dumps(jsonable({
        "version": C.VERSION, "criteria": C.CRITERIA, "stop_rule": C.STOP_RULE, "episode_steps": C.EPISODE_STEPS,
        "source_dataset": f"{C.SIM10_REPO}@{C.SIM10_REV}", "candidates": manifest}), indent=1))
    figures(rows, SD, OUT)
    print(json.dumps(jsonable({"pilot": summary["pilot"], "smoke_accepted": {c: summary["smoke"][c]["accepted"] for c in "ABC"},
                               "source_stats": {k: source_stats[k] for k in ("pass_std_xyz_m", "can0_std_xy_m",
                                                                             "receiver_capture_under_prediction")},
                               "gt": gt, "runtime_s": summary["runtime_s"]}), indent=1))


def figures(rows, SD, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    sp = np.array([d["pass"] for d in SD])
    col = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    for cond in "ABC":
        R = [r for r in rows if r["cond"] == cond and r["stage"] == "pilot" and r["status"] != "rejected"]
        P = np.array([r["descriptor"]["pass"] for r in R])
        ax[0].scatter(P[:, 0], P[:, 1], s=18, c=col[cond], label=f"{cond} planned (n={len(R)})", alpha=0.7)
        ax[1].scatter([r["descriptor"]["axial_sep"] for r in R], [r["descriptor"]["rel_azimuth_deg"] for r in R],
                      s=18, c=col[cond], alpha=0.7, label=cond)
        ax[2].scatter([r["novelty"]["pass"] for r in R], [r["novelty"]["path"] for r in R], s=18, c=col[cond],
                      alpha=0.7, label=cond)
    ax[0].scatter(sp[:, 0], sp[:, 1], marker="*", s=120, c="k", label="10 sources")
    can0 = np.array([r["can0"] for r in rows if r["cond"] == "A" and r["stage"] == "pilot"])
    ax[0].scatter(can0[:, 0], can0[:, 1], marker="x", s=14, c="grey", label="pilot can starts")
    ax[0].set(xlabel="x (m)", ylabel="y (m)", title="pass point (can centre at the receiver close), planned")
    ax[1].scatter([d["axial_sep"] for d in SD], [d["rel_azimuth_deg"] for d in SD], marker="*", s=120, c="k")
    ax[1].set(xlabel="receiver - giver grasp along the can axis (m)", ylabel="relative azimuth (deg)",
              title="receiver grasp on the can, planned")
    ax[2].axvline(0.5, ls=":", c="k")
    ax[2].axhline(0.5, ls=":", c="k")
    ax[2].set(xlabel="pass novelty ratio", ylabel="receiver path novelty ratio", xscale="symlog", yscale="symlog",
              title="planned novelty vs sources (1 = source-to-source NN)")
    for a in ax:
        a.legend(fontsize=7)
    fig.suptitle("Dry run: PLANNED hand-overs (prediction, not simulation)")
    fig.tight_layout()
    fig.savefig(out / "planned_handover_diversity.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
