"""Paired comparison of the two matched evaluations (same 100 can poses, 330 simulated steps each).

    cd ~/Stanley_ws/lerobot_openarm
    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv-v2/bin/python research/generated_data_diagnosis/matched_evaluation/analyze.py \
        [--a results_manual_10 --b results_generated_500 --out figures --json comparison.json]

Reads each results folder (robot_client.py --results_dir: run_meta*.json, trials.jsonl, steps/), checks that the
two runs are comparable (same pose file, same poses applied, same settings, different checkpoints, every trial
complete with exactly the step budget), and writes comparison.json + the figures. CPU only.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2

HERE = Path(__file__).resolve().parent
X_BINS = [(-1.0, 0.203, "x<0.203"), (0.203, 0.27, "0.203-0.27"), (0.27, 1.0, "x>=0.27")]
Y_BINS = [(-1.0, -0.01, "y<-0.01"), (-0.01, 0.03, "-0.01..0.03"), (0.03, 1.0, "y>=0.03")]
METRICS = [("success_any", "any-step success (original)"), ("stable", "stable hand-over"),
           ("holding_end", "holding at step 330")]
COLORS = {"a": "#2a78d6", "b": "#eb6834", "both": "#1baf7a", "neither": "#9a9a96"}  # dataviz reference slots 1-3
HISTORICAL = {  # EVALUATION_COMPARISON.md (random, unlogged poses; 15 s wall clock)
    "success_any": {"a": 49, "b": 47, "n": 100},
    "holding_last_frame": {"a": 3, "b": 47, "n": 100},
    "x<0.203": {"a": (25, 43), "b": (6, 37)},
    "0.203-0.27": {"a": (16, 46), "b": (30, 47)},
    "x>=0.27": {"a": (8, 11), "b": (11, 16)},
}
SETTINGS_IGNORED = ("--pretrained_name_or_path", "--results_dir", "--record_repo_id", "--record_root", "--resume",
                    "--num_episodes")
BLOCK, CMD_CLOSED = 0.15, 0.6  # the diagnosis's gripper hold detector (scripts/eval_analysis.py, validated)


# -- loading and checks ------------------------------------------------------------------------------------------
def load_run(root: Path) -> dict:
    metas = [json.loads(p.read_text()) for p in sorted(root.glob("run_meta*.json"))]
    trials = [json.loads(line) for line in (root / "trials.jsonl").read_text().splitlines() if line.strip()]
    final = {}
    for t in trials:  # the last complete attempt of each trial
        if t["status"] == "complete":
            final[t["trial"]] = t
    return {"root": root, "metas": metas, "trials": trials, "final": final}


def _settings(meta: dict) -> list[str]:
    return sorted(a for a in meta["argv"][1:] if not a.startswith(SETTINGS_IGNORED))


def verify(a: dict, b: dict, budget: int, n: int) -> dict:
    """Everything that must hold for the pair to be a matched comparison; each check is reported, not assumed."""
    ma, mb = a["metas"][0], b["metas"][0]
    checks = {}
    checks["same_pose_file_sha256"] = len({m["poses_sha256"] for m in a["metas"] + b["metas"]}) == 1
    checks["same_episode_steps"] = all(m["episode_steps"] == budget for m in a["metas"] + b["metas"])
    checks["different_checkpoints"] = (ma["checkpoint"].get("hub_revision", ma["checkpoint"].get("resolved"))
                                       != mb["checkpoint"].get("hub_revision", mb["checkpoint"].get("resolved")))
    checks["same_client_settings"] = all(_settings(m) == _settings(ma) for m in a["metas"] + b["metas"])
    sim_keys = ("task", "task_mode", "domain_randomization", "control_hz", "cameras")
    checks["same_simulator"] = all({k: m["simulator"].get(k) for k in sim_keys} == {k: ma["simulator"].get(k) for k in sim_keys}
                                   for m in a["metas"] + b["metas"])
    checks["same_source_commits"] = all(
        {k: v.get("commit") for k, v in m["git"].items()} == {k: v.get("commit") for k, v in ma["git"].items()}
        for m in a["metas"] + b["metas"])
    checks["same_uncommitted_code"] = all(  # IsaacLab + lerobot_openarm edits: identical bytes for every run start
        {k: v.get("worktree_sha256") for k, v in m["git"].items()} == {k: v.get("worktree_sha256") for k, v in ma["git"].items()}
        for m in a["metas"] + b["metas"])
    for name, run in (("a", a), ("b", b)):
        f = run["final"]
        checks[f"{name}_all_trials_complete"] = sorted(f) == list(range(n))
        checks[f"{name}_exactly_{budget}_steps"] = all(t["policy_steps"] == budget == t["steps"] for t in f.values())
        checks[f"{name}_no_reset_during_trial"] = all(t["scene_resets_during_trial"] == 0 for t in f.values())
        checks[f"{name}_pose_applied"] = all(t["placed"]["error_xy_m"] <= 0.002 for t in f.values())
        checks[f"{name}_stable_matches_task_stage"] = all(t["stable_matches_stage"] for t in f.values())
        checks[f"{name}_rehomed_trials"] = sorted(t["trial"] for t in f.values() if t["placed"].get("rehomed"))
        checks[f"{name}_code_versions"] = [{"started": m["started"], **{k: v.get("worktree_sha256", "")[:12]
                                                                        for k, v in m["git"].items()}} for m in run["metas"]]
    common = sorted(set(a["final"]) & set(b["final"]))
    checks["same_requested_pose_per_trial"] = all(
        a["final"][i]["placed"]["requested_xy"] == b["final"][i]["placed"]["requested_xy"] for i in common)
    checks["max_pose_error_mm"] = 1000 * max(t["placed"]["error_xy_m"] for r in (a, b) for t in r["final"].values())
    checks["max_pair_actual_pose_diff_mm"] = 1000 * max(
        math.dist(a["final"][i]["placed"]["actual_pose"][:2], b["final"][i]["placed"]["actual_pose"][:2]) for i in common)
    return checks


# -- statistics ---------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(xa: np.ndarray, xb: np.ndarray) -> dict:
    """Exact (binomial) McNemar on paired booleans; the paired difference with a Newcombe-style Wald CI."""
    only_a = int(np.sum(xa & ~xb))
    only_b = int(np.sum(~xa & xb))
    n = len(xa)
    if n == 0:
        return {"both": 0, "neither": 0, "only_a": 0, "only_b": 0, "p_exact": None, "diff_b_minus_a": None,
                "diff_ci95": None}
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    diff = (only_b - only_a) / n
    se = math.sqrt(max((only_a + only_b) / n - diff ** 2, 0.0) / n)
    return {"both": int(np.sum(xa & xb)), "neither": int(np.sum(~xa & ~xb)), "only_a": only_a, "only_b": only_b,
            "p_exact": p, "diff_b_minus_a": diff, "diff_ci95": (diff - 1.96 * se, diff + 1.96 * se)}


def logistic(X: np.ndarray, y: np.ndarray, iters: int = 100) -> dict:
    """IRLS logistic regression: coefficients, Wald SEs and p-values, deviance."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        H = X.T @ (w[:, None] * X)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    mu = np.clip(1 / (1 + np.exp(-(X @ beta))), 1e-12, 1 - 1e-12)
    cov = np.linalg.inv(X.T @ ((mu * (1 - mu))[:, None] * X))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    dev = -2 * np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
    return {"beta": beta.tolist(), "se": se.tolist(), "p": (2 * (1 - _norm_cdf(np.abs(z)))).tolist(), "deviance": float(dev)}


def _norm_cdf(z):
    from scipy.stats import norm

    return norm.cdf(z)


def interaction(xa, xb, x, y) -> dict:
    """Policy x position, conditioned on the pose (1:1 matched pairs): among discordant poses, P(generated wins)
    = logistic(b0 + bx * x_std + by * y_std). bx, by are the policy x position interactions of conditional
    logistic regression; b0 is the policy effect at the mean pose. Likelihood-ratio test of (bx, by) = 0."""
    disc = xa != xb
    out = {"discordant": int(disc.sum())}
    if disc.sum() < 8:
        out["note"] = "too few discordant pairs for a regression"
        return out
    xs, ys = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    wins_b = xb[disc].astype(float)
    full = logistic(np.c_[np.ones(disc.sum()), xs[disc], ys[disc]], wins_b)
    null = logistic(np.ones((disc.sum(), 1)), wins_b)
    lr = null["deviance"] - full["deviance"]
    out.update({"coef_names": ["policy (generated) at mean pose", "policy x x (per sd)", "policy x y (per sd)"],
                "full": full, "lr_stat": lr, "lr_p": float(1 - chi2.cdf(lr, 2)),
                "x_sd_m": float(x.std()), "y_sd_m": float(y.std())})
    return out


def bins(values, edges):
    return np.array([next(name for lo, hi, name in edges if lo <= v < hi) for v in values])


# -- gripper detector cross-check -------------------------------------------------------------------------------
def detector_stable(steps: list[dict]) -> bool:
    """The diagnosis's gripper-based stable hand-over (left holds, right does not, 30 steps running, after both
    hands held it), on this trial's commanded and measured grippers -- independent of the simulator's grasp tests."""
    from lerobot_robot_openarm_umeow.common import MOTOR_KEYS, load_calibration

    cal = load_calibration(str(HERE.parents[2] / "calibration.json"))
    A = np.array([s["action"] for s in steps])
    S = np.array([s["state"] for s in steps])

    def hold(side, prefix):
        j = MOTOR_KEYS.index(f"{prefix}J8.pos")
        opn = cal[side]["gripper"]["open_raw"]
        cmd, meas = A[:, j] / opn, S[:, j] / opn
        return (cmd < CMD_CLOSED) & (meas - cmd > BLOCK)

    hR, hL = hold("right", "R"), hold("left", "L")
    if not hR.any() or not hL.any():
        return False
    tR = int(np.flatnonzero(hR)[0])
    run = 0
    for t, p in enumerate(hL & ~hR):
        run = run + 1 if p else 0
        if run >= 30 and t > tR:
            return True
    return False


# -- figures --------------------------------------------------------------------------------------------------------
def _nonneg(err):
    return np.clip(np.asarray(err, dtype=float), 0, None)  # Wilson bounds can sit a float epsilon past the estimate


def _fig(w, h, n=1, m=1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 10,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    return plt.subplots(n, m, figsize=(w, h))


def figures(out: Path, rows: dict, cmp: dict, names: dict) -> list[str]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    made = []
    A, B = names["a"], names["b"]
    x, y = rows["x"], rows["y"]
    n = len(x)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        made.append(name)

    # 1 + 2: the three metrics, both models, Wilson 95 % CI
    fig, ax = _fig(8, 4.2)
    idx = np.arange(len(METRICS))
    for j, (key, color, label) in enumerate((("a", COLORS["a"], A), ("b", COLORS["b"], B))):
        k = np.array([rows[f"{m}_{key}"].sum() for m, _ in METRICS])
        lo_hi = np.array([wilson(int(v), n) for v in k])
        ax.bar(idx + (j - 0.5) * 0.36, k / n * 100, 0.34, color=color, label=label, edgecolor="#fcfcfb", linewidth=2)
        ax.errorbar(idx + (j - 0.5) * 0.36, k / n * 100, yerr=_nonneg([k / n * 100 - lo_hi[:, 0] * 100, lo_hi[:, 1] * 100 - k / n * 100]),
                    fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
        for i, v in enumerate(k):
            ax.text(idx[i] + (j - 0.5) * 0.36, 2, f"{v}", ha="center", va="bottom", color="#fcfcfb", fontsize=9, weight="bold")
    ax.set_xticks(idx, [label for _, label in METRICS])
    ax.set_ylabel("trials (%) of 100, Wilson 95 % CI")
    ax.set_ylim(0, 100)
    ax.set_title("Same 100 can poses, 330 simulated steps: three success metrics")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "success_metrics.png")

    # overall (any-step) alone with the historical numbers for context
    fig, ax = _fig(6.5, 3.6)
    labels = ["historical (random poses, 15 s wall)", "matched (same poses, 330 steps)"]
    for j, (key, color, label) in enumerate((("a", COLORS["a"], A), ("b", COLORS["b"], B))):
        vals = [HISTORICAL["success_any"][key], int(rows[f"success_any_{key}"].sum())]
        ci = [wilson(v, 100) for v in vals]
        pos = np.arange(2) + (j - 0.5) * 0.36
        ax.barh(pos, vals, 0.34, color=color, label=label, edgecolor="#fcfcfb", linewidth=2)
        ax.errorbar(vals, pos, xerr=_nonneg([[v - c[0] * 100 for v, c in zip(vals, ci)], [c[1] * 100 - v for v, c in zip(vals, ci)]]),
                    fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
        for p, v in zip(pos, vals):
            ax.text(1.5, p, str(v), va="center", color="#fcfcfb", fontsize=9, weight="bold")
    ax.set_yticks(np.arange(2), labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("any-step success, of 100 (Wilson 95 % CI)")
    ax.set_title("Overall success, historical vs matched")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "overall_success.png")

    # 3 + 4: success vs x and vs y (per bin, each model, Wilson CI)
    for coord, edges, fname in (("x", X_BINS, "success_vs_x.png"), ("y", Y_BINS, "success_vs_y.png")):
        fig, axs = _fig(10, 3.6, 1, len(METRICS))
        b = bins(rows[coord], edges)
        for ax, (m, label) in zip(axs, METRICS):
            for j, (key, color, name) in enumerate((("a", COLORS["a"], A), ("b", COLORS["b"], B))):
                ks = [int(rows[f"{m}_{key}"][b == e[2]].sum()) for e in edges]
                ns = [int((b == e[2]).sum()) for e in edges]
                p = [k / max(nn, 1) * 100 for k, nn in zip(ks, ns)]
                ci = [wilson(k, nn) for k, nn in zip(ks, ns)]
                pos = np.arange(len(edges)) + (j - 0.5) * 0.36
                ax.bar(pos, p, 0.34, color=color, label=name, edgecolor="#fcfcfb", linewidth=2)
                ax.errorbar(pos, p, yerr=_nonneg([[v - c[0] * 100 for v, c in zip(p, ci)], [c[1] * 100 - v for v, c in zip(p, ci)]]),
                            fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
                for q, k, nn in zip(pos, ks, ns):
                    ax.text(q, 101, f"{k}\n/{nn}", ha="center", va="bottom", fontsize=7.5, color="#52514e")
            ax.set_xticks(np.arange(len(edges)), [e[2] for e in edges], fontsize=8.5)
            ax.set_ylim(0, 130)
            ax.set_title(label, fontsize=10)
        axs[0].set_ylabel("success (%) per region, Wilson 95 % CI")
        axs[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
        fig.suptitle(f"Success vs starting {coord} (same poses for both models; small bins: descriptive)", fontsize=11)
        save(fig, fname)

    # 5: XY map of paired outcomes, any-step and stable
    for m, fname in (("success_any", "xy_paired_outcomes.png"), ("holding_end", "xy_paired_outcomes_holding_end.png")):
        fig, ax = _fig(7.2, 5.4)
        xa, xb = rows[f"{m}_a"], rows[f"{m}_b"]
        cats = [("both", xa & xb, "o", f"both succeed"), ("neither", ~xa & ~xb, "x", "both fail"),
                ("a", xa & ~xb, "^", f"only {A}"), ("b", ~xa & xb, "s", f"only {B}")]
        for key, mask, marker, label in cats:
            ax.scatter(x[mask], y[mask], s=46, marker=marker, color=COLORS[key], label=f"{label} ({mask.sum()})",
                       edgecolors="#fcfcfb" if marker != "x" else None, linewidths=0.8 if marker != "x" else 1.4)
        for lo, hi, name in X_BINS[1:]:
            ax.axvline(lo, color="#52514e", lw=0.8, ls="--")
        ax.set_xlabel("can start x (m, env frame; away from the robot)")
        ax.set_ylabel("can start y (m; + = left arm side)")
        ax.set_title(f"Paired outcome per pose: {dict(METRICS)[m]}")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save(fig, fname)

    # 6: stage-wise outcome
    order = ["no_right_grasp", "right_grasp_no_pass", "pass_not_secured", "stable_then_lost", "stable_held_to_end"]
    labels = ["no right grasp", "right grasp,\nno pass", "pass, left never\nsecured it", "stable,\nlost before 330",
              "stable,\nheld to 330"]
    fig, ax = _fig(8.5, 3.8)
    for j, (key, color, name) in enumerate((("a", COLORS["a"], A), ("b", COLORS["b"], B))):
        counts = [int(np.sum(rows[f"outcome_{key}"] == o)) for o in order]
        pos = np.arange(len(order)) + (j - 0.5) * 0.36
        ax.bar(pos, counts, 0.34, color=color, label=name, edgecolor="#fcfcfb", linewidth=2)
        for q, c in zip(pos, counts):
            ax.text(q, c + 0.8, str(c), ha="center", fontsize=8.5, color="#0b0b0b")
    ax.set_xticks(np.arange(len(order)), labels, fontsize=8.5)
    ax.set_ylabel("trials (of 100)")
    ax.set_title("Furthest stage reached (simulator ground truth)")
    ax.legend(frameon=False)
    save(fig, "stagewise_outcomes.png")

    # 7: hand-over timing
    events = [("right_grasp_step", "right grasp"), ("left_grasp_step", "pass (left takes it)"),
              ("right_release_step", "right releases"), ("stable_completion_step", "stable (30 steps held)")]
    fig, ax = _fig(8.5, 3.8)
    for j, (key, color, name) in enumerate((("a", COLORS["a"], A), ("b", COLORS["b"], B))):
        data = [[v for v in rows[f"{e}_{key}"] if v is not None and not (isinstance(v, float) and math.isnan(v))]
                for e, _ in events]
        pos = np.arange(len(events)) + (j - 0.5) * 0.36
        bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=True,
                        medianprops={"color": "#0b0b0b"}, flierprops={"markersize": 3})
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
        for q, d in zip(pos, data):
            ax.text(q, 335, f"n={len(d)}", ha="center", fontsize=7.5, color="#52514e")
        ax.plot([], [], color=color, lw=6, label=name)
    for k in (240, 270, 300, 330):
        ax.axhline(k, color="#c3c2b7", lw=0.6, ls=":")
    ax.set_xticks(np.arange(len(events)), [label for _, label in events], fontsize=9)
    ax.set_ylabel("policy step (30 Hz)")
    ax.set_ylim(0, 350)
    ax.set_title("When each hand-over event happens (trials that reach it)")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "handover_timing.png")

    # 8: steps and wall-clock per trial
    fig, axs = _fig(10, 3.6, 1, 2)
    for j, (key, color, name) in enumerate((("a", COLORS["a"], A), ("b", COLORS["b"], B))):
        axs[0].plot(np.arange(n), rows[f"steps_{key}"], "o", ms=3, color=color, label=name)
        axs[1].hist(rows[f"wall_{key}"], bins=20, color=color, alpha=0.75, label=name, edgecolor="#fcfcfb")
    axs[0].set_xlabel("trial")
    axs[0].set_ylabel("policy-controlled simulation steps")
    axs[0].set_title("Steps per trial (every trial must be 330)")
    axs[1].set_xlabel("wall-clock seconds per trial")
    axs[1].set_ylabel("trials")
    axs[1].set_title("Wall-clock duration (not the budget)")
    axs[1].legend(frameon=False)
    save(fig, "episode_steps_and_wallclock.png")
    return made


# -- main -----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(HERE / "results_manual_10"))
    ap.add_argument("--b", default=str(HERE / "results_generated_500"))
    ap.add_argument("--a_name", default="manual-10")
    ap.add_argument("--b_name", default="generated-500")
    ap.add_argument("--budget", type=int, default=330)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=str(HERE / "figures"))
    ap.add_argument("--json", default=str(HERE / "comparison.json"))
    args = ap.parse_args()

    a, b = load_run(Path(args.a)), load_run(Path(args.b))
    checks = verify(a, b, args.budget, args.n)
    common = sorted(set(a["final"]) & set(b["final"]))
    rows = {"trial": np.array(common)}
    rows["x"] = np.array([a["final"][i]["pose"]["x"] for i in common])
    rows["y"] = np.array([a["final"][i]["pose"]["y"] for i in common])
    for key, run in (("a", a), ("b", b)):
        f = [run["final"][i] for i in common]
        for m, _ in METRICS:
            rows[f"{m}_{key}"] = np.array([bool(t[m]) for t in f])
        for k in (240, 270, 300, 330):
            rows[f"holding_at_{k}_{key}"] = np.array([bool(t[f"holding_at_{k}"]) for t in f])
        rows[f"outcome_{key}"] = np.array([t["outcome"] for t in f])
        rows[f"steps_{key}"] = np.array([t["policy_steps"] for t in f])
        rows[f"wall_{key}"] = np.array([t["wall_seconds"] for t in f])
        for e in ("right_grasp_step", "left_grasp_step", "right_release_step", "stable_completion_step",
                  "first_success_step"):
            rows[f"{e}_{key}"] = [t[e] for t in f]
        rows[f"incl_return_{key}"] = np.array([bool(t.get("success_incl_return")) for t in f])
        rows[f"detector_stable_{key}"] = np.array([detector_stable(json.loads(
            (run["root"] / "steps" / f"trial_{t['trial']:03d}_{t['attempt']}.json").read_text())) for t in f])
        rows[f"timing_{key}"] = [t.get("timing") or {} for t in f]

    n = len(common)
    out = {"checks": checks, "n_paired": n, "names": {"a": args.a_name, "b": args.b_name}, "metrics": {}, "regions": {},
           "historical": HISTORICAL}
    for m, label in METRICS + [(f"holding_at_{k}", f"holding at step {k}") for k in (240, 270, 300)]:
        ka, kb = int(rows[f"{m}_a"].sum()), int(rows[f"{m}_b"].sum())
        out["metrics"][m] = {"label": label, "a": ka, "b": kb, "a_ci95": wilson(ka, n), "b_ci95": wilson(kb, n),
                             "mcnemar": mcnemar(rows[f"{m}_a"], rows[f"{m}_b"])}
    for coord, edges in (("x", X_BINS), ("y", Y_BINS)):
        b_ = bins(rows[coord], edges)
        for lo, hi, name in edges:
            sel = b_ == name
            out["regions"][name] = {m: {"n": int(sel.sum()), "a": int(rows[f"{m}_a"][sel].sum()),
                                        "b": int(rows[f"{m}_b"][sel].sum()),
                                        "mcnemar": mcnemar(rows[f"{m}_a"][sel], rows[f"{m}_b"][sel])}
                                    for m, _ in METRICS}
    out["interaction"] = {m: interaction(rows[f"{m}_a"], rows[f"{m}_b"], rows["x"], rows["y"]) for m, _ in METRICS}
    out["per_pose"] = [{"trial": int(i), "x": float(xx), "y": float(yy),
                        **{f"{m}_{k}": bool(rows[f"{m}_{k}"][j]) for m, _ in METRICS for k in "ab"},
                        "outcome_a": str(rows["outcome_a"][j]), "outcome_b": str(rows["outcome_b"][j])}
                       for j, (i, xx, yy) in enumerate(zip(common, rows["x"], rows["y"]))]
    out["outcomes"] = {k: {o: int(np.sum(rows[f"outcome_{k}"] == o)) for o in sorted(set(rows[f"outcome_{k}"]))}
                       for k in "ab"}
    out["timing_steps"] = {}
    for k in "ab":
        for e in ("right_grasp_step", "left_grasp_step", "right_release_step", "stable_completion_step"):
            v = np.array([t for t in rows[f"{e}_{k}"] if t is not None], dtype=float)
            out["timing_steps"][f"{e}_{k}"] = ({"n": int(len(v)), "median": float(np.median(v)), "min": float(v.min()),
                                                "max": float(v.max())} if len(v) else {"n": 0})
    out["runtime"] = {k: {"steps_min": int(rows[f"steps_{k}"].min()), "steps_max": int(rows[f"steps_{k}"].max()),
                          "wall_median_s": float(np.median(rows[f"wall_{k}"])),
                          "wall_min_s": float(rows[f"wall_{k}"].min()), "wall_max_s": float(rows[f"wall_{k}"].max()),
                          "latency_mean_ms": float(np.nanmean([t.get("latency_mean_ms") or np.nan for t in rows[f"timing_{k}"]])),
                          "guided_mean": float(np.nanmean([t.get("guided_mean") or np.nan for t in rows[f"timing_{k}"]])),
                          "chunks_mean": float(np.nanmean([t.get("chunks") or np.nan for t in rows[f"timing_{k}"]])),
                          "dropped_whole_total": int(np.nansum([t.get("dropped_whole") or 0 for t in rows[f"timing_{k}"]])),
                          "starved_pct_mean": float(np.nanmean([t.get("starved_pct") or np.nan for t in rows[f"timing_{k}"]]))}
                      for k in "ab"}
    out["return_contamination"] = {k: {"success_incl_return": int(rows[f"incl_return_{k}"].sum()),
                                       "success_policy_steps_only": int(rows[f"success_any_{k}"].sum()),
                                       "only_during_return": int(np.sum(rows[f"incl_return_{k}"] & ~rows[f"success_any_{k}"]))}
                                   for k in "ab"}
    out["detector_agreement"] = {k: {"agree": int(np.sum(rows[f"detector_stable_{k}"] == rows[f"stable_{k}"])),
                                     "detector_only": int(np.sum(rows[f"detector_stable_{k}"] & ~rows[f"stable_{k}"])),
                                     "truth_only": int(np.sum(~rows[f"detector_stable_{k}"] & rows[f"stable_{k}"]))}
                                 for k in "ab"}
    out["figures"] = figures(Path(args.out), rows, out, {"a": args.a_name, "b": args.b_name})
    Path(args.json).write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print(json.dumps({"checks": checks, "metrics": {m: (v["a"], v["b"], round(v["mcnemar"]["p_exact"], 4))
                                                     for m, v in out["metrics"].items()}}, indent=1))


if __name__ == "__main__":
    main()
