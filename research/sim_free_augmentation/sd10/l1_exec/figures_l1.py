"""SD10-L1 figures (descriptive): endpoints with recording CIs and per-seed values, contrasts vs thresholds,
error versus initial object position, training phase exposure and source pace. Writes figures/*.png.
Palette: dataviz reference slots 1-3 (validated; aqua <3:1 -> every series is direct-labelled / legended).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import l1lib as L  # noqa: E402

PT = L.PT
FIG = L.EXEC / "figures"
COL = {"Original10": "#2a78d6", "SD10": "#eb6834", "Random10": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
ENDPOINTS = [("far/grasp", "Far grasp (6 rec.)"), ("near/grasp", "Near grasp (15)"), ("all21/handover", "Handover (21)"),
             ("all21/phase_balanced", "Phase-balanced (21)"), ("all21/overall_time_weighted", "Overall (21)")]
CONTRASTS = [("far_grasp", "Far grasp"), ("near_grasp", "Near grasp"), ("handover_all21", "Handover"),
             ("phase_balanced_all21", "Phase-balanced"), ("overall_time_weighted_all21", "Overall")]


def fig_endpoints(R):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for j, (k, lab) in enumerate(ENDPOINTS):
        for i, arm in enumerate(L.ARMS):
            d = R["absolute_errors"][arm][k]
            y = j + (i - 1) * .24
            ax.plot(d["ci95"], [y, y], color=COL[arm], lw=2, solid_capstyle="round")
            ax.plot(d["mean"], y, "o", ms=8, color=COL[arm], mec=SURF, mew=2, label=arm if j == 0 else None)
            ax.plot(d["per_seed"], [y] * 3, "|", ms=7, color=INK2)
    ax.set_yticks(range(len(ENDPOINTS)), [l for _, l in ENDPOINTS])
    ax.invert_yaxis()
    ax.set_xlabel("14-arm-joint MAE (rad); bar = recording 95% CI, ticks = seeds")
    ax.set_title("Absolute error per arm (lower is better)", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    for j, (k, lab) in enumerate(CONTRASTS):
        for i, (cn, cl, cc) in enumerate([("SD10_vs_Original10", "SD10 vs Original10", COL["Original10"]),
                                          ("SD10_vs_Random10", "SD10 vs Random10", COL["Random10"])]):
            d = R["contrasts"][cn][k]["relative"]
            y = j + (i - .5) * .3
            ax.plot(np.array(d["ci95"]) * 100, [y, y], color=cc, lw=2, solid_capstyle="round")
            ax.plot(d["relative_improvement"] * 100, y, "D", ms=7, color=cc, mec=SURF, mew=2, label=cl if j == 0 else None)
            ax.plot(np.array(d["per_seed"]) * 100, [y] * 3, "|", ms=7, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(5, color=INK2, lw=1, ls=":")
    ax.axvline(-2, color=INK2, lw=1, ls="--")
    ax.text(5.5, -.55, "5% primary threshold", color=INK2, fontsize=7.5, ha="left", va="center")
    ax.text(-2.5, -.55, "−2% harm guard", color=INK2, fontsize=7.5, ha="right", va="center")
    ax.set_yticks(range(len(CONTRASTS)), [l for _, l in CONTRASTS])
    ax.set_xlabel("relative improvement (%), positive = SD10 lower error")
    ax.set_ylim(len(CONTRASTS) - .4, -.8)
    ax.set_title("SD10 contrasts (paired recording bootstrap)", loc="left", color=INK, pad=22)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(FIG / "l1_endpoints.png", dpi=160)


def fig_position(R, man):
    rec = man["recordings"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    far = set(man["recording_groups"]["far"])
    for ax, key, title in [(axes[0], "Original10", "vs distance to nearest Original10 source (common bin axis)"),
                           (axes[1], None, "vs distance to the arm's own nearest source")]:
        for arm in L.ARMS:
            x = [rec[str(e)]["distance_to_sources_px"][key or arm] for e in PT.EVAL]
            y = [np.mean([R["per_record"][f"{arm}_s{s}"][str(e)]["grasp"] for s in L.SEEDS]) for e in PT.EVAL]
            ax.scatter(x, y, s=[40 if e in far else 18 for e in PT.EVAL], color=COL[arm], alpha=.85, edgecolor=SURF, lw=.8, label=arm)
        ax.set_xlabel("initial lid-centre distance (native body px)")
        ax.set_ylabel("grasp-phase MAE (rad), seed mean")
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        if key:
            ax.axvline(40, color=INK2, lw=1, ls=":")
            ax.text(41, ax.get_ylim()[1], "far > 40 px", color=INK2, fontsize=8, va="top")
        ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    uv = np.array([rec[str(e)]["lid_uv_native_px"] for e in PT.EVAL])
    d = np.array([np.mean([R["per_record"][f"SD10_s{s}"][str(e)]["grasp"] - R["per_record"][f"Original10_s{s}"][str(e)]["grasp"]
                           for s in L.SEEDS]) for e in PT.EVAL])
    m = np.abs(d).max()
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=d, cmap="RdBu_r", vmin=-m, vmax=m, s=70, edgecolor=INK2, lw=.6)
    for (u, v), e in zip(uv, PT.EVAL):
        ax.text(u + 4, v - 4, str(e), fontsize=7, color=INK2)
    caps = L.RT["captures"]
    for arm, mk in (("Original10", "s"), ("SD10", "^")):
        p = np.array([caps[str(e)]["lid_uv_native_px"] for e in L.ARMS[arm]])
        ax.scatter(p[:, 0], p[:, 1], marker=mk, facecolor="none", edgecolor=COL[arm], s=45, lw=1.4, label=f"{arm} sources")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("SD10 − Original10 grasp MAE at eval positions\n(red = SD10 worse, blue = SD10 better)", loc="left", fontsize=8.5, color=INK)
    ax.set_xlabel("u (px)"); ax.set_ylabel("v (px)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.colorbar(sc, ax=ax, shrink=.8, label="Δ MAE (rad)")
    fig.tight_layout()
    fig.savefig(FIG / "l1_error_vs_position.png", dpi=160)


def fig_exposure(X):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    w = .26
    for i, arm in enumerate(L.ARMS):
        v = [X["exposure"][arm]["seed_mean_target_phase_counts"][p] / 1000 for p in PT.PHASES]
        ax.bar(np.arange(5) + (i - 1) * (w + .02), v, w, color=COL[arm], label=arm)
    ax.set_xticks(range(5), PT.PHASES)
    ax.set_ylabel("target timesteps per fit (thousands)")
    ax.set_ylim(0, 270)
    ax.set_title("Realized training target-phase exposure (equal 800k total)", loc="left", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax = axes[1]
    for i, arm in enumerate(L.ARMS):
        for j, k in enumerate(["approach_to_rc_s", "rc_to_lc_s", "lc_to_ro_s", "ro_to_end_s"]):
            v = X["pace"][arm][k]["values"]
            ax.scatter(np.full(len(v), j + (i - 1) * .22) + np.linspace(-.05, .05, len(v)), v, s=14, color=COL[arm],
                       label=arm if j == 0 else None)
    ax.set_xticks(range(4), ["start→right close", "right close→left close", "left close→right open", "right open→end"], fontsize=8)
    ax.set_ylabel("seconds (command-proxy events)")
    ax.set_title("Training-source pace per arm (10 sources each)", loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "l1_exposure_pace.png", dpi=160)


def main():
    FIG.mkdir(exist_ok=True)
    R = json.loads((L.OUT / "results.json").read_text())
    assert R["complete"]
    man = json.loads((L.L1 / "locked_evaluation_manifest.json").read_text())
    X = json.loads((L.OUT / "exposure_pace.json").read_text())
    fig_endpoints(R)
    fig_position(R, man)
    fig_exposure(X)
    print("figures written")


if __name__ == "__main__":
    main()
