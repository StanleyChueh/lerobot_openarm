"""Build the local visual diagnosis from saved evidence; no experiments or network.

Run from the repository root with the existing .venv/bin/python -B.
Outputs: results/eval_v00/index.html and charts/*.svg, *.png.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parents[1]
RESULT = HERE / "results/eval_v00"
REPO = HERE.parents[2]
BLUE, TEAL, ORANGE, INK, GREY = "#3465ae", "#137f79", "#c76731", "#1e3346", "#9ba8b4"
GROUPS = ["front-right", "rear-right", "rear-left", "front-left", "middle"]
LABELS = ["1  Front right", "2  Rear right", "3  Rear left", "4  Front left", "5  Middle"]
PHASES = ["approach", "final_approach", "grasp", "handover"]


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def success_interval(k, n):
    assert 0 <= k <= n and n > 0
    ci = binomtest(k, n).proportion_ci()
    return k / n, ci.low, ci.high


def save(fig, name, out):
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(data, out):
    o = data["outcomes"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.65]}, layout="constrained")
    for model, color, y in (("real", BLUE, 1), ("mixed", TEAL, 0)):
        k, n = sum(o[model]), sum(o["n"]); value, lo, hi = success_interval(k, n)
        axes[0].errorbar(100 * value, y, xerr=[[100 * (value - lo)], [100 * (hi - value)]], fmt="o", color=color, capsize=5, ms=10, lw=2)
        axes[0].text(100 * value, y + .18, f"{k}/{n}  ({100 * value:.0f}%)", ha="center", color=color, weight="bold")
        for i, (k, n) in enumerate(zip(o[model], o["n"])):
            value, lo, hi = success_interval(k, n); yy = i + (-.15 if model == "real" else .15)
            axes[1].errorbar(100 * value, yy, xerr=[[100 * (value - lo)], [100 * (hi - value)]], fmt="o", color=color, capsize=3, ms=7)
            axes[1].text(104, yy, f"{k}/{n}", color=color, va="center", fontsize=10)
    axes[0].set(yticks=[0, 1], yticklabels=["Real + v00", "Real only"], ylim=(-.4, 1.5), xlim=(-4, 104), xlabel="Reported success (%)", title="Overall: 50% → 75%")
    axes[1].set(yticks=range(5), yticklabels=LABELS, ylim=(4.6, -.6), xlim=(-4, 115), xlabel="Reported success (%)", title="Reported gains are in positions 1–3")
    for ax in axes:
        ax.set_xticks([0, 25, 50, 75, 100]); ax.grid(axis="x", alpha=.15)
    fig.suptitle("Promising outcomes, with wide uncertainty", fontsize=18, weight="bold")
    fig.supxlabel(f"Exact binomial 95% intervals · user-reported outcomes · Fisher p = {o['fisher_two_sided_p']:.3f} · batch 8 vs 16 is a confound", fontsize=10)
    save(fig, "outcomes", out)


def plot_coverage(data, coverage, out):
    sources = np.array(coverage["real_xy_cm"])
    targets = [None, np.array(coverage["v00_xy_cm"]), np.array(coverage["v01_xy_cm"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5), sharex=True, sharey=True, layout="constrained")
    for i, (ax, extra) in enumerate(zip(axes, targets)):
        training = sources if extra is None else np.r_[sources, extra]
        h = ConvexHull(training); H = training[np.r_[h.vertices, h.vertices[0]]]
        ax.fill(H[:, 1], H[:, 0], color=TEAL if i else BLUE, alpha=.055)
        ax.plot(H[:, 1], H[:, 0], color=TEAL if i else BLUE, lw=1, alpha=.5, label="Training convex hull")
        if extra is not None:
            ax.scatter(extra[:, 1], extra[:, 0], s=14, c=TEAL, alpha=.42, marker="+", label=f"{len(extra)} synthetic starts")
        ax.scatter(sources[:, 1], sources[:, 0], s=65, c=BLUE, marker="*", zorder=3, label="10 real starts")
        for g, label in zip(GROUPS, LABELS):
            xy = np.array([r["can_xy_cm"] for r in data["rows"] if r["group"] == g]); rear = g.startswith("rear")
            ax.scatter(xy[:, 1], xy[:, 0], s=26, facecolors=ORANGE if rear else "white", edgecolors=ORANGE if rear else INK, marker="s" if rear else "o", zorder=4)
            center = xy.mean(0); ax.annotate(label[0], (center[1], center[0]), xytext=(10, 6), textcoords="offset points", fontsize=12, weight="bold", color=ORANGE if rear else INK)
        ax.set(title=["10 real demos", "+ v00: 65 synthetic", "+ v01: 269 synthetic"][i], xlabel="World y (cm; robot left)", aspect="equal", xlim=(11, -12), ylim=(10, 44))
        ax.grid(alpha=.12)
    axes[0].set_ylabel("World x (cm; forward ↑ / rear ↓)")
    axes[1].text(.5, .985, "Orange squares: rear evaluation starts", transform=axes[1].transAxes, ha="center", va="top", fontsize=9, color=ORANGE)
    axes[2].legend(loc="upper right", fontsize=8)
    fig.suptitle("More positions do not automatically fill the rear corners", fontsize=18, weight="bold")
    fig.supxlabel("All 40 measured evaluation starts · approximate calibrated coordinates · rear starts outside hull: real 14/14; real + v00 13/14", fontsize=10)
    save(fig, "coverage_comparison", out)


def plot_spatial(data, coverage, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for key, color, offset, label in (("real", BLUE, -.16, "10 real"), ("real_plus_v00", TEAL, .16, "+ v00")):
        for i, g in enumerate(GROUPS):
            v = np.array([r[key]["nn_cm"] for r in coverage["rows"] if r["group"] == g]); y = i + offset
            axes[0].scatter(v, np.full(len(v), y), color=color, alpha=.22, s=28)
            axes[0].scatter([v.mean()], [y], color=color, marker="D", s=45, label=label if i == 0 else None)
            axes[0].text(v.mean() + .12, y - .07, f"{v.mean():.1f}", color=color, fontsize=9)
    for model, color, offset, label in (("real", BLUE, -.16, "Real policy"), ("mixed", TEAL, .16, "Mixed policy")):
        for i, g in enumerate(GROUPS):
            r = data["summary"][model][g]["adjusted_error_cm"]; m = r["median"]; lo, hi = r["ci95"]
            axes[1].errorbar(m, i + offset, xerr=[[m - lo], [hi - m]], fmt="o", color=color, capsize=3, label=label if i == 0 else None)
    for ax in axes:
        ax.set(yticks=range(5), yticklabels=LABELS, ylim=(4.6, -.6)); ax.grid(axis="x", alpha=.15); ax.legend(loc="lower right", fontsize=9)
    axes[0].set(xlabel="Distance to nearest training position (cm)", title="Training positions get closer to rear starts", xlim=(0, 8))
    axes[1].set(xlabel="Close-location proxy error (cm)", title="The mixed policy follows placement better", xlim=(0, 15))
    fig.supxlabel("Left: dots = all 40 starts; diamonds = group means. Right: trial medians, 95% bootstrap intervals; not actual grasp accuracy after the can moves.", fontsize=9)
    save(fig, "spatial_tracking", out)


def plot_timing(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, layout="constrained")
    for ax, model, color, title in zip(axes, ("real", "mixed"), (BLUE, TEAL), ("Real policy · 7 rear trials", "Mixed policy · 7 rear trials")):
        rows = [r for r in data["rows"] if r["model"] == model and r["group"].startswith("rear")]
        assert len(rows) == 7 and all(r["motion_before_close_s"] > 0 for r in rows)
        for i, r in enumerate(rows):
            dt = r["motion_before_close_s"]
            ax.barh(i, dt, left=-dt, height=.48, color=color, alpha=.22)
            ax.scatter([-dt], [i], color=color, s=45)
            ax.text(-dt - .035, i, f"{dt:.2f} s", ha="right", va="center", fontsize=10, color=color)
        ax.axvline(0, color=INK, lw=1.5); ax.set(yticks=range(7), yticklabels=[f"E{r['episode']:02d} · {'right' if r['group']=='rear-right' else 'left'}" for r in rows], ylim=(6.6, -.7), xlim=(-1.35, .08), xlabel="Time relative to close command (s)", title=title)
        ax.grid(axis="x", alpha=.12)
    fig.suptitle("In every rear trial, the can moved before closure", fontsize=18, weight="bold")
    fig.supxlabel("Dot = first 3-frame lid displacement >5 px. Video/command timing proxy; independent camera latency and physical force are not measured.", fontsize=9)
    save(fig, "preclose_timing", out)


def plot_wrist(wrist, out):
    obj = wrist["summary"]["object_position"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    labels = ["Approach", "Final\napproach", "Grasp", "Hand-over"]
    for ax, key, color, ylabel in ((axes[0], "camera_translation_cm", BLUE, "Camera translation (cm)"),
                                   (axes[1], "can_delta_px", TEAL, "Can-surface image displacement (px)")):
        for i, ph in enumerate(PHASES):
            r = obj[ph][key]; m = r["median"]; lo, hi = r["ci95"]
            ax.errorbar(i, m, yerr=[[max(0, m-lo)], [max(0, hi-m)]], color=color, fmt="o", capsize=5, ms=9, lw=2)
            ax.text(i, m + (.17 if key.startswith("camera") else .009), f"{m:.2f}" if key.startswith("camera") else f"{m:.3f}", ha="center", fontsize=11, weight="bold", color=color)
        ax.set(xticks=range(4), xticklabels=labels, ylabel=ylabel); ax.grid(axis="y", alpha=.15)
    axes[0].set(ylim=(-.15, 3.7), title="The camera changes its world position")
    axes[1].set(ylim=(-.006, .135), title="The relative can view barely changes")
    axes[1].text(.5, .96, "Zoomed axis: the 5 px threshold is above this range", ha="center", va="top", transform=axes[1].transAxes, fontsize=9)
    fig.suptitle("A changed camera pose is not the same as a wrong can view", fontsize=17, weight="bold")
    fig.supxlabel("46 object-position variants · source-cluster 95% intervals (7 measurable sources in approach, 8 afterward) · modelled geometry, not a policy ablation", fontsize=9)
    save(fig, "wrist_relative_geometry", out)

    fig, ax = plt.subplots(figsize=(10, 3.1), layout="constrained"); ax.set(xlim=(0, 10), ylim=(0, 3)); ax.axis("off")
    for x, label, color in ((1.2, "Source scene", BLUE), (6.2, "Edited scene", TEAL)):
        ax.add_patch(Rectangle((x, .7), 1, .6, facecolor=color, alpha=.85)); ax.text(x+.5, .45, "wrist camera", ha="center", fontsize=11)
        ax.add_patch(Circle((x+2, 1), .25, color=ORANGE)); ax.text(x+2, .45, "can", ha="center", fontsize=11)
        ax.add_patch(FancyArrowPatch((x+1.1, 1), (x+1.7, 1), arrowstyle="<->", mutation_scale=15, color=INK)); ax.text(x+1.5, 1.5, "same relative view", ha="center", fontsize=11)
        ax.text(x+1, 2.25, label, ha="center", weight="bold", fontsize=14, color=color)
    ax.add_patch(FancyArrowPatch((4, 1.05), (5.7, 1.05), arrowstyle="->", mutation_scale=25, color=INK, lw=2)); ax.text(4.85, 1.65, "translate both", ha="center", fontsize=12)
    fig.supxlabel("Schematic of rigid translation. Background, the other arm and contact slip can still disagree; those effects are not ruled out.", fontsize=10)
    save(fig, "wrist_explanation", out)


def generation_payload():
    old = load(REPO / "artifacts/eval_v00_inputs/training_v00/meta/sfa_provenance.json")["episodes"][10:]
    new = load(REPO / "artifacts/eval_v00_inputs/training_v01/meta/sfa_provenance.json")["episodes"]
    shifts = [np.linalg.norm(np.array(p["can_initial_world_m"])[:2] - np.array(p["source_can_initial_world_m"])[:2]) * 100 for p in old if p["variation"].startswith("init_")]
    return dict(v00_seconds=[p["n_frames"] / 30 for p in old], v01_seconds=[p["n_frames"] / 30 for p in new],
                v00_shifts_cm=shifts, v01_shifts_cm=[np.linalg.norm(p["shift_m"][:2]) * 100 for p in new],
                lengthened=sum(p["frames_added"] > 0 for p in new), total=len(new))


def plot_generation(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained"); rng = np.random.default_rng(0)
    for ax, keys in zip(axes, (("v00_seconds", "v01_seconds"), ("v00_shifts_cm", "v01_shifts_cm"))):
        for y, (key, color) in enumerate(zip(keys, (BLUE, TEAL))):
            v = np.array(d[key]); ax.scatter(v, y + rng.uniform(-.12, .12, len(v)), s=15, c=color, alpha=.35)
            q = np.quantile(v, [.25, .5, .75]); ax.plot(q[[0, 2]], [y, y], color=color, lw=5); ax.scatter(q[1], y, color="white", edgecolors=color, s=65, zorder=4)
            ax.text(v.max(), y-.25, f"max {v.max():.1f}", ha="right", color=color, fontsize=10)
        ax.set(yticks=[0, 1], yticklabels=[f"v00 (n={len(d[keys[0]])})", f"v01 (n={len(d[keys[1]])})"], ylim=(1.55, -.4)); ax.grid(axis="x", alpha=.12)
    axes[0].set(title="Longer episodes: 119/269 v01 episodes lengthened", xlabel="Episode duration (s; 30 frames/s)", xlim=(0, 36))
    axes[1].axvline(6, color=ORANGE, ls="--", lw=1); axes[1].set(title="Larger starting-can shifts", xlabel="Can translation from its source (cm)", xlim=(0, 12))
    fig.suptitle("Two confirmed changes can explain the slower, distorted appearance", fontsize=17, weight="bold")
    fig.supxlabel("Dots = episodes; line = middle 50%; white dot = median. Shift panel compares 46 v00 can-position variants with 269 v01 variants; no causal quality claim.", fontsize=9)
    save(fig, "generation_changes", out)


def plot_renderer(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True, layout="constrained")
    names = ["sfagen_minus_legacy", "sfagen_lid_minus_sfagen", "sfagen_lid_minus_legacy"]
    for ax, region, title in zip(axes, ("all", "can"), ("Moving foreground", "Can region")):
        for i, key in enumerate(names):
            r = summary["metrics"][region][key]; m = r["mean_db"]; lo, hi = r["ci95"]; color = ORANGE if hi < 0 else BLUE
            ax.errorbar(m, i, xerr=[[m-lo], [hi-m]], fmt="o", capsize=4, color=color, ms=8)
            ax.text(1.12, i, f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=9)
        ax.axvline(0, color=INK, lw=1, ls="--"); ax.set(xlim=(-2.05, 2.3), ylim=(2.5, -.5), xlabel="PSNR difference (dB; higher is better)", title=title, yticks=range(3), yticklabels=["sfagen − legacy", "lid pose − sfagen", "lid pose − legacy"]); ax.set_xticks([-2, -1, 0, 1]); ax.grid(axis="x", alpha=.12)
    fig.suptitle("Changing to the legacy lid pose did not establish a gain", fontsize=18, weight="bold")
    fig.supxlabel("360 matched scenes / 90 ordered pairs · source/target-cluster 95% intervals · crossing zero = uncertain direction, not proof of equivalence · reused development data", fontsize=9)
    save(fig, "renderer_effects", out)


def main():
    out = RESULT / "charts"; out.mkdir(exist_ok=True)
    experiment_files = [RESULT / x for x in ("rollouts.json", "coverage.json", "wrist.json", "wrist_frames.csv", "render/frames.json", "render/summary.json", "PROTOCOL.md")]
    before = {str(p.relative_to(REPO)): digest(p) for p in experiment_files}
    data, coverage, wrist, renderer = [load(RESULT / p) for p in ("rollouts.json", "coverage.json", "wrist.json", "render/summary.json")]
    assert len(data["rows"]) == len(coverage["rows"]) == 40
    assert renderer["scenes"] == 360 and renderer["pairs"] == 90
    generation = generation_payload(); assert generation["lengthened"] == 119 and generation["total"] == 269
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": INK,
                         "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "eval-v00"})
    plot_outcomes(data, out); plot_coverage(data, coverage, out); plot_spatial(data, coverage, out)
    plot_timing(data, out); plot_wrist(wrist, out); plot_generation(generation, out); plot_renderer(renderer, out)
    payload = dict(rows=data["rows"], summary=data["summary"], outcomes=data["outcomes"],
                   shifts=[p.name for p in sorted((RESULT / "render").glob("shift_e*.jpg"))],
                   cross=[p.name for p in sorted((RESULT / "render").glob("cross_A*.jpg"))],
                   pins=load(RESULT / "run_meta.json")["pins"])
    html = (HERE / "tools/eval_v00_report.html").read_text().replace("__REPORT_DATA__", json.dumps(payload).replace("</", "<\\/"))
    assert "__REPORT_DATA__" not in html
    (RESULT / "index.html").write_text(html)
    assert before == {str(p.relative_to(REPO)): digest(p) for p in experiment_files}, "Visual build changed experimental evidence"
    (RESULT / "visual_build.json").write_text(json.dumps(dict(experimental_input_hashes=before, generation=generation,
                                                             builder_sha256=digest(Path(__file__)), template_sha256=digest(HERE / "tools/eval_v00_report.html"),
                                                             chart_count=8, experimental_results_changed=False), indent=2))
    print(f"Built {RESULT / 'index.html'} and 8 charts in SVG + PNG; numerical evidence unchanged.")


if __name__ == "__main__":
    main()
