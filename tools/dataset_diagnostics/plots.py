"""Static figures (PNG) for the readiness report. Every figure has a CSV/JSON table next to it."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

# Categorical slots in fixed order (dataviz reference palette): dataset 1, dataset 2, recommendations.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf",
                                                     "#184f95", "#0d366b"])
SEQ = SEQ.with_extremes(bad="#d9d8d4")  # unreachable cells: neutral, not "zero"


def _style(ax, title, cov):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    u = cov.config.units
    if cov.config.frame_type == "image":
        ax.set_xlabel(f"u ({u})", color=INK2, fontsize=9)
        ax.set_ylabel(f"v ({u})", color=INK2, fontsize=9)
        ax.set_xlim(*cov.config.x_range)
        ax.set_ylim(cov.config.y_range[1], cov.config.y_range[0])  # image rows grow downward
    else:
        ax.set_xlabel(f"x ({u}) [{cov.config.frame_id}]", color=INK2, fontsize=9)
        ax.set_ylabel(f"y ({u})", color=INK2, fontsize=9)
        ax.set_xlim(*cov.config.x_range)
        ax.set_ylim(*cov.config.y_range)
    ax.set_aspect("equal")


def _overlay_workspace(ax, cov):
    cfg = cov.config
    if cfg.polygon:
        p = np.asarray(cfg.polygon + [cfg.polygon[0]])
        ax.plot(p[:, 0], p[:, 1], color=INK2, lw=1, ls="--")
    for x0, x1, y0, y1 in cfg.keep_out:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=INK2, hatch="//", lw=0.8, alpha=0.6))
    for name, r in cfg.regions.items():
        if "center" in r:
            ax.add_patch(Circle(r["center"], r["radius"], fc="none", ec=INK2, lw=0.8, ls=":"))
            ax.annotate(name, r["center"], fontsize=7, color=INK2, ha="center", va="center")
        else:
            x0, x1, y0, y1 = r["box"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=INK2, lw=0.8, ls=":"))
            ax.annotate(name, (x0, y1), fontsize=7, color=INK2)


def plot_positions(covs: list, labels: list[str], path: str, title: str, recommendations: bool = True) -> None:
    """Starting XY scatter (one frame), with duplicate groups ringed and recommendations as diamonds."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _style(ax, title, covs[0])
    _overlay_workspace(ax, covs[0])
    for k, (cov, label) in enumerate(zip(covs, labels)):
        if len(cov.points):
            ax.scatter(cov.points[:, 0], cov.points[:, 1], s=46, color=SERIES[k], edgecolor=SURFACE, lw=1.5,
                       label=f"{label} (n={len(cov.points)})", zorder=3)
            if len(covs) == 1:
                for e, (x, y) in zip(cov.episodes, cov.points):
                    ax.annotate(str(e), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
        for g in cov.duplicates:
            idx = [cov.episodes.index(e) for e in g]
            c = cov.points[idx].mean(0)
            ax.add_patch(Circle(c, cov.config.duplicate_radius, fc="none", ec=INK, lw=1))
    if recommendations and covs[0].recommendations:
        r = np.array([[d["x"], d["y"]] for d in covs[0].recommendations])
        ax.scatter(r[:, 0], r[:, 1], marker="D", s=52, color=SERIES[2], edgecolor=INK, lw=0.8,
                   label="recommended next", zorder=4)
        for d in covs[0].recommendations:
            ax.annotate(f"R{d['rank']}", (d["x"], d["y"]), xytext=(5, -9), textcoords="offset points", fontsize=7.5,
                        color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_density(cov, label: str, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ext = [cov.config.x_range[0], cov.config.x_range[1], cov.config.y_range[0], cov.config.y_range[1]]
    occ = np.where(cov.reachable.reshape(cov.occupancy.shape), cov.occupancy, np.nan)
    for ax, grid, ttl, cb in ((axes[0], occ, f"{label}: demos per {cov.config.cell_size:g} {cov.config.units} cell",
                               "count"),
                              (axes[1], cov.density, f"{label}: smoothed density (bw {cov.config.kde_bandwidth or cov.config.coverage_radius:g})",
                               f"demos / {cov.config.units}²")):
        _style(ax, ttl, cov)
        origin = "upper" if cov.config.frame_type == "image" else "lower"
        im = ax.imshow(grid, extent=ext if origin == "lower" else [ext[0], ext[1], ext[3], ext[2]], origin=origin,
                       cmap=SEQ, interpolation="nearest", aspect="equal")
        _style(ax, ttl, cov)
        if len(cov.points):
            ax.scatter(cov.points[:, 0], cov.points[:, 1], s=10, color=INK, zorder=3)
        _overlay_workspace(ax, cov)
        c = fig.colorbar(im, ax=ax, shrink=0.8)
        c.set_label(cb, fontsize=8, color=INK2)
        c.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_nearest(cov, label: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ext = [cov.config.x_range[0], cov.config.x_range[1], cov.config.y_range[0], cov.config.y_range[1]]
    origin = "upper" if cov.config.frame_type == "image" else "lower"
    im = ax.imshow(cov.nn_dist, extent=ext if origin == "lower" else [ext[0], ext[1], ext[3], ext[2]], origin=origin,
                   cmap=SEQ, interpolation="nearest", aspect="equal")
    _style(ax, f"{label}: distance to nearest demonstration (reachable cells)", cov)
    xs, ys = cov.xs, cov.ys
    if np.isfinite(cov.nn_dist).any():
        ax.contour(xs, ys, cov.nn_dist, levels=[cov.config.coverage_radius], colors=[INK], linewidths=1)
    if len(cov.points):
        ax.scatter(cov.points[:, 0], cov.points[:, 1], s=12, color=SURFACE, edgecolor=INK, lw=0.8, zorder=3)
    for d in cov.recommendations:
        ax.scatter([d["x"]], [d["y"]], marker="D", s=40, color=SERIES[2], edgecolor=INK, lw=0.8, zorder=4)
        ax.annotate(f"R{d['rank']}", (d["x"], d["y"]), xytext=(5, -9), textcoords="offset points", fontsize=7.5)
    _overlay_workspace(ax, cov)
    c = fig.colorbar(im, ax=ax, shrink=0.85)
    c.set_label(f"{cov.config.units} (contour = coverage radius {cov.config.coverage_radius:g})", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_real_sim(tables: dict, la: str, lb: str, joint_speed: dict, path: str) -> None:
    """Per-dimension standardized mean difference (state & action) + per-frame arm speed distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for ax, key in zip(axes[:2], ("observation.state", "action")):
        rows = tables.get(key, [])
        if not rows:
            ax.set_visible(False)
            continue
        dims = [r["dim"] for r in rows]
        smd = [r["std_mean_diff"] for r in rows]
        y = np.arange(len(dims))
        ax.barh(y, smd, color=[SERIES[1] if v > 0 else SERIES[0] for v in smd], height=0.7)
        ax.axvline(0, color=INK2, lw=0.8)
        for t in (-1, 1):
            ax.axvline(t, color=INK2, lw=0.6, ls=":")
        ax.set_yticks(y, dims, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(f"{key}: (mean {lb} − mean {la}) / pooled std", fontsize=10, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=7.5)
    ax = axes[2]
    for k, (lab, v) in enumerate(joint_speed.items()):
        v = v[v > 1e-3]
        if len(v):
            ax.hist(v, bins=np.linspace(0, np.quantile(np.concatenate([x[x > 1e-3] for x in joint_speed.values()]),
                                                       0.99), 40),
                    histtype="step", lw=2, color=SERIES[k], density=True, label=f"{lab} (median {np.median(v):.4f})")
    ax.set_title("moving arm joint step per frame (rad)", fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_annotation_overlay(frames: list, path: str, cols: int = 5) -> None:
    """Grid of starting frames with the annotated/proposed point, for human review."""
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (ep, img, p, tag) in zip(np.atleast_1d(axes).ravel(), frames):
        ax.imshow(img)
        if p is not None:
            ax.scatter([p[0]], [p[1]], s=80, marker="+", color="#00ff66", lw=2)
        ax.set_title(f"ep {ep} {tag}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90, facecolor="white", bbox_inches="tight")
    plt.close(fig)
