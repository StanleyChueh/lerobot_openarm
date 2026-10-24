"""Workspace coverage of initial object positions, and where to put the next demonstrations.

Works in one declared 2-D frame at a time: metric ("metric", metres, e.g. a robot base frame) or image
("image", pixels of one camera). The two are never combined. Only REACHABLE cells (inside the declared safe
boundary, outside keep-out regions) count; an unreachable cell is never reported as missing.

Coverage is measured against a target distribution over the reachable workspace -- uniform by default, or the
evaluation distribution the user supplies -- so "under-covered" always means "under-covered for that target".
Nothing here predicts training success.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .positions import Position


@dataclass
class WorkspaceConfig:
    name: str
    frame_type: str  # "metric" | "image"
    frame_id: str  # e.g. "openarm_world" or "image:observation.images.body_cam"
    units: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    cell_size: float
    coverage_radius: float  # a target point counts as covered if a demo lies within this distance
    duplicate_radius: float  # two starts closer than this are near-duplicates (one independent example)
    min_separation: float  # recommendations keep at least this far from demos and each other
    polygon: list[list[float]] | None = None  # optional safe boundary (inside = reachable)
    keep_out: list[list[float]] = field(default_factory=list)  # [x0, x1, y0, y1] boxes, unreachable
    eval_distribution: dict = field(default_factory=lambda: {"type": "uniform"})
    regions: dict[str, dict] = field(default_factory=dict)  # name -> {"center": [x, y], "radius": r} | {"box": [..]}
    kde_bandwidth: float | None = None
    thresholds: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path_or_dict) -> "WorkspaceConfig":
        d = json.loads(Path(path_or_dict).read_text()) if not isinstance(path_or_dict, dict) else dict(path_or_dict)
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        if d.get("frame_type") not in ("metric", "image"):
            raise ValueError("workspace config: frame_type must be 'metric' or 'image'")
        for k in ("x_range", "y_range"):
            lo, hi = d[k]
            if not hi > lo:
                raise ValueError(f"workspace config: {k} must be [low, high] with high > low")
        for k in ("cell_size", "coverage_radius", "duplicate_radius", "min_separation"):
            if float(d[k]) <= 0:
                raise ValueError(f"workspace config: {k} must be > 0")
        d["x_range"], d["y_range"] = tuple(d["x_range"]), tuple(d["y_range"])
        return cls(**d)

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.arange(self.x_range[0] + self.cell_size / 2, self.x_range[1], self.cell_size)
        ys = np.arange(self.y_range[0] + self.cell_size / 2, self.y_range[1], self.cell_size)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        return xs, ys, np.c_[X.ravel(), Y.ravel()]

    def reachable(self, pts: np.ndarray) -> np.ndarray:
        ok = ((pts[:, 0] >= self.x_range[0]) & (pts[:, 0] <= self.x_range[1])
              & (pts[:, 1] >= self.y_range[0]) & (pts[:, 1] <= self.y_range[1]))
        if self.polygon:
            from matplotlib.path import Path as MplPath

            ok &= MplPath(np.asarray(self.polygon, float)).contains_points(pts)
        for x0, x1, y0, y1 in self.keep_out:
            ok &= ~((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        return ok

    def target_weights(self, pts: np.ndarray, reach: np.ndarray) -> np.ndarray:
        """Target (evaluation) density over grid cells, zero outside the reachable set, sums to 1."""
        ed = self.eval_distribution or {"type": "uniform"}
        t = ed.get("type", "uniform")
        if t == "uniform":
            w = np.ones(len(pts))
        elif t == "points":  # mixture of isotropic Gaussians around evaluation positions
            sig = float(ed.get("sigma", self.coverage_radius))
            w = np.zeros(len(pts))
            for p in ed["points"]:
                wt = p[2] if len(p) > 2 else 1.0
                w += wt * np.exp(-0.5 * np.sum((pts - np.asarray(p[:2])) ** 2, 1) / sig**2)
        elif t == "gaussian":
            mu, sd = np.asarray(ed["mean"], float), np.asarray(ed["std"], float)
            w = np.exp(-0.5 * np.sum(((pts - mu) / sd) ** 2, 1))
        else:
            raise ValueError(f"unknown eval_distribution type {t!r}")
        w = np.where(reach, w, 0.0)
        if w.sum() <= 0:
            raise ValueError("eval distribution puts no weight on reachable cells")
        return w / w.sum()


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1), 0))


def near_duplicate_groups(pts: np.ndarray, radius: float) -> list[list[int]]:
    """Single-linkage groups of points closer than `radius` (union-find)."""
    n = len(pts)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1:
        d = _pairwise(pts, pts)
        for i, j in zip(*np.where(np.triu(d < radius, 1))):
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def kernel_effective_n(pts: np.ndarray, bandwidth: float) -> float:
    """n_eff = n^2 / sum_ij K(d_ij), Gaussian K: equals n for well-separated samples, 1 for identical ones."""
    if len(pts) == 0:
        return 0.0
    k = np.exp(-0.5 * (_pairwise(pts, pts) / bandwidth) ** 2)
    return float(len(pts) ** 2 / k.sum())


@dataclass
class CoverageResult:
    config: WorkspaceConfig
    episodes: list[int]
    points: np.ndarray  # (n, 2) positions in the config frame
    xs: np.ndarray
    ys: np.ndarray
    grid_pts: np.ndarray
    reachable: np.ndarray
    target: np.ndarray
    occupancy: np.ndarray  # (len(ys), len(xs)) counts
    density: np.ndarray  # (len(ys), len(xs)) KDE, demos per unit area
    nn_dist: np.ndarray  # (len(ys), len(xs)) distance to the nearest demo (nan outside reachable)
    metrics: dict
    duplicates: list[list[int]]
    regions: dict
    outside: list[int]
    recommendations: list[dict]


def analyze_coverage(positions: dict[int, Position], cfg: WorkspaceConfig, *, n_recommend: int = 5,
                     use_image: bool | None = None) -> CoverageResult:
    use_image = cfg.frame_type == "image" if use_image is None else use_image
    eps, pts = [], []
    for e, p in sorted(positions.items()):
        if use_image and p.has_image and (cfg.frame_id in (p.camera, f"image:{p.camera}")):
            eps.append(e)
            pts.append((p.u, p.v))
        elif not use_image and p.has_metric and p.frame_id == cfg.frame_id:
            eps.append(e)
            pts.append((p.x, p.y))
    pts = np.asarray(pts, float).reshape(-1, 2)
    xs, ys, G = cfg.grid()
    reach = cfg.reachable(G)
    target = cfg.target_weights(G, reach)
    inside = cfg.reachable(pts) if len(pts) else np.zeros(0, bool)
    outside = [eps[i] for i in np.where(~inside)[0]]

    shape = (len(ys), len(xs))
    occ = np.zeros(shape)
    if len(pts):
        ix = np.clip(((pts[:, 0] - cfg.x_range[0]) / cfg.cell_size).astype(int), 0, len(xs) - 1)
        iy = np.clip(((pts[:, 1] - cfg.y_range[0]) / cfg.cell_size).astype(int), 0, len(ys) - 1)
        for a, b, ok in zip(iy, ix, inside):
            if ok:
                occ[a, b] += 1
    bw = cfg.kde_bandwidth or cfg.coverage_radius
    if len(pts):
        K = np.exp(-0.5 * (_pairwise(G, pts) / bw) ** 2) / (2 * np.pi * bw**2)
        dens = K.sum(1)
        nn = _pairwise(G, pts).min(1)
    else:
        dens = np.zeros(len(G))
        nn = np.full(len(G), np.inf)
    dens = np.where(reach, dens, np.nan).reshape(shape)
    nn_grid = np.where(reach, nn, np.nan).reshape(shape)

    dups = near_duplicate_groups(pts, cfg.duplicate_radius) if len(pts) else []
    dup_groups = [[eps[i] for i in g] for g in dups if len(g) > 1]
    covered = nn <= cfg.coverage_radius
    reach_w = reach.astype(float) / max(reach.sum(), 1)
    metrics = {
        "n_with_position": int(len(pts)),
        "n_inside_reachable": int(inside.sum()) if len(pts) else 0,
        "n_independent_groups": len(dups),
        "kernel_effective_n": round(kernel_effective_n(pts[inside], cfg.coverage_radius), 2) if len(pts) else 0.0,
        "target_covered_fraction": round(float((target * covered).sum()), 4),
        "reachable_area_covered_fraction": round(float((reach_w * covered).sum()), 4),
        "target_mean_nn_distance": round(float((target * np.where(np.isfinite(nn), nn, 0)).sum()), 4)
        if len(pts) else None,
        "covering_radius_reachable": round(float(nn[reach].max()), 4) if len(pts) and reach.any() else None,
        "bbox": [float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())]
        if len(pts) else None,
        "spread_std": [round(float(s), 4) for s in pts.std(0)] if len(pts) > 1 else None,
        "units": cfg.units, "frame_id": cfg.frame_id, "coverage_radius": cfg.coverage_radius,
        "duplicate_radius": cfg.duplicate_radius,
    }
    regions = {}
    for name, r in cfg.regions.items():
        if "center" in r:
            d = np.linalg.norm(pts - np.asarray(r["center"]), axis=1) if len(pts) else np.zeros(0)
            member = d <= r["radius"]
        else:
            x0, x1, y0, y1 = r["box"]
            member = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)) \
                if len(pts) else np.zeros(0, bool)
        idx = np.where(member)[0]
        indep = len(near_duplicate_groups(pts[idx], cfg.duplicate_radius)) if len(idx) else 0
        regions[name] = {"episodes": [eps[i] for i in idx], "count": int(len(idx)), "independent": indep}
    recs = recommend_positions(pts[inside] if len(pts) else pts, G, reach, target, cfg, n_recommend, regions)
    return CoverageResult(cfg, eps, pts, xs, ys, G, reach, target, occ, dens, nn_grid, metrics, dup_groups, regions,
                          outside, recs)


def coverage_fragility(points: np.ndarray, cfg: WorkspaceConfig) -> dict:
    """How much the target-covered fraction depends on single demonstrations (leave-one-out). With few demos
    this is the honest uncertainty of the coverage number: it says how far it would move if one demo were bad."""
    if len(points) < 2:
        return {"loo_min": None, "loo_max": None, "most_critical_index": None}
    _, _, G = cfg.grid()
    reach = cfg.reachable(G)
    target = cfg.target_weights(G, reach)
    vals = []
    for i in range(len(points)):
        keep = np.delete(points, i, 0)
        vals.append(float((target * (_pairwise(G, keep).min(1) <= cfg.coverage_radius)).sum()))
    vals = np.asarray(vals)
    return {"loo_min": round(float(vals.min()), 4), "loo_max": round(float(vals.max()), 4),
            "most_critical_index": int(np.argmin(vals))}


def recommend_positions(pts: np.ndarray, G: np.ndarray, reach: np.ndarray, target: np.ndarray,
                        cfg: WorkspaceConfig, k: int, regions: dict | None = None) -> list[dict]:
    """Greedy: maximise target weight x (distance to nearest demo, capped at 3 coverage radii), only among
    reachable cells at least `min_separation` from every demo and every earlier recommendation."""
    chosen: list[np.ndarray] = []
    have = pts.copy()
    tnorm = target / max(target.max(), 1e-12)
    out = []
    for rank in range(k):
        d = _pairwise(G, have).min(1) if len(have) else np.full(len(G), 3 * cfg.coverage_radius)
        ok = reach & (d >= cfg.min_separation) & (target > 0)
        if not ok.any():
            break
        score = np.where(ok, tnorm * np.minimum(d, 3 * cfg.coverage_radius), -np.inf)
        i = int(np.argmax(score))
        p = G[i]
        in_regions = [n for n, r in (regions or {}).items()
                      if ("center" in r and np.linalg.norm(p - np.asarray(cfg.regions[n]["center"]))
                          <= cfg.regions[n]["radius"])
                      or ("box" in cfg.regions.get(n, {}) and cfg.regions[n]["box"][0] <= p[0] <= cfg.regions[n]["box"][1]
                          and cfg.regions[n]["box"][2] <= p[1] <= cfg.regions[n]["box"][3])]
        reasons = [f"nearest demonstration {d[i]:.3g} {cfg.units} away"
                   + (" (outside coverage radius)" if d[i] > cfg.coverage_radius else ""),
                   f"target-weight percentile {100 * (target[reach] <= target[i]).mean():.0f}"]
        for n in in_regions:
            reasons.append(f"region '{n}' has {regions[n]['independent']} independent demos")
        out.append({"rank": rank + 1, "x": round(float(p[0]), 4), "y": round(float(p[1]), 4), "units": cfg.units,
                    "frame_id": cfg.frame_id, "nearest_demo_distance": round(float(d[i]), 4),
                    "target_weight_rel": round(float(tnorm[i]), 3), "regions": in_regions,
                    "reason": "; ".join(reasons)})
        chosen.append(p)
        have = np.vstack([have, p[None]]) if len(have) else p[None].copy()
    return out
