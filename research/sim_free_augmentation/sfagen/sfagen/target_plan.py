"""Absolute planar object targets, assigned to the nearest eligible source demos."""
import numpy as np


def target_offsets(rec, cfg):
    name = cfg["object"]
    obj = next(o for o in rec.objects if o["name"] == name)
    targets = np.asarray(cfg["positions_xy_m"], float)
    sources = {int(e): np.asarray(p, float) for e, p in cfg["source_positions_xy_m"].items()}
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
        raise ValueError("Object targets must be finite world x/y positions in metres")
    if len(np.unique(targets, axis=0)) != len(targets):
        raise ValueError("Duplicate object targets")
    if not sources or any(p.shape != (2,) or not np.isfinite(p).all() for p in sources.values()):
        raise ValueError("Source positions must be finite world x/y positions")
    if rec.e not in sources or not np.allclose(sources[rec.e], np.asarray(obj["table"])[:2, 3], atol=1e-9, rtol=0):
        raise ValueError("Frozen target-assignment source position differs from the reconstruction")
    cap = float(cfg.get("max_shift_m", .08)); minimum = float(cfg.get("min_shift_m", .005)); nearest = int(cfg.get("nearest_sources", 3))
    if not 0 <= minimum < cap or nearest < 1:
        raise ValueError("Invalid target source cap/count")
    planner = cfg.get("planner", "f1")
    if planner not in ("f1", "f1w"):
        raise ValueError("object_targets.planner must be f1 or f1w")
    family = "object_workspace" if planner == "f1w" else "object_position"   # f1w: generate.py uses plan_with_time_scale
    out = []; only = cfg.get("indices"); suffix = str(cfg.get("id_suffix", ""))
    for index, target in enumerate(targets):
        if only is not None and index not in only:
            continue
        ranked = sorted((float(np.linalg.norm(target - p)), e) for e, p in sources.items()
                        if minimum <= np.linalg.norm(target - p) <= cap + 1e-12)
        if rec.e in [e for _, e in ranked[:nearest]]:
            out.append((f"{name}_target_{index:03d}{suffix}", family, name, np.r_[target - sources[rec.e], 0.]))
    return out
