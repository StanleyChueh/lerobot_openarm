"""REAL/SIM COMPATIBILITY: can two datasets be mixed (co-training) or a policy moved between them?

Schema (names, order, shapes, cameras, fps, robot_type) must match exactly; distributions are compared
per dimension and reported, since a shift is expected between domains and is not by itself a defect.
"""

from __future__ import annotations

import numpy as np

from .checks import FAIL, PASS, WARN, Check
from .loader import DatasetView


def _per_frame_speed(view: DatasetView, key: str = "observation.state", dims: list[int] | None = None) -> np.ndarray:
    v = []
    for s in view.vectors(key).values():
        if len(s) > 1:
            d = np.abs(np.diff(s[:, dims] if dims is not None else s, axis=0))
            v.append(d.max(1))
    return np.concatenate(v) if v else np.zeros(0)


def _action_lead(view: DatasetView, dims: list[int], max_lag: int = 6) -> float | None:
    """Frames by which the action leads the measured state (argmin over lags of mean |a[t] - s[t+lag]|)."""
    errs = np.zeros(max_lag + 1)
    acts, sts = view.vectors("action"), view.vectors("observation.state")
    n = 0
    for e, a in acts.items():
        s = sts[e]
        if len(a) <= max_lag + 1:
            continue
        for lag in range(max_lag + 1):
            errs[lag] += np.abs(a[: len(a) - lag, dims] - s[lag:, dims]).mean()
        n += 1
    return int(np.argmin(errs)) if n else None


def compare_datasets(a: DatasetView, b: DatasetView, *, arm_prefixes=("RJ", "LJ"),
                     gripper_names=("RJ8.pos", "LJ8.pos"), hold_levels: dict | None = None) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    tables: dict = {}
    la, lb = a.label, b.label
    # ------------------------------------------------------------ schema
    for key in ("observation.state", "action"):
        na, nb = a.names(key), b.names(key)
        status = PASS if na == nb else FAIL
        checks.append(Check(f"schema_{key}", status, "identical names and order" if status == PASS else
                            f"{la} {na} vs {lb} {nb}", {la: na, lb: nb}))
    ca = {c: a.features[c].get("shape") for c in a.camera_keys}
    cb = {c: b.features[c].get("shape") for c in b.camera_keys}
    checks.append(Check("schema_cameras", PASS if ca == cb else FAIL,
                        "same camera keys and shapes" if ca == cb else f"{la} {ca} vs {lb} {cb}", {la: ca, lb: cb}))
    meta_diff = {k: (a.info.get(k), b.info.get(k)) for k in ("fps", "robot_type", "codebase_version")
                 if a.info.get(k) != b.info.get(k)}
    checks.append(Check("schema_meta", PASS if not meta_diff else (FAIL if "fps" in meta_diff or "robot_type" in meta_diff
                                                                  else WARN),
                        "fps, robot_type and version agree" if not meta_diff else f"differs: {meta_diff}", meta_diff))
    if any(c.status == FAIL for c in checks):
        return checks, tables

    names = a.names("observation.state")
    arm = [i for i, n in enumerate(names) if n.startswith(arm_prefixes) and n not in gripper_names]
    grip = [names.index(g) for g in gripper_names if g in names]

    # ------------------------------------------------------------ state/action distributions
    for key in ("observation.state", "action"):
        xa = np.stack(a.data[key].values).astype(float)
        xb = np.stack(b.data[key].values).astype(float)
        sa, sb = xa.std(0), xb.std(0)
        pooled = np.sqrt((sa**2 + sb**2) / 2) + 1e-6
        smd = (xb.mean(0) - xa.mean(0)) / pooled
        ratio = (sb + 1e-6) / (sa + 1e-6)
        q01, q99 = np.quantile(xb, 0.01, 0), np.quantile(xb, 0.99, 0)
        out_frac = ((xa < q01) | (xa > q99)).mean(0)  # share of A frames outside B's 1-99% band
        rows = [{"dim": names[i], f"mean_{la}": round(float(xa[:, i].mean()), 4),
                 f"mean_{lb}": round(float(xb[:, i].mean()), 4), "std_mean_diff": round(float(smd[i]), 3),
                 f"std_ratio_{lb}/{la}": round(float(ratio[i]), 3),
                 f"frac_{la}_outside_{lb}_1-99%": round(float(out_frac[i]), 3)} for i in range(len(names))]
        tables[key] = rows
        big = [names[i] for i in range(len(names)) if abs(smd[i]) > 1.0 or ratio[i] > 2 or ratio[i] < 0.5]
        checks.append(Check(f"distribution_{key}", WARN if big else PASS,
                            f"large shift (|SMD|>1 or std ratio outside 0.5-2) in {big}" if big else
                            "per-dimension means/stds comparable", {"shifted_dims": big}))

    # ------------------------------------------------------------ grippers
    if grip:
        ga = {names[j]: [round(float(x), 3) for x in np.quantile(np.stack(a.data["observation.state"].values)[:, j],
                                                                   [0.01, 0.5, 0.99])] for j in grip}
        gb = {names[j]: [round(float(x), 3) for x in np.quantile(np.stack(b.data["observation.state"].values)[:, j],
                                                                   [0.01, 0.5, 0.99])] for j in grip}
        ev = {f"{la}_state_q01_q50_q99": ga, f"{lb}_state_q01_q50_q99": gb}
        if hold_levels:
            ev["hold_openness"] = hold_levels
        diff = []
        if hold_levels and la in hold_levels and lb in hold_levels:
            for side in ("right", "left"):
                ha, hb = hold_levels[la].get(f"{side}_hold_openness_median"), hold_levels[lb].get(
                    f"{side}_hold_openness_median")
                if ha is not None and hb is not None and abs(ha - hb) > 0.1:
                    diff.append(f"{side} hold openness {la} {ha:.2f} vs {lb} {hb:.2f}")
        checks.append(Check("gripper_semantics", WARN if diff else PASS,
                            "; ".join(diff) + " -- the policy sees a different gripper state while holding"
                            if diff else "gripper ranges/hold levels comparable", ev))

    # ------------------------------------------------------------ timing / control rate
    va, vb = _per_frame_speed(a, dims=arm), _per_frame_speed(b, dims=arm)
    moving_a, moving_b = va[va > 1e-3], vb[vb > 1e-3]
    if len(moving_a) and len(moving_b):
        med_a, med_b = float(np.median(moving_a)), float(np.median(moving_b))
        r = med_b / med_a
        checks.append(Check("joint_speed_per_frame", WARN if r > 1.4 or r < 0.7 else PASS,
                            f"median moving arm step {la} {med_a:.4f} vs {lb} {med_b:.4f} rad/frame (ratio {r:.2f}); "
                            "a ~1.5x ratio is the signature of data recorded at 20 Hz but labelled 30 fps",
                            {"ratio": r}))
    lens_a = a.data.groupby("episode_index").size() / a.fps
    lens_b = b.data.groupby("episode_index").size() / b.fps
    checks.append(Check("episode_duration", PASS,
                        f"{la} {lens_a.mean():.1f}±{lens_a.std():.1f} s, {lb} {lens_b.mean():.1f}±{lens_b.std():.1f} s",
                        {}))
    lead_a, lead_b = _action_lead(a, arm), _action_lead(b, arm)
    checks.append(Check("action_state_lead", PASS if lead_a == lead_b else WARN,
                        f"action leads state by {lead_a} frames in {la}, {lead_b} in {lb}", {la: lead_a, lb: lead_b}))
    sa0 = np.median(np.stack([s[0] for s in a.vectors("observation.state").values()]), 0)
    sb0 = np.median(np.stack([s[0] for s in b.vectors("observation.state").values()]), 0)
    d0 = float(np.abs(sa0[arm] - sb0[arm]).max())
    checks.append(Check("start_pose", PASS if d0 < 0.05 else WARN,
                        f"median start arm pose differs by up to {d0:.3f} rad", {"max_abs_rad": d0}))
    return checks, tables
