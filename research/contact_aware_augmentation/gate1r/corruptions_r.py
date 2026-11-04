"""Gate 1R corruption suite and counterexample fixtures (deterministic). Each sample carries the full joint stream
(measured state q and command a), the object track and its evidence mask, so every verifier sees the same input.

Labels: "valid" (original and hard negatives that a correct verifier must not reject), "invalid" (known by
construction), "ambiguous" (rigid offsets that are indistinguishable from a different legitimate grasp without
measured geometry; reported separately, never counted as detections or false alarms).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gate1")]
import gate1_core as gc  # noqa: E402
import verifier_r as vr  # noqa: E402
from gate1_core import g0  # noqa: E402

ARM_OFFSETS = (0.05, 0.1, 0.25, 0.5, 1.0)
FRAMES = (3, 6, 10)
EXCURSION_PX = (15, 30, 60)
EXCURSION_CM = (1, 2, 4)
MISSING_FRAC = (0.25, 0.5, 0.75, 1.0)


def _clone(s: vr.Sample, name, family, label, **kw) -> vr.Sample:
    c = copy.deepcopy(s)
    c.name, c.family, c.label = name, family, label
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def swap_lr(x: np.ndarray, names: list) -> np.ndarray:
    y = x.copy()
    for k in range(1, 9):
        i, j = names.index(f"RJ{k}.pos"), names.index(f"LJ{k}.pos")
        y[:, i], y[:, j] = x[:, j], x[:, i]
    return y


def gate1_windows(s: vr.Sample, tail: int):
    """The Gate 1 verifier's own windows (Codex's construction of the excursion uses them)."""
    ev = vr.command_events(s)
    gs = gc.Sample("x", "x", True, 0, 0, None, {"right_close": ev["raw"]["right"][0][0], "left_close": ev["raw"]["left"][0][0],
                                                 "right_open": ev["raw"]["right"][0][1]}, s.uv, s.ok)
    return gc.windows(gs, tail)


def lid_world(s: vr.Sample, poses, cam):
    """3-D lid estimate per frame from attachments fitted on the ORIGINAL sample (used only to construct
    perspective-correct injections; it is the same nominal model the verifier uses, so it is a model stress test)."""
    ev = vr.command_events(s)
    T = len(s.uv)
    w = gc.WINDOW
    gwin = np.arange(ev["g_close"] + w["giver_start"], ev["r_close"])
    rwin = np.arange(ev["g_open"] + 8, min(ev["g_open"] + 38, T))
    fits = {}
    for key, hand, idx in (("giver", ev["giver"], gwin), ("recv", ev["receiver"], rwin)):
        sel = idx[s.ok[idx]]
        fits[key] = gc.fit_point(poses[hand][sel], s.uv[sel], cam)["p"]
    X = np.full((T, 3), np.nan)
    for t in range(ev["g_close"], T):
        h, p = (ev["giver"], fits["giver"]) if t < ev["g_open"] else (ev["receiver"], fits["recv"])
        X[t] = poses[h][t, :3, :3] @ p + poses[h][t, :3, 3]
    return X, fits


def suite(orig: vr.Sample, poses, cam, tail: int, other_tracks: dict, rng_seed: int = 0) -> list[vr.Sample]:
    """All Gate 1R samples for one episode. other_tracks: {episode: (uv, ok, T)} for cross-episode pairs."""
    rng = np.random.default_rng(rng_seed + orig.joints_ep)
    names = orig.names
    arm = [i for i, n in enumerate(names) if not n.endswith("8.pos")]
    T = len(orig.uv)
    ev = vr.command_events(orig)
    seg = vr.segments(ev, T)
    out = [orig]
    # ---- valid hard negatives
    ok = orig.ok.copy()
    drop = rng.choice(np.where(ok)[0], size=int(0.10 * ok.sum()), replace=False)
    ok[drop] = False
    out.append(_clone(orig, "hardneg_dropout_10pct", "hard_negative", "valid", ok=ok))
    out.append(_clone(orig, "hardneg_noise_1p5px", "hard_negative", "valid", uv=orig.uv + rng.normal(0, 1.5, orig.uv.shape)))
    # ---- Codex F1 and action families (video and measured state untouched)
    for c in ARM_OFFSETS:
        a = orig.a.copy(); a[:, arm] += c
        out.append(_clone(orig, f"action_offset_all_{c:g}rad", "action_offset_all", "invalid", a=a, meta={"rad": c}))
        a = orig.a.copy(); a[:, names.index("RJ4.pos")] += c
        out.append(_clone(orig, f"action_offset_RJ4_{c:g}rad", "action_offset_joint", "invalid", a=a, meta={"rad": c}))
    for k in FRAMES:
        for sign, tag in ((+1, "lag"), (-1, "lead")):
            src = np.clip(np.arange(T) - sign * k, 0, T - 1)
            out.append(_clone(orig, f"action_time_{tag}_{k}", "action_time", "invalid", a=orig.a[src].copy(), meta={"k": sign * k}))
    out.append(_clone(orig, "action_swap_LR", "action_swap", "invalid", a=swap_lr(orig.a, names)))
    # ---- Codex F2: excursion confined to the Gate 1 unscored gap, and 3-D transition excursions
    w1 = gate1_windows(orig, tail)
    start, stop = int(w1["giver"][-1]) + 1, int(w1["recv"][0])
    shape = np.sin(np.linspace(0, np.pi, stop - start + 2))[1:-1] ** 2
    for px in EXCURSION_PX:
        uv = orig.uv.copy(); uv[start:stop, 0] += px * shape
        out.append(_clone(orig, f"excursion_gap_{px}px", "excursion_px", "invalid", uv=uv, meta={"px": px, "frames": [start, stop - 1]}))
    X, fits = lid_world(orig, poses, cam)
    tw = seg["transition"]
    shp = np.sin(np.linspace(0, np.pi, len(tw) + 2))[1:-1] ** 2
    for cm in EXCURSION_CM:
        uv = orig.uv.copy()
        dX = np.zeros((len(tw), 3)); dX[:, 0] = cm / 100.0 * shp
        uv[tw] = uv[tw] + g0.project(X[tw] + dX, cam)[0] - g0.project(X[tw], cam)[0]
        out.append(_clone(orig, f"excursion_transition_{cm}cm", "excursion_3d", "invalid", uv=uv, meta={"cm": cm}))
    # ---- Codex F3 and graded missing evidence (the object track itself is unchanged)
    rseg = np.concatenate([seg["recv_calib"], seg["recv_pred"]])
    for f in MISSING_FRAC:
        ok = orig.ok.copy(); ok[rseg[: int(round(f * len(rseg)))]] = False
        out.append(_clone(orig, f"missing_receiver_{int(f * 100)}pct", "missing_evidence", "invalid", ok=ok, meta={"frac": f}))
    ok = orig.ok.copy(); ok[w1["recv"]] = False
    out.append(_clone(orig, "missing_receiver_gate1_window", "missing_evidence", "invalid", ok=ok))
    ok = orig.ok.copy(); ok[seg["transition"]] = False
    out.append(_clone(orig, "missing_transition_100pct", "missing_evidence", "invalid", ok=ok))
    ok = orig.ok.copy(); ok[seg["giver_pred"]] = False
    out.append(_clone(orig, "missing_giver_hold_100pct", "missing_evidence", "invalid", ok=ok))
    # ---- Gate 1 families (track edits from gate1_core.inject, perspective-correct, continuity-fixed)
    E = gc.Episode(orig.joints_ep, orig.q, orig.a, names, poses,
                   {"right_close": ev["raw"]["right"][0][0], "left_close": ev["raw"]["left"][0][0], "right_open": ev["raw"]["right"][0][1]},
                   orig.uv, orig.ok)
    for g in gc.inject(E, {"giver": fits["giver"], "recv": fits["recv"]}, cam):
        if g.family in ("phase",):
            continue  # re-expressed below as command edits, so the joint stream stays one consistent object
        label = "ambiguous" if g.family in ("displacement_hand", "displacement_px") else "invalid"
        out.append(_clone(orig, g.name, g.family, label, uv=g.uv.copy(), ok=g.ok.copy()))
    # ---- phase errors as edits of the gripper COMMAND columns (measured jaw state unchanged)
    gi, ri = names.index(vr.GRIP[ev["giver"]][0]), names.index(vr.GRIP[ev["receiver"]][0])
    gopen, ropen = vr.GRIP[ev["giver"]][1], vr.GRIP[ev["receiver"]][1]
    a = orig.a.copy(); a[ev["r_close"] - 10:, gi] = gopen
    out.append(_clone(orig, "phase_giver_releases_10_before_receiver", "phase", "invalid", a=a))
    a = orig.a.copy()
    a[ev["r_close"] - 15: ev["r_close"], ri] = vr.GRIP[ev["receiver"]][2]
    a[ev["g_open"] - 15: ev["g_open"], gi] = gopen
    out.append(_clone(orig, "phase_receiver_window_shifted_early_15", "phase", "invalid", a=a))
    out.append(_clone(orig, "phase_roles_swapped", "phase", "invalid", q=swap_lr(orig.q, names), a=swap_lr(orig.a, names)))
    # ---- cross-episode: this episode's joints with another pilot episode's video track
    for j, (uv, okj, Tj) in other_tracks.items():
        n = min(T, Tj)
        out.append(_clone(orig, f"cross_joints{orig.joints_ep}_video{j}", "cross_episode", "invalid", q=orig.q[:n], a=orig.a[:n],
                          uv=uv[:n].copy(), ok=okj[:n].copy(), video_ep=j))
    return out
