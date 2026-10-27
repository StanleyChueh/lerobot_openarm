"""Phase 1: per-episode quality of the 500 generated episodes vs the 10 manual source episodes.

Ground truth comes from the generation HDF5 (can pose every step, both EEF positions, success attr), which is
proven 1:1 with the published dataset (provenance_check.py). Gripper events come from the dataset's own
action/state (identical units for both sets).
"""
import collections
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ARM, DATA, GRIP, load, load_hdf5, openness, savefig

REST_Z = 0.380        # can origin standing on the pad (measured: min z of every episode)
LIFT = 0.02           # m above rest = lifted
HOLD_DIST = 0.11      # EEF_TO_CAN_THRESHOLD_HANDOVER in openarm_task_modes.py
rng = np.random.default_rng(0)

h = load_hdf5()
SRC, ANN, GEN = h["source"]["episodes"], h["annotated"]["episodes"], h["generated"]["episodes"]
DS = {"manual": load("manual"), "generated": load("generated")}


# ---------------------------------------------------------------- source attribution (left arm replayed untransformed)
def shift_match(g, s, max_shift=70):
    best = (np.inf, 0)
    for sh in range(max_shift):
        n = min(len(s), len(g) - sh)
        if n < 100:
            continue
        m = np.linalg.norm(g[sh:sh + n] - s[:n], axis=1)[n // 2:].mean()
        if m < best[0]:
            best = (m, sh)
    return best


attr = []
for g in GEN:
    ds = [shift_match(g["obs/eef_pos"], a["obs/eef_pos"]) for a in ANN]
    o = np.argsort([d[0] for d in ds])
    attr.append({"source": int(o[0]), "match_m": ds[o[0]][0], "second_m": ds[o[1]][0], "shift": ds[o[0]][1]})


# ---------------------------------------------------------------- per-episode events
def first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return int(idx[0] + start) if len(idx) else None


def tilt_deg(q):  # (T,4) w,x,y,z -> angle between the can axis and world z
    w, x, y, z = q.T
    cz = 1 - 2 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(cz, -1, 1)))


def events(ep_h, S, A, ann=None):
    can = ep_h["states/rigid_object/can/root_pose"]
    cv = ep_h["states/rigid_object/can/root_velocity"]
    init = ep_h["initial_state/rigid_object/can/root_pose"][0]
    T = len(A)
    # post-step can pose of step t-1 = pre-step pose of step t
    can_pre = np.r_[init[None, :7], can[:-1]]
    pR, pL = ep_h["obs/right_eef_pos"], ep_h["obs/eef_pos"]
    dR, dL = np.linalg.norm(pR - can_pre[:, :3], axis=1), np.linalg.norm(pL - can_pre[:, :3], axis=1)
    cmdR, cmdL = openness(A[:, GRIP["R"]], "R") < 0.5, openness(A[:, GRIP["L"]], "L") < 0.5
    mR, mL = openness(S[:, GRIP["R"]], "R"), openness(S[:, GRIP["L"]], "L")
    holdR = cmdR & (mR > 0.25) & (mR < 0.97) & (dR < HOLD_DIST)
    holdL = cmdL & (mL > 0.25) & (mL < 0.97) & (dL < HOLD_DIST)
    z = can_pre[:, 2]
    lifted = z > REST_Z + LIFT
    tRc = first(cmdR); tLc = first(cmdL)
    tRr = first(~cmdR, tRc) if tRc is not None else None
    tLr = first(~cmdL, tLc) if tLc is not None else None
    tlift = first(lifted)
    # drop: once lifted, the can is back on the pad (or below it) while neither hand holds it
    unsupported = (~holdR) & (~holdL)
    back_down = (z < REST_Z + 0.008) & unsupported
    t_drop = first(back_down, tlift) if tlift is not None else None
    off_table = bool((can[:, 2] < REST_Z - 0.03).any())
    e = {
        "n_frames": T, "x0": float(init[0]), "y0": float(init[1]),
        "t_R_close": tRc, "t_R_release": tRr, "t_L_close": tLc, "t_L_release": tLr, "t_lift": tlift,
        "overlap_frames": (tRr - tLc) if (tRr is not None and tLc is not None) else None,
        "push_before_grasp_m": float(np.linalg.norm(can_pre[tRc, :2] - init[:2])) if tRc is not None else None,
        "grasp_dx": float(pR[tRc, 0] - can_pre[tRc, 0]) if tRc is not None else None,
        "grasp_dy": float(pR[tRc, 1] - can_pre[tRc, 1]) if tRc is not None else None,
        "grasp_dz": float(pR[tRc, 2] - can_pre[tRc, 2]) if tRc is not None else None,
        "grasp_eef_dist": float(dR[tRc]) if tRc is not None else None,
        "max_lift_m": float(z.max() - REST_Z),
        "dropped": t_drop is not None, "t_drop": t_drop, "off_table": off_table,
        "can_speed_max": float(np.linalg.norm(cv[:, :3], axis=1).max()),
        "tilt_max_deg": float(tilt_deg(can[:, 3:7]).max()),
        "end_held_left": bool(holdL[-1] and z[-1] > REST_Z + 0.005),  # HANDOVER_PUT_DOWN_MARGIN = 5 mm
        "end_held_low": bool(holdL[-1] and z[-1] < REST_Z + LIFT), "end_left_holding_frames": int(holdL[-30:].sum()),
        "end_can_z": float(can[-1, 2]),
        "R_hold_open_med": float(np.median(mR[holdR])) if holdR.any() else None,
        "L_hold_open_med": float(np.median(mL[holdL])) if holdL.any() else None,
        "gap_max": float(np.abs(A[:, ARM] - S[:, ARM]).max()),
        "jump_max": float(np.abs(np.diff(A[:, ARM], axis=0)).max()),
        "jerk_p95": float(np.percentile(np.abs(np.diff(S[:, ARM], n=3, axis=0) * 30 ** 3).max(1), 95)),
        "speed_max": float(np.abs(np.diff(S[:, ARM], axis=0) * 30).max()),
        "elbow_min_target": float(min(A[:, 6].min(), A[:, 7].min())),  # RJ4, LJ4 columns
        "end_home_dist": float(np.abs(S[-1, ARM] - S[0, ARM]).max()),
        "hdf5_success": bool(ep_h["attrs"].get("success", False)),
    }
    if ann is not None:
        e |= {f"sig_{k}": first(v > 0.5) for k, v in ann["signals"].items()}
    return e


rows = []
# manual: can pose and initial state from the ORIGINAL recording (== the published dataset); EEF positions exist
# only in the annotated copy, a re-simulation that drifts <= 8 mm / 0.04 rad from it (measured).
MAN_H = [{**s, "obs/eef_pos": a["obs/eef_pos"], "obs/right_eef_pos": a["obs/right_eef_pos"]} for s, a in zip(SRC, ANN)]
for name, eps_h in [("manual", MAN_H), ("generated", GEN)]:
    d = DS[name]
    for i, (a, b) in enumerate(d["episodes"]):
        r = events(eps_h[i], d["state"][a:b], d["action"][a:b], ANN[i] if name == "manual" else None)
        r |= {"set": name, "episode": i}
        if name == "generated":
            r |= attr[i]
        rows.append(r)
ev = pd.DataFrame(rows)
ev.to_csv(DATA / "episode_events.csv", index=False)
man, gen = ev[ev.set == "manual"], ev[ev.set == "generated"]

# validate the event detector on the manual demos' annotated signals (annotate_demos.py --auto)
val = man[["t_R_close", "sig_grasp_right", "t_L_close", "sig_handover"]]
print("detector vs annotation (manual):\n", val.to_string())

# ---------------------------------------------------------------- flags
man_rng = {k: (man[k].min(), man[k].max()) for k in ["n_frames", "overlap_frames", "push_before_grasp_m", "jerk_p95", "tilt_max_deg", "can_speed_max"]}
flags = {}
for _, r in ev.iterrows():
    proven, susp = [], []
    if not r.end_held_left:
        proven.append("not_held_by_left_at_end")
    if r.dropped:
        proven.append("can_back_on_pad_unheld_after_lift")
    if r.off_table:
        proven.append("can_fell_off_pad")
    if r.gap_max > 0.5:
        proven.append("command_gap>0.5rad(eval safety hold)")
    if r.jump_max > 0.25:
        proven.append("command_jump>0.25rad(eval safety hold)")
    if r.t_L_close is None or r.t_R_release is None:
        proven.append("no_handover_events")
    if r.overlap_frames is not None and r.overlap_frames < 1:
        susp.append("right_released_before_left_closed")
    if r.jerk_p95 > man_rng["jerk_p95"][1]:
        susp.append("jerk_above_manual_max")
    if r.push_before_grasp_m is not None and r.push_before_grasp_m > max(0.03, man_rng["push_before_grasp_m"][1]):
        susp.append("can_pushed_before_grasp")
    if r.tilt_max_deg > max(20, man_rng["tilt_max_deg"][1]):
        susp.append("can_tilted")
    if r.can_speed_max > max(1.0, man_rng["can_speed_max"][1]):
        susp.append("can_velocity_spike")
    if r.end_held_low and r.end_held_left:
        susp.append("held_by_left_but_within_2cm_of_pad_at_end")
    if r.overlap_frames is not None and r.overlap_frames > man_rng["overlap_frames"][1]:
        susp.append("handover_overlap_above_manual_max")
    if r.n_frames < 0.8 * man_rng["n_frames"][0] or r.n_frames > 1.25 * man_rng["n_frames"][1]:
        susp.append("duration_outside_manual")
    flags[(r.set, r.episode)] = (proven, susp)
ev["proven_invalid"] = [";".join(flags[(s, e)][0]) for s, e in zip(ev.set, ev.episode)]
ev["suspicious"] = [";".join(flags[(s, e)][1]) for s, e in zip(ev.set, ev.episode)]
ev.to_csv(DATA / "episode_events.csv", index=False)
man, gen = ev[ev.set == "manual"], ev[ev.set == "generated"]

# ---------------------------------------------------------------- spawn distributions
X_EVAL, Y_EVAL, G_EVAL = (0.12, 0.40), (-0.08, 0.08), ((0.235, 0.07), (0.01, 0.04))
KEEP = [(0.19, 0.42, 0.085, 0.215), (0.19, 0.42, -0.215, -0.085), (0.265, 0.377, 0.029, 0.278), (0.265, 0.377, -0.278, -0.029),
        (0.265, 0.440, 0.060, 0.278), (0.265, 0.440, -0.278, -0.060)]
X_GEN, Y_GEN = (0.2029, 0.4217), (-0.0769, 0.0769)


def eval_samples(n):
    out = []
    while len(out) < n:
        x = rng.normal(*G_EVAL[0], 4 * n); y = rng.normal(*G_EVAL[1], 4 * n)
        ok = (x >= X_EVAL[0]) & (x <= X_EVAL[1]) & (y >= Y_EVAL[0]) & (y <= Y_EVAL[1])
        for x0, x1, y0, y1 in KEEP:
            ok &= ~((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        out.extend(zip(x[ok], y[ok]))
    return np.array(out[:n])


E = eval_samples(200_000)
P_man = man[["x0", "y0"]].to_numpy(); P_gen = gen[["x0", "y0"]].to_numpy()
in_keep = np.zeros(len(P_gen), bool)
for x0, x1, y0, y1 in KEEP:
    in_keep |= (P_gen[:, 0] >= x0) & (P_gen[:, 0] <= x1) & (P_gen[:, 1] >= y0) & (P_gen[:, 1] <= y1)


def nn(P, Q):
    return np.sqrt(((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(1)


def cover(P, r):
    return float(np.mean(np.concatenate([nn(E[i:i + 20000], P) for i in range(0, len(E), 20000)]) <= r))


cov = {f"r={r}": {"manual": cover(P_man, r), "generated": cover(P_gen, r), "manual+generated": cover(np.r_[P_man, P_gen], r)}
       for r in (0.02, 0.03, 0.04)}
pos = {
    "eval_frac_x_below_gen_min": float((E[:, 0] < X_GEN[0]).mean()),
    "eval_frac_outside_gen_box": float(((E[:, 0] < X_GEN[0]) | (E[:, 0] > X_GEN[1]) | (np.abs(E[:, 1]) > Y_GEN[1])).mean()),
    "gen_frac_in_eval_keepout": float(in_keep.mean()),
    "gen_x_range": [float(P_gen[:, 0].min()), float(P_gen[:, 0].max())], "gen_y_range": [float(P_gen[:, 1].min()), float(P_gen[:, 1].max())],
    "manual_x_range": [float(P_man[:, 0].min()), float(P_man[:, 0].max())], "manual_y_range": [float(P_man[:, 1].min()), float(P_man[:, 1].max())],
    "eval_coverage_within_r": cov,
}
print("positions:", json.dumps(pos, indent=1))

# ---------------------------------------------------------------- diversity / near-duplicates
counts = collections.Counter(gen.source.astype(int))
p = np.array(list(counts.values())) / len(gen)
div = {"source_counts": {int(k): int(v) for k, v in sorted(counts.items())},
       "sources_never_used": [s for s in range(10) if s not in counts],
       "effective_sources_exp_entropy": float(np.exp(-(p * np.log(p)).sum())),
       "attribution_match_m": [float(gen.match_m.median()), float(gen.match_m.max())],
       "attribution_second_over_best_min": float((gen.second_m / gen.match_m).min())}


def resample(S, n=60):
    t = np.linspace(0, len(S) - 1, n)
    return np.stack([np.interp(t, np.arange(len(S)), S[:, j]) for j in range(S.shape[1])], 1)


traj = {name: np.stack([resample(DS[name]["state"][a:b][:, ARM]).ravel() for a, b in DS[name]["episodes"]]) for name in DS}
scale = np.sqrt(len(ARM) * 60)
Dg = np.sqrt(((traj["generated"][:, None] - traj["generated"][None]) ** 2).sum(-1)) / scale
np.fill_diagonal(Dg, np.inf)
Dm = np.sqrt(((traj["manual"][:, None] - traj["manual"][None]) ** 2).sum(-1)) / scale
np.fill_diagonal(Dm, np.inf)
Dgm = np.sqrt(((traj["generated"][:, None] - traj["manual"][None]) ** 2).sum(-1)) / scale
gen_nn, man_nn = Dg.min(1), Dm.min(1)
same_src = gen.source.to_numpy(int)[:, None] == gen.source.to_numpy(int)[None]
dup_thr = float(np.percentile(man_nn, 0) * 0.25)  # a quarter of the closest pair of distinct human demos
div |= {"manual_nn_rms_rad": [float(man_nn.min()), float(np.median(man_nn))],
        "generated_nn_rms_rad": [float(gen_nn.min()), float(np.median(gen_nn)), float(np.percentile(gen_nn, 95))],
        "generated_to_own_source_rms_rad_median": float(np.median([Dgm[i, s] for i, s in enumerate(gen.source.astype(int))])),
        "near_duplicate_threshold_rms_rad": dup_thr,
        "generated_with_near_duplicate": int((gen_nn < dup_thr).sum()),
        "nn_is_same_source_frac": float(np.mean([same_src[i, np.argmin(Dg[i])] for i in range(len(gen))]))}
# kernel effective n over start positions (4 cm Gaussian kernel)
def keff(P, s=0.04):
    K = np.exp(-((P[:, None] - P[None]) ** 2).sum(-1) / (2 * s * s))
    return float(len(P) ** 2 / K.sum())
div |= {"kernel_eff_n_4cm": {"manual": keff(P_man), "generated": keff(P_gen)}}
print("diversity:", json.dumps(div, indent=1))

summary = {
    "counts": {"manual": len(man), "generated": len(gen)},
    "hdf5_success_all": bool(gen.hdf5_success.all()),
    "proven_invalid": {name: {"episodes": int((g.proven_invalid != "").sum()),
                              "by_flag": dict(collections.Counter(f for s in g.proven_invalid if s for f in s.split(";")))}
                       for name, g in [("manual", man), ("generated", gen)]},
    "suspicious": {name: {"episodes": int((g.suspicious != "").sum()),
                          "by_flag": dict(collections.Counter(f for s in g.suspicious if s for f in s.split(";")))}
                   for name, g in [("manual", man), ("generated", gen)]},
    "manual_ranges": {k: [float(v[0]), float(v[1])] for k, v in man_rng.items()},
    "distributions": {k: {"manual": [float(man[k].quantile(q)) for q in (0, .5, 1)],
                          "generated": [float(gen[k].quantile(q)) for q in (0, .05, .5, .95, 1)]}
                      for k in ["n_frames", "t_R_close", "t_lift", "t_L_close", "t_R_release", "overlap_frames", "push_before_grasp_m",
                                "grasp_dx", "grasp_dy", "grasp_dz", "max_lift_m", "tilt_max_deg", "can_speed_max", "jerk_p95",
                                "speed_max", "elbow_min_target", "R_hold_open_med", "L_hold_open_med", "end_home_dist"]},
    "positions": pos, "diversity": div,
    "invalid_episode_ids": gen.loc[gen.proven_invalid != "", ["episode", "source", "proven_invalid"]].to_dict("records"),
}
json.dump(summary, open(DATA / "generated_quality.json", "w"), indent=1, default=lambda o: None if o is None else (o.item() if hasattr(o, "item") else str(o)))
print(json.dumps({k: summary[k] for k in ("hdf5_success_all", "proven_invalid", "suspicious", "invalid_episode_ids")}, indent=1, default=str))
for k, v in summary["distributions"].items():
    print(f"  {k:22s} manual {np.round(v['manual'], 3)}  generated {np.round(v['generated'], 3)}")

# ---------------------------------------------------------------- figures
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6b6b"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
H, xe, ye = np.histogram2d(E[:, 0], E[:, 1], bins=[np.linspace(0.10, 0.44, 35), np.linspace(-0.10, 0.10, 21)], density=True)
for a in ax:
    a.pcolormesh(xe, ye, H.T, cmap="Greys", alpha=0.55, shading="auto")
    a.add_patch(plt.Rectangle((X_GEN[0], Y_GEN[0]), X_GEN[1] - X_GEN[0], Y_GEN[1] - Y_GEN[0], fill=False, ec=ORANGE, lw=1.2, ls="--"))
    for x0, x1, y0, y1 in KEEP[2:]:
        a.add_patch(plt.Rectangle((x0, max(y0, -0.1)), x1 - x0, min(y1, 0.1) - max(y0, -0.1), fill=False, ec="k", lw=0.6, hatch="//", alpha=0.4))
    a.set_aspect("equal"); a.set_xlabel("x (m, env frame)")
ax[0].scatter(P_man[:, 0], P_man[:, 1], s=40, c=BLUE, edgecolor="white", lw=0.8, zorder=3)
ax[0].set_title("manual sim_10 starts (n=10)\nshade: evaluation spawn density", fontsize=10)
ax[1].scatter(P_gen[:, 0], P_gen[:, 1], s=9, c=ORANGE, alpha=0.7, zorder=3)
ax[1].set_title("generated_500 starts (n=500)\ndashed: generation range; hatched: eval keep-out", fontsize=10)
H2, _, _ = np.histogram2d(P_gen[:, 0], P_gen[:, 1], bins=[xe, ye])
Hn = H / H.sum() - H2 / H2.sum()
m = np.abs(Hn).max()
im = ax[2].pcolormesh(xe, ye, Hn.T, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
ax[2].set_title("eval − generated share per cell\n(red: under-covered by generation)", fontsize=10)
ax[0].set_ylabel("y (m)")
fig.colorbar(im, ax=ax[2], shrink=0.8)
savefig(fig, "position_heatmaps.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for col, lab in [("t_R_close", "right close"), ("t_lift", "lift"), ("t_L_close", "left close"), ("t_R_release", "right release"), ("n_frames", "episode end")]:
    ax[0].scatter(gen[col], np.full(len(gen), lab), s=6, alpha=0.25, c=ORANGE)
    ax[0].scatter(man[col], np.full(len(man), lab), s=40, marker="|", c=BLUE, lw=2)
ax[0].set_xlabel("frame (30 Hz)"); ax[0].set_title("Event timing: manual (blue ticks) vs generated (orange)")
ax[1].hist(gen.overlap_frames.dropna(), bins=np.arange(-5, 40), color=ORANGE, alpha=0.7, label="generated", density=True)
ax[1].hist(man.overlap_frames.dropna(), bins=np.arange(-5, 40), color=BLUE, alpha=0.7, label="manual", density=True)
ax[1].set_xlabel("both-hands-closed overlap at hand-over (frames)"); ax[1].legend(); ax[1].set_title("Hand-over overlap")
savefig(fig, "event_timing.png")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
cnt = [counts.get(s, 0) for s in range(10)]
ax[0].bar(range(10), cnt, color=ORANGE); ax[0].set_xticks(range(10)); ax[0].set_xlabel("source demo"); ax[0].set_ylabel("generated episodes")
ax[0].set_title(f"Source usage (effective sources {div['effective_sources_exp_entropy']:.1f} of 10)")
ax[1].hist(gen_nn, bins=40, color=ORANGE, alpha=0.8, label="generated → nearest generated")
for v in man_nn:
    ax[1].axvline(v, color=BLUE, lw=1)
ax[1].set_xlabel("RMS joint distance to nearest other episode (rad)"); ax[1].set_title("Near-duplicates (blue = manual pairs)"); ax[1].legend(fontsize=8)
fl = collections.Counter(f for s in gen.proven_invalid if s for f in s.split(";")) + collections.Counter(f for s in gen.suspicious if s for f in s.split(";"))
names_ = list(fl); ax[2].barh(names_, [fl[n] for n in names_], color=[("#c0392b" if any(n in s for s in gen.proven_invalid) else GREY) for n in names_])
ax[2].set_xlabel("generated episodes flagged"); ax[2].set_title("Flags (red = proven invalid, grey = heuristic)")
savefig(fig, "generated_quality_duplicates.png")
