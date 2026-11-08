"""SD10-G1 figures (descriptive; from saved results only). Palette: dataviz reference slots 1-3 (validated earlier);
every series is legended/direct-labelled (aqua <3:1 contrast).

figures/source_vs_generation_yield.png  - yields (all/near/far/regions) + paired differences vs the +10 pt gate
figures/failure_categories.png          - failed gates per condition (non-exclusive) incl. preparation failures
figures/source_distance_vs_validity.png - selection distance / SE(3) translation vs Y_A, table->giver switch residual
figures/target_acceptance_map.png       - the same 100 targets, accepted/rejected/prep-failed per condition
figures/visual_comparisons.png          - GEOMETRY-ONLY projection overlays (no RGB synthesis) on a deterministic
                                          target sample (T000,T033,T066,T099), every condition, accepted or not
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import g1lib as G  # noqa: E402

FIG = G.EXEC / "figures"
COL = {"Original10": "#2a78d6", "SD10": "#eb6834", "Random10": "#1baf7a"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True})
GATE_LABEL = {"source_preprocessing_failed": "source prep failed", "1_arrays_support": "1 arrays/support",
              "2_initial_candidate": "2 initial pose", "3_ik_residual": "3 IK residual", "4_kinematic_limits": "4 limits/speed",
              "5_clearance_penetration": "5 clearance/penetration", "6_steady_holds": "6 steady holds",
              "7_receiver_capture": "7 receiver capture", "8_object_continuity": "8 object continuity",
              "9_labels": "9 labels", "exception": "exception", "timeout_120s": "timeout"}


def cp(k, n):
    from scipy.stats import beta
    lo = beta.ppf(.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def load():
    R = json.loads((G.OUT / "results.json").read_text())
    rows = [json.loads(l) for l in open(G.OUT / "geometry_attempts.jsonl")]
    prim = {(r["target_id"], r["condition"]): r for r in rows if r["kind"] == "primary"}
    return R, prim


def fig_yield(R):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    groups = [("all100", "All 100", 100), ("near54", "Near 54", 54), ("far46", "Far 46", 46),
              ("central", "Central 16", 16), ("intermediate", "Intermed. 48", 48), ("edge", "Edge 36", 36)]
    w = .26
    for i, c in enumerate(G.CONDS):
        for j, (k, lab, n) in enumerate(groups):
            y = R["yield"][c][k]
            lo, hi = cp(round(y * n), n)
            x = j + (i - 1) * (w + .02)
            ax.bar(x, 100 * y, w, color=COL[c], label=c if j == 0 else None)
            ax.plot([x, x], [100 * lo, 100 * hi], color=INK2, lw=1)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    if all(R["yield"][c]["count"] == 0 for c in G.CONDS):
        ax.text(2.5, 14, "0 / 100 complete demonstrations in every condition\n(lines = exact 95% upper bounds)", ha="center",
                fontsize=10, color=INK)
    ax.set_ylabel("complete geometric yield Y_A (%)")
    ax.set_title("G1-A yield on the same 100 targets (bars: exact 95% CI)", loc="left", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax = axes[1]
    for j, (k, lab, cc) in enumerate([("SD10_vs_Original10", "SD10 − Original10", COL["Original10"]),
                                      ("SD10_vs_Random10", "SD10 − Random10", COL["Random10"])]):
        d = R["paired"][k]
        for o, (ci, nm) in enumerate([(d["paired_ci95"], "target CI"), (d["spatial_block_ci95"], "spatial-block CI")]):
            y = j + o * .28
            ax.plot(100 * np.array(ci), [y, y], color=cc, lw=2 if o == 0 else 1.2, ls="-" if o == 0 else "--")
            ax.plot(100 * d["difference"], y, "D", color=cc, ms=7, mec=SURF, mew=2)
            ax.text(100 * ci[1] + 1, y, nm, va="center", fontsize=7.5, color=INK2)
        ax.text(-49, j + .14, f"{lab}\nMcNemar p={d.get('exact_mcnemar_p_two_sided', float('nan')):.3g}", fontsize=8, va="center", color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(10, color=INK2, lw=1, ls=":")
    ax.text(10.5, -.45, "+10 pt gate", color=INK2, fontsize=8)
    ax.set_xlim(-50, 50)
    ax.set_ylim(1.6, -.6)
    ax.set_yticks([])
    ax.set_xlabel("paired yield difference (percentage points)")
    ax.set_title("Registered paired contrasts", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "source_vs_generation_yield.png", dpi=160)


def fig_fail(R):
    fc = R["failure_categories"]
    keys = [k for k in GATE_LABEL if any(fc[c]["any_fail_by_gate"].get(k, 0) for c in G.CONDS)]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(keys) + 1.4))
    h = .26
    for i, c in enumerate(G.CONDS):
        v = [fc[c]["any_fail_by_gate"].get(k, 0) for k in keys]
        y = np.arange(len(keys)) + (i - 1) * (h + .02)
        ax.barh(y, v, h, color=COL[c], label=c)
        for yy, vv in zip(y, v):
            ax.text(vv + .8, yy, str(vv), va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(keys)), [GATE_LABEL[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("attempts failing this gate (of 100; an attempt can fail several)")
    ax.set_title("G1-A failure categories by condition", loc="left", color=INK, pad=24)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(FIG / "failure_categories.png", dpi=160)


def fig_distance(prim):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    tids = [t["target_id"] for t in G.TGT["targets"]]
    rng = np.random.default_rng(0)
    for c in G.CONDS:
        rs = [prim[(t, c)] for t in tids]
        d = np.array([r["selection_distance_px"] for r in rs])
        y = np.array([int(bool(r.get("Y_A"))) for r in rs])
        prep = np.array([r.get("status") == "source_preprocessing_failed" for r in rs])
        jit = (rng.random(len(y)) - .5) * .12
        axes[0].scatter(d[~prep], y[~prep] + jit[~prep] + {"Original10": -.2, "SD10": 0, "Random10": .2}[c] * .3,
                        s=14, color=COL[c], alpha=.8, label=c)
        axes[0].scatter(d[prep], np.full(prep.sum(), -.35), marker="x", s=16, color=COL[c])
        tr = np.array([r.get("transform", {}).get("translation_m", np.nan) for r in rs])
        sw = np.array([((r.get("gates") or {}).get("8_object_continuity", {}).get("switches") or {}).get("table_to_giver", {}).get("step_m", np.nan) for r in rs])
        axes[1].scatter(d, 100 * tr, s=14, color=COL[c], alpha=.8, label=c)
        axes[2].scatter(100 * tr, 1000 * sw, s=14, color=COL[c], alpha=.8, label=c)
    axes[0].set_yticks([-.35, 0, 1], ["prep failed (x)", "Y_A = 0", "Y_A = 1"])
    axes[0].set_xlabel("source-to-target initial-lid distance (native px)")
    axes[0].set_title("Selection distance vs complete validity", loc="left", fontsize=9, color=INK)
    axes[1].set_xlabel("source-to-target initial-lid distance (native px)")
    axes[1].set_ylabel("SE(3) translation (cm)")
    axes[1].set_title("Pixel distance vs metric SE(3) translation\n(includes the common ≈13 cm height shift)", loc="left", fontsize=9, color=INK)
    axes[2].axhline(5, color=INK2, ls=":", lw=1)
    axes[2].text(axes[2].get_xlim()[0] if False else 0.5, 5.5, "5 mm switch gate", fontsize=7.5, color=INK2, transform=axes[2].get_yaxis_transform())
    axes[2].set_xlabel("SE(3) translation (cm)")
    axes[2].set_ylabel("table→giver switch step (mm)")
    axes[2].set_title("Retarget size vs table→giver can jump", loc="left", fontsize=9, color=INK)
    for ax in axes:
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "source_distance_vs_validity.png", dpi=160)


def fig_map(prim):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, c in zip(axes, G.CONDS):
        for t in G.TGT["targets"]:
            r = prim[(t["target_id"], c)]
            u, v = t["requested_lid_uv_native_px"]
            if r.get("status") == "source_preprocessing_failed":
                ax.scatter(u, v, marker="x", s=28, color=INK2)
            elif r.get("Y_A"):
                ax.scatter(u, v, s=42, color=COL[c], edgecolor=SURF, lw=.8)
            else:
                ax.scatter(u, v, s=42, facecolor="none", edgecolor=COL[c], lw=1)
        src = np.array([G.CFG["source_metadata"][str(e)]["initial_lid_uv"] for e in G.CFG["conditions"][c]])
        ax.scatter(src[:, 0], src[:, 1], marker="^", s=40, color=INK, label="source initial lid")
        ax.set_title(c, loc="left", fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlim(230, 480); ax.set_ylim(410, 70)
        ax.set_xlabel("u (px)")
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    axes[0].set_ylabel("v (px)")
    fig.suptitle("Same 100 targets per condition: filled = Y_A 1, open = Y_A 0, x = assigned source failed preparation; ▲ = source initial lid",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "target_acceptance_map.png", dpi=160)


def fig_visual(prim):
    """Geometry-only overlays: assigned source body frame at native row 3 with the PLANNED can outline (projected),
    the requested target lid (+) and the source can outline; no pixels are synthesized."""
    import cv2
    from sfa.recover import cylinder_points
    from sfa.robot import Camera
    cam = Camera()
    data = G.PoolData()
    P = cylinder_points([0, 0, 1.0], 24, 6) * np.array([1, 1, 1])
    sample = ["T000", "T033", "T066", "T099"]
    fig, axes = plt.subplots(len(sample), 3, figsize=(10.5, 2.6 * len(sample)))
    frames = {}
    for i, tid in enumerate(sample):
        t = next(x for x in G.TGT["targets"] if x["target_id"] == tid)
        for j, c in enumerate(G.CONDS):
            ax = axes[i, j]
            r = prim[(tid, c)]
            e = r["source_episode"]
            if e not in frames:
                F, _ = data.decode_body(e)
                frames[e] = F[3].copy()
            im = frames[e].copy()
            if r.get("arrays_file"):
                z = np.load(G.STORE / "attempts" / r["arrays_file"])
                for C, col in ((z["can"][0], (255, 140, 0)),):
                    uv, _ = cam.project((C[:3, :3] @ P.T).T + C[:3, 3])
                    cv2.polylines(im, [cv2.convexHull(uv.astype(np.float32)).astype(np.int32)], True, col, 2)
            u, v = t["requested_lid_uv_native_px"]
            cv2.drawMarker(im, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            status = "prep failed" if r.get("status") == "source_preprocessing_failed" else ("Y_A=1" if r.get("Y_A") else
                     "Y_A=0: " + ",".join(g.split("_")[0] for g in r.get("failed_gates", [])))
            ax.set_title(f"{tid} {c} src {e} | {status}", fontsize=7.5, loc="left", color=INK)
    fig.suptitle("Geometry-only overlays (NO RGB synthesis): source frame, planned can at export row 0 (orange), requested lid (+)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "visual_comparisons.png", dpi=130)


def main():
    FIG.mkdir(exist_ok=True)
    R, prim = load()
    fig_yield(R)
    fig_fail(R)
    fig_distance(prim)
    fig_map(prim)
    fig_visual(prim)
    print("figures written")


if __name__ == "__main__":
    main()
