"""Figures and contact sheets for SD10.

figures/source_diversity_comparison.png    coverage maps (original / SD10 / random), coverage metrics vs random-ten null,
                                           interaction features
figures/contact_sheet_full50.jpg           all 50 episodes at frame 0 with the measured lid centre and status
figures/contact_sheet_original_ten.jpg     original ten: frame 0, rc+20, lc, ro+10 (body camera)
figures/contact_sheet_sd10.jpg             SD10 selection, same layout
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sd10 import common as C  # noqa: E402

# dataviz reference palette, first three categorical slots (validated all-pairs, light); aqua < 3:1 -> direct labels
COL = {"original_ten": "#2a78d6", "sd10_selected": "#eb6834", "random_ten_control": "#1baf7a"}
NAME = {"original_ten": "Original ten (pool 0–9)", "sd10_selected": "SD10 selection", "random_ten_control": "Random ten (seed 20261107)"}
MARK = {"original_ten": "o", "sd10_selected": "D", "random_ten_control": "^"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
KEY = C.OUT / "keyframes"


def rows():
    R = list(csv.DictReader(open(HERE / "full50_episode_features.csv")))
    for r in R:
        r["episode"] = int(r["episode"])
    return R


def background():
    ds = C.pool()
    F = np.stack([ds.decode(C.BODY, e, [0])[0] for e in range(50)])
    return np.median(F, 0).astype(np.uint8)


def label(im, txt, scale=0.5, org=(5, 18)):
    cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def comparison(R, det, bg):
    P = np.array([[float(r["lid_u"]), float(r["lid_v"])] for r in R])
    locked = list(C.V4A_LOCKED)
    sets = det["sets"]
    M = det["metrics"]
    null = det["null_random_ten"]["pool50"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig = plt.figure(figsize=(15, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 0.8, 0.95])
    for j, k in enumerate(["original_ten", "sd10_selected", "random_ten_control"]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(bg, alpha=0.45)
        ax.scatter(P[:, 0], P[:, 1], s=14, c="#8a8984", lw=0, label="pool can position (50)", zorder=2)
        ax.scatter(P[locked, 0], P[locked, 1], s=60, facecolors="none", edgecolors=INK, marker="s", lw=1.2,
                   label="v4A locked (ineligible)", zorder=3)
        S = sets[k]
        for e in S:
            ax.add_patch(plt.Circle(P[e], 40, fill=False, ec=COL[k], lw=1, alpha=0.8, zorder=3))
        ax.scatter(P[S, 0], P[S, 1], s=48, c=COL[k], marker=MARK[k], edgecolors=SURF, lw=1.2, zorder=4, label=NAME[k])
        for e in S:
            ax.annotate(str(e), P[e], xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc=SURF, ec="none", alpha=0.8))
        m = M[k]["pool50"]
        ax.set_title(f"{NAME[k]}\nradius {m['covering_radius_px']:.0f} px · mean nearest {m['mean_nearest_px']:.0f} px · "
                     f"≤40 px {m['frac_within_40px']:.0%}", fontsize=9)
        ax.set_xlim(150, 560); ax.set_ylim(470, 0)
        ax.set_xlabel("body-camera u (px)"); ax.set_ylabel("v (px)")
        ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(0, 1, "Can position = frame-0 lid-ellipse centre\n(SAM2.1, body camera; image proxy,\nno metric 3D calibration).\n\n"
                  "Circles: 40 px around each source\n(the v4B 'far' radius).\n\n"
                  "Coverage targets: all 50 pool positions.\n"
                  "Grey band below: 5–95 % of 10,000\nrandom tens from the 44 eligible\n(seed 20261108); tick = median.\n\n"
                  "Selection used position (u, v) plus\nleft-receive TCP xyz and grasp→receive\ntime (weight 0.25); no VLA scores.",
            va="top", fontsize=9, color=INK2)
    specs = [("covering_radius_px", "Covering radius (px) ↓", 1), ("mean_nearest_px", "Mean nearest source (px) ↓", 1),
             ("frac_within_40px", "Pool within 40 px of a source ↑", 100), ("hull_area_px2", "Source hull area (k px²) ↑", 1e-3)]
    for j, (key, title, sc) in enumerate(specs):
        ax = fig.add_subplot(gs[1, j])
        nb = null[key]
        ax.axvspan(nb["p05"] * sc, nb["p95"] * sc, color=GRID, zorder=0)
        ax.axvline(nb["median"] * sc, color="#8a8984", lw=1, zorder=1)
        for i, k in enumerate(["original_ten", "sd10_selected", "random_ten_control"]):
            x = M[k]["pool50"][key] * sc
            ax.scatter([x], [i], s=60, c=COL[k], marker=MARK[k], zorder=3, edgecolors=SURF, lw=1)
            ax.annotate(f"{x:.0f}{'%' if sc == 100 else ''}", (x, i), xytext=(7, -3), textcoords="offset points", fontsize=8, color=INK)
        ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["Original", "SD10", "Random"] if j == 0 else [])
        ax.set_ylim(2.6, -0.6); ax.set_title(title, fontsize=9.5)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # interaction panels
    def col(k):
        return np.array([float(r[k]) for r in R])
    lx, ly, dt = col("left_receive_x_m"), col("left_receive_y_m"), col("handover_dt_s")
    for j, (xa, ya, xl, yl) in enumerate([(lx * 100, ly * 100, "left receive TCP x (cm)", "left receive TCP y (cm)"),
                                           (dt, col("left_receive_z_m") * 100, "right grasp → left receive (s)", "left receive TCP z (cm)")]):
        ax = fig.add_subplot(gs[2, j * 2:(j + 1) * 2] if False else gs[2, j])
        ax.scatter(xa, ya, s=12, c="#8a8984", lw=0, label="pool (50)")
        for k in ["original_ten", "sd10_selected", "random_ten_control"]:
            S = sets[k]
            ax.scatter(xa[S], ya[S], s=40, facecolors="none" if k != "sd10_selected" else COL[k], edgecolors=COL[k],
                       marker=MARK[k], lw=1.3, label=NAME[k], alpha=0.95)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if j == 0:
            ax.legend(fontsize=7, loc="best", framealpha=0.85)
        ax.set_title("Interaction diversity (FK of measured state)" if j == 0 else "Handover timing vs receive height", fontsize=9.5)
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")
    tab = [["", "Original", "SD10", "Random"]]
    def fmt(k, f):
        return [f(M[s]) for s in ("original_ten", "sd10_selected", "random_ten_control")]
    tab.append(["Episodes"] + [", ".join(map(str, sets[s][:5])) + ",\n" + ", ".join(map(str, sets[s][5:])) for s in ("original_ten", "sd10_selected", "random_ten_control")])
    tab.append(["Covering radius, v4A-locked 6 (px)"] + fmt("", lambda m: f"{m['v4a_locked6']['covering_radius_px']:.0f}"))
    tab.append(["Locked 6 within 40 px"] + fmt("", lambda m: f"{m['v4a_locked6']['frac_within_40px'] * 6:.0f}/6"))
    tab.append(["Pool within 20 px"] + fmt("", lambda m: f"{m['pool50']['frac_within_20px']:.0%}"))
    tab.append(["Min source spacing (px)"] + fmt("", lambda m: f"{m['pool50']['min_pairwise_spacing_px']:.0f}"))
    tab.append(["Grasp→receive range (s)"] + fmt("", lambda m: f"{m['interaction']['handover_dt_s']['min']:.2f}–{m['interaction']['handover_dt_s']['max']:.2f}"))
    tab.append(["Episode length range"] + fmt("", lambda m: f"{m['interaction']['length']['min']:.0f}–{m['interaction']['length']['max']:.0f}"))
    tab.append(["Mean quality q"] + fmt("", lambda m: f"{m['mean_quality_q']:.2f}"))
    t = ax.table(cellText=tab, loc="center", cellLoc="left", colWidths=[0.34, 0.22, 0.22, 0.22])
    t.auto_set_font_size(False); t.set_fontsize(8)
    for (r_, c_), cell in t.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_facecolor(SURF)
        if r_ == 0:
            cell.set_text_props(color=INK, weight="bold")
    t.scale(1, 1.35)
    for c_ in range(4):
        t[1, c_].set_height(t[1, c_].get_height() * 1.9)
    fig.suptitle("SD10 source-diversity curation — pool real_v00 @2c1bc860 (development data; open-loop coverage only, no policy result)",
                 fontsize=11, color=INK)
    C.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG / "source_diversity_comparison.png", dpi=130, facecolor=SURF)


def full50(R, det):
    ds = C.pool()
    sel, orig = set(det["sets"]["sd10_selected"]), set(det["sets"]["original_ten"])
    tiles = []
    for r in R:
        e = r["episode"]
        im = ds.decode(C.BODY, e, [0])[0].copy()
        u, v = float(r["lid_u"]), float(r["lid_v"])
        cv2.ellipse(im, ((u, v), (float(r["lid_major_px"]), float(r["lid_minor_px"])), 0), (0, 255, 0), 1)
        cv2.drawMarker(im, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        im = cv2.resize(im, (320, 240))
        tag = "LOCKED v4A" if e in C.V4A_LOCKED else ("SD10" if e in sel else "eligible")
        if e in orig:
            tag += " +orig10"
        label(im, f"ep{e} ({u:.0f},{v:.0f}) T={r['length']}", 0.5)
        label(im, tag, 0.5, (5, 232))
        border = (235, 104, 52) if e in sel else ((42, 120, 214) if e in orig else ((20, 20, 20) if e in C.V4A_LOCKED else (200, 200, 200)))
        im = cv2.copyMakeBorder(im, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border)
        tiles.append(im)
    rows_ = [np.hstack(tiles[i:i + 10]) for i in range(0, 50, 10)]
    out = np.vstack(rows_)
    cv2.imwrite(str(C.FIG / "contact_sheet_full50.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def set_sheet(R, eps, name, color):
    rr = {r["episode"]: r for r in R}
    rows_ = []
    for e in eps:
        z = np.load(KEY / f"keys_{e:02d}.npz")
        idx = list(z["idx"])
        tiles = []
        for k, nm in ((0, "start"), (2, "rc+20"), (3, "lc"), (4, "ro+10")):
            im = cv2.resize(z["body"][k], (320, 240)).copy()
            if k == 0:
                u, v = float(rr[e]["lid_u"]) / 2, float(rr[e]["lid_v"]) / 2
                cv2.drawMarker(im, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 12, 2)
            label(im, f"ep{e} {nm} t={idx[k]}")
            tiles.append(im)
        if "right" in z:
            for arr, k, nm in ((z["right"], 2, "R-wrist rc+20"), (z["left"], 5, "L-wrist last")):
                tiles.append(label(cv2.resize(arr[k], (320, 240)).copy(), nm))
        r = rr[e]
        info = np.full((240, 320, 3), 252, np.uint8)
        lines = [f"episode {e}", f"lid (u,v) = ({float(r['lid_u']):.0f}, {float(r['lid_v']):.0f}) px",
                 f"left receive xyz = ({float(r['left_receive_x_m']):.3f}, {float(r['left_receive_y_m']):.3f}, {float(r['left_receive_z_m']):.3f}) m",
                 f"grasp->receive {float(r['handover_dt_s']):.2f} s, T={r['length']}",
                 f"right grip open {float(r['meas_right_grasp_open']):.2f}, left hold {float(r['meas_left_hold_open']):.2f}",
                 f"q = {float(r['quality_score_q']):.2f}" if r["quality_score_q"] not in ("", "None") else "q = n/a",
                 "gates Q1-Q6: PASS" if r["passes_quality"] == "True" else "gates: FAIL"]
        for i, t in enumerate(lines):
            cv2.putText(info, t, (8, 26 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(info)
        rows_.append(np.hstack(tiles))
    out = np.vstack(rows_)
    head = np.full((40, out.shape[1], 3), 252, np.uint8)
    cv2.putText(head, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    out = np.vstack([head, out])
    small = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return small


def main():
    R = rows()
    det = json.loads((C.OUT / "selection_details.json").read_text())
    comparison(R, det, background())
    full50(R, det)
    o = set_sheet(R, det["sets"]["original_ten"], "Original ten = pool episodes 0-9 (real_10_v00 @cebdee98)", (42, 120, 214))
    s = set_sheet(R, det["sets"]["sd10_selected"], "SD10 selection (protocol + A1; k-center + p-median)", (235, 104, 52))
    cv2.imwrite(str(C.FIG / "contact_sheet_original_ten.jpg"), cv2.cvtColor(o, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(str(C.FIG / "contact_sheet_sd10.jpg"), cv2.cvtColor(s, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(sorted(p.name for p in C.FIG.iterdir()))


if __name__ == "__main__":
    main()
