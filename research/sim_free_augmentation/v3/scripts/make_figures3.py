"""v3 tables and figures (reads outputs/ep*/results.json, init_sweep.json, dataset_export/*, bench/*, train_screen_*).

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/make_figures3.py
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from sfa3.recon3 import rb  # noqa: E402

COL = {"source": "#52514e", "G+": "#2a78d6", "C": "#1baf7a", "heldout": "#eb6834"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
SURF, GRID, INK, INK2 = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2.0,
                     "legend.frameon": False, "axes.titlesize": 10, "axes.titlecolor": INK})
FIG, TAB = HERE / "outputs/figures", HERE / "outputs/tables"
CFG = json.loads((HERE / "configs/v3.json").read_text())
EPS = CFG["source_episodes"]
CONDS = ("G+", "C")
FAMS = list(CFG["families"])


def R(E):
    return json.loads((HERE / f"outputs/ep{E}/results.json").read_text())


def cand(E, key):
    return np.load(HERE / f"outputs/ep{E}/candidates/{key}.npz")


# ----------------------------------------------------------------------------- counts
def counts(res, exp):
    rows = []
    erow = {(r["episode"], r["variation"]): r for r in exp["rows"]} if exp else {}
    for E, r in res.items():
        for vid, p in r["plans"].items():
            if vid == "identity":
                continue
            fam = p["family"]
            n_prop = sum(1 for k in r["candidates"] if k.startswith(vid + "__"))
            for c in CONDS:
                s = p["own_scale"][c]
                e = erow.get((E, vid), {})
                rows.append({"ep": E, "var": vid, "fam": fam, "cond": c, "proposed_tries": n_prop,
                             "accepted": s is not None, "full": s == 1.0, "scale": s, "common": p["common_scale"],
                             "export_ok": e.get(f"{c}_export_ok"), "reasons": e.get(f"{c}_reasons"),
                             "matched_ok": e.get("matched_ok"),
                             "rejected_checks": sorted({x for k, cc in r["candidates"].items() if k.startswith(f"{vid}__{c}__") for x in cc["verify_failed"] + ([] if cc["labels_ok"] else ["labels"]) + ([] if cc["targets_ok"] else ["targets"]) + ([] if cc["ik"]["ok"] else ["ik"])})})
    return rows


def count_tables(rows, exp):
    out = ["| family | condition | variations x episodes | accepted (own scale) | at full scale | export checks pass | matched pairs (both conditions, common scale, both pass) |",
           "|---|---|---|---|---|---|---|"]
    for f in FAMS:
        for c in CONDS:
            rr = [r for r in rows if r["fam"] == f and r["cond"] == c]
            out.append(f"| {f} | {c} | {len(rr)} | {sum(r['accepted'] for r in rr)} | {sum(r['full'] for r in rr)} | "
                       f"{sum(bool(r['export_ok']) for r in rr)} | {sum(bool(r['matched_ok']) for r in rr) if c == 'C' else ''} |")
    rr = [r for r in rows if r["cond"] == "C"]
    out.append(f"| all | both | {len(rr)} per condition | G+ {sum(r['accepted'] for r in rows if r['cond'] == 'G+')}, C {sum(r['accepted'] for r in rr)} | "
               f"G+ {sum(r['full'] for r in rows if r['cond'] == 'G+')}, C {sum(r['full'] for r in rr)} | "
               f"G+ {sum(bool(r['export_ok']) for r in rows if r['cond'] == 'G+')}, C {sum(bool(r['export_ok']) for r in rr)} | {sum(bool(r['matched_ok']) for r in rr)} |")
    # rejection reasons
    rej = collections.Counter()
    for r in rows:
        if not r["accepted"]:
            for x in r["rejected_checks"]:
                rej[(r["cond"], x)] += 1
    out += ["", "| condition | check failing on a never-accepted case | count |", "|---|---|---|"]
    out += [f"| {c} | {x} | {n} |" for (c, x), n in sorted(rej.items(), key=lambda kv: -kv[1])]
    ex = collections.Counter()
    for r in rows:
        if r["accepted"] and not r["export_ok"]:
            for x in r["reasons"] or []:
                ex[(r["cond"], x.split(":")[0].split(" lid")[0])] += 1
    out += ["", "| condition | export-time failure among accepted candidates | count |", "|---|---|---|"]
    out += [f"| {c} | {x} | {n} |" for (c, x), n in sorted(ex.items(), key=lambda kv: -kv[1])]
    return "\n".join(out)


def per_variation_table(rows):
    out = ["| variation | family | " + " | ".join(f"ep{E}" for E in EPS) + " |", "|---|---|" + "---|" * len(EPS)]
    vids = []
    for r in rows:
        if r["var"] not in vids:
            vids.append(r["var"])
    for v in vids:
        cells = []
        for E in EPS:
            g = next(r for r in rows if r["ep"] == E and r["var"] == v and r["cond"] == "G+")
            c = next(r for r in rows if r["ep"] == E and r["var"] == v and r["cond"] == "C")
            sym = lambda r: ("✓" if r["export_ok"] else "a") if r["accepted"] else "✗"  # noqa: E731
            cells.append(f"{sym(g)}/{sym(c)}")
        out.append(f"| {v} | {g['fam'][:1]} | " + " | ".join(cells) + " |")
    out.append("\nG+/C per cell: ✓ accepted and passes every export check, a = accepted but fails an export check (image QA / image lag), ✗ rejected at every scale.")
    return "\n".join(out)


def fig_counts(rows):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(FAMS))
    w = 0.38
    for i, c in enumerate(CONDS):
        acc = np.array([sum(r["export_ok"] is True for r in rows if r["fam"] == f and r["cond"] == c) for f in FAMS])
        aonly = np.array([sum(r["accepted"] and not r["export_ok"] for r in rows if r["fam"] == f and r["cond"] == c) for f in FAMS])
        rej = np.array([sum(not r["accepted"] for r in rows if r["fam"] == f and r["cond"] == c) for f in FAMS])
        xs = x + (i - 0.5) * w
        ax.bar(xs, acc, w * 0.92, color=COL[c], label=f"{c}: passes export checks")
        ax.bar(xs, aonly, w * 0.92, bottom=acc, color=COL[c], alpha=0.45, label=f"{c}: accepted, fails image/sync QA")
        ax.bar(xs, rej, w * 0.92, bottom=acc + aonly, color=STATUS["critical"], alpha=0.35 if i else 0.6, label=f"{c}: rejected (kinematic/label)")
        for xi, a, b, r_ in zip(xs, acc, aonly, rej):
            ax.text(xi, a + b + r_ + 0.5, f"{a}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f.replace("_", " ") for f in FAMS])
    ax.set_ylabel("variation x episode cases")
    ax.set_title("Candidates per family (10 episodes); number on top = passes every export check", loc="left")
    ax.legend(fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(FIG / "candidate_counts.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- coverage
def coverage(res, exp):
    from sfa3.recon3 import clicks
    cam = rb.Camera()
    ck = clicks()
    sw = json.loads((HERE / "outputs/init_sweep.json").read_text())
    src_xy = {E: np.array(sw["episodes"][str(E)]["source_table_xy"]) for E in EPS}
    gen = []                                   # (E, xy, uv, exported?)
    for E, r in res.items():
        for k, c in r["candidates"].items():
            if c["cond"] == "C" and c["accepted"] and np.linalg.norm(c["init_shift_m"]) > 0 and r["plans"][c["var"]]["own_scale"]["C"] == c["scale"]:
                xy = src_xy[E] + np.asarray(c["init_shift_m"][:2])
                z = cand(E, k)["table_new"]
                uv = cam.project(z[:3, 3][None])[0][0]
                ok = any(row["episode"] == E and row["variation"] == c["var"] and row["C_export_ok"] for row in exp["rows"]) if exp else None
                gen.append((E, xy, uv, ok, c["var"]))
    held = [ck[e] for e in range(10, 50)]
    src_uv = [ck[e] for e in EPS]
    # quantitative: nearest-source distance of generated initial positions; held-out coverage in image space
    d_src = [float(min(np.linalg.norm(g[1] - src_xy[E2]) for E2 in EPS)) for g in gen]
    def covered(points, by, r):
        by = np.asarray(by)
        return float(np.mean([np.min(np.linalg.norm(by - np.asarray(p), axis=1)) <= r for p in points])) if len(by) else 0.0
    gen_ok_uv = [g[2] for g in gen if g[3]]
    covq = {r: {"sources_only": covered(held, src_uv, r), "sources_plus_generated": covered(held, list(src_uv) + gen_ok_uv, r)}
            for r in (20, 40, 60)}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axs[0]
    rows = [q for v in sw["episodes"].values() for q in v["rows"] if q["cond"] == "C"]
    xs, ys = sw["grid"]["x"], sw["grid"]["y"]
    M = np.zeros((len(ys), len(xs)))
    for q in rows:
        M[ys.index(q["dy"]), xs.index(q["dx"])] += q["accepted"]
    im = ax.imshow(M / len(EPS), origin="lower", cmap="Greens", vmin=0, vmax=1,
                   extent=[100 * xs[0] - 1.25, 100 * xs[-1] + 1.25, 100 * ys[0] - 1.25, 100 * ys[-1] + 1.25])
    ax.grid(False)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            ax.text(100 * x, 100 * y, f"{int(M[i, j])}", ha="center", va="center", fontsize=7, color=INK if M[i, j] < 7 else "white")
    ax.set_xlabel("can shift x (cm, forward +)")
    ax.set_ylabel("can shift y (cm, left +)")
    ax.set_title("Kinematic feasibility of new initial positions\n(C, plan-only sweep; cell = episodes of 10 accepted)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of episodes")
    ax = axs[1]
    for E in EPS:
        ax.scatter(*(100 * src_xy[E]), marker="x", s=60, color=COL["source"], zorder=4)
    g_ok = np.array([100 * g[1] for g in gen if g[3]]) if any(g[3] for g in gen) else np.zeros((0, 2))
    g_no = np.array([100 * g[1] for g in gen if not g[3]]) if any(not g[3] for g in gen) else np.zeros((0, 2))
    if len(g_no):
        ax.scatter(g_no[:, 0], g_no[:, 1], s=26, facecolor="none", edgecolor=COL["C"], lw=1.2, label=f"C accepted, fails image QA (n={len(g_no)})")
    if len(g_ok):
        ax.scatter(g_ok[:, 0], g_ok[:, 1], s=30, color=COL["C"], label=f"C exported (n={len(g_ok)})")
    ax.scatter([], [], marker="x", color=COL["source"], label="source initial positions (10)")
    ax.set_xlabel("x (cm, robot base)")
    ax.set_ylabel("y (cm, robot base)")
    ax.set_title("Initial can positions on the table (world)", loc="left")
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal")
    ax = axs[2]
    hu = np.array(held)
    ax.scatter(hu[:, 0], hu[:, 1], s=22, color=COL["heldout"], marker="s", label="held-out real (pool 10-49, human clicks)")
    su = np.array(src_uv)
    ax.scatter(su[:, 0], su[:, 1], s=60, color=COL["source"], marker="x", label="source real (0-9)")
    if gen_ok_uv:
        gu = np.array(gen_ok_uv)
        ax.scatter(gu[:, 0], gu[:, 1], s=26, color=COL["C"], label="generated, exported")
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title("Initial lid position in body_cam (image space)", loc="left")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "initial_position_coverage.png", dpi=150)
    plt.close(fig)
    return {"n_generated_initial_positions": len(gen), "n_exported": len(gen_ok_uv),
            "dist_to_nearest_source_cm": {"median": 100 * float(np.median(d_src)) if d_src else None,
                                          "max": 100 * float(np.max(d_src)) if d_src else None},
            "heldout_initial_positions_within_r_px": covq,
            "note": "pass-point (hand-over) changes do not move the initial position; only family C/D candidates appear here"}


def pass_coverage(res):
    out = {}
    for fam in FAMS:
        P = []
        for E, r in res.items():
            for k, c in r["candidates"].items():
                if c["cond"] == "C" and c["accepted"] and r["plans"][c["var"]]["family"] == fam and r["plans"][c["var"]]["own_scale"]["C"] == c["scale"]:
                    out.setdefault(fam, []).append(c["pass_can_displacement_mm"])
    return {f: {"n": len(v), "median_pass_displacement_mm": float(np.median(v)), "max": float(np.max(v))} for f, v in out.items()}


# ----------------------------------------------------------------------------- trajectories / timing
def fig_trajectory(res, E=0, vid="init_x+6"):
    r = res[E]
    s = r["plans"].get(vid, {}).get("own_scale", {}).get("C")
    if s is None:
        cands = [v for v, p in r["plans"].items() if p["family"].startswith("C_") and p["own_scale"]["C"] is not None]
        if not cands:
            return
        vid = cands[-1]
        s = r["plans"][vid]["own_scale"]["C"]
    z = cand(E, f"{vid}__C__s{s}")
    zg = cand(E, f"{vid}__G+__s{r['plans'][vid]['own_scale']['G+']}") if r["plans"][vid]["own_scale"]["G+"] else None
    src = cand(E, "identity__C__s1.0")
    robot = rb.Robot()

    def tcp(st):
        g = rb.Robot.split(st)
        out = []
        for t in range(len(st)):
            robot.fk(g["R"][t], g["L"][t])
            out.append(robot.frame(robot.tcp["R"])[:3, 3])
        return 100 * np.array(out)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    P0, P1 = tcp(src["state"]), tcp(z["state"])
    axs[0].plot(P0[:, 0], P0[:, 1], color=COL["source"], ls="--", label="source giver TCP")
    if zg is not None:
        axs[0].plot(*tcp(zg["state"])[:, :2].T, color=COL["G+"], lw=1.2, label="G+")
    axs[0].plot(P1[:, 0], P1[:, 1], color=COL["C"], label="C")
    axs[0].scatter(*(100 * src["can"][0][:2, 3]), marker="x", s=70, color=COL["source"], label="source can")
    axs[0].scatter(*(100 * z["can"][0][:2, 3]), marker="o", s=60, color=COL["C"], label="new can")
    axs[0].set_title(f"ep{E} {vid}: giver TCP path (top view, cm)", loc="left")
    axs[0].set_aspect("equal")
    axs[0].legend(fontsize=7)
    t = np.arange(len(z["state"]))
    for j, ax in zip((1, 4), axs[1:]):
        i = rb.NAMES.index(f"RJ{j}.pos")
        ax.plot(t, src["state"][:, i], color=COL["source"], ls="--", label="source state")
        ax.plot(t, z["state"][:, i], color=COL["C"], label="C planned state")
        ax.plot(t, z["action"][:, i], color=COL["C"], lw=1.0, alpha=0.6, label="C action (leads 2 frames)")
        ax.set_title(f"RJ{j} (rad)", loc="left")
        ax.set_xlabel("frame")
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"trajectory_ep{E}_{vid}.png", dpi=150)
    plt.close(fig)


def fig_timing(res, E=0):
    r = res[E]
    vids = ["identity", "timing_early", "timing_late", "rot_handover_early6", "rot_handover_nominal", "rot_handover_late6"]
    fig, ax = plt.subplots(figsize=(11, 4))
    y = 0
    ticks = []
    for vid in vids:
        p = r["plans"].get(vid)
        if p is None:
            continue
        s = p["own_scale"]["C"] or p["common_scale"] or 1.0
        key = f"{vid}__C__s{s}" if (HERE / f"outputs/ep{E}/candidates/{vid}__C__s{s}.npz").exists() else None
        if key is None:
            continue
        z = cand(E, key)
        g = rb.Robot.split(z["state"])
        a = rb.Robot.split(np.nan_to_num(z["action"], nan=0.0))
        for side, alpha in (("R", 0.9), ("L", 0.5)):
            cl = g["g" + side] < 0.7
            idx = np.flatnonzero(np.diff(np.r_[0, cl.astype(int), 0]))
            for t0, t1 in zip(idx[::2], idx[1::2]):
                ax.add_patch(plt.Rectangle((t0, y + 0.15), t1 - t0, 0.7, color=COL["C"], alpha=alpha, lw=0))
            cm = a["g" + side] < 0.5
            idx = np.flatnonzero(np.diff(np.r_[0, cm.astype(int), 0]))
            for t0, t1 in zip(idx[::2], idx[1::2]):
                ax.plot([t0, t1], [y + 0.5, y + 0.5], color=INK, lw=1.0)
            acc = "✓" if p["own_scale"]["C"] else "✗"
            ticks.append((y + 0.5, f"{vid} {'giver' if side == 'R' else 'receiver'} {acc}"))
            y += 1
        m = json.loads(json.dumps(r["candidates"][key]["marks"]))
        if "tilt_start" in m:
            ax.axvspan(m["tilt_start"], m["tilt_end"], ymin=1 - (y) / 12.5, ymax=1 - (y - 2) / 12.5, color=COL["G+"], alpha=0.12, lw=0)
    ax.set_yticks([p for p, _ in ticks], [l for _, l in ticks], fontsize=7)
    ax.set_ylim(y, 0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("frame")
    ax.set_title(f"ep{E} C candidates: gripper closed by STATE (bars) and COMMAND (black); blue band = dual-hold tilt window", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"grasp_handover_timing_ep{E}.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- corruption, runtime
def corruption_table(res):
    tests = collections.OrderedDict()
    for E, r in res.items():
        ct = r.get("corruption_tests", {})
        for k, v in ct.get("tests", {}).items():
            tests.setdefault(k, []).append(v.get("detected"))
    out = ["| control (base: pos_right_up / C) | detected on episodes | detectors that fired (ep0) |", "|---|---|---|"]
    r0 = res[EPS[0]].get("corruption_tests", {}).get("tests", {})
    for k, v in tests.items():
        n = sum(bool(x) for x in v)
        fired = ""
        if k in r0:
            d = r0[k]
            fired = ", ".join(n_ for n_, f in (("construction", not d.get("labels_construction", True)), ("per-joint lead", not d.get("labels_per_joint", True)),
                                               ("gripper chronology", not d.get("labels_gripper", True)), ("verifier", not d.get("verifier", True)),
                                               ("global lead", d.get("global_lead") not in (None, 2)), ("image lag", d.get("image_lag_ok") is False)) if f)
            if k == "tampered_hash_differs":
                fired = "hash"
        label = "intended (must NOT fire)" if k == "intended" else k
        out.append(f"| {label} | {n}/{len(v)} {'(fired)' if k == 'intended' else ''} | {fired} |")
    return "\n".join(out)


def runtime(res, exp_val, train_runs, train_eval):
    pre_u = json.loads((HERE / "outputs/bench/preprocess_uncached.json").read_text())
    pre_c = json.loads((HERE / "outputs/bench/preprocess_cached.json").read_text())
    gen = json.loads((HERE / "outputs/bench/generate.json").read_text()) if (HERE / "outputs/bench/generate.json").exists() else {}
    stages = ["plans", "render_metrics", "corruption_tests", "comparison_videos"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    ax = axs[0]
    bottom = np.zeros(len(EPS))
    cols = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    for s, c in zip(["preprocess_recon_plate"] + stages, cols):
        v = np.array([res[E]["runtime_s"].get(s, 0) for E in EPS])
        ax.bar(np.arange(len(EPS)), v, 0.7, bottom=bottom, color=c, label=s)
        bottom += v
    ax.set_xticks(np.arange(len(EPS)), [f"ep{E}" for E in EPS])
    ax.set_ylabel("s")
    ax.set_title("Per-episode generation time by stage (2 lanes in parallel, 11 CPU workers each)", loc="left")
    ax.legend(fontsize=7)
    ax = axs[1]
    rows = list(csv.reader(open(HERE / "outputs/bench/nvidia_smi_generate.csv"))) if (HERE / "outputs/bench/nvidia_smi_generate.csv").exists() else []
    for gpu, c in (("0", COL["G+"]), ("1", COL["C"])):
        u = [float(r[2]) for r in rows if r[1].strip() == gpu]
        m = [float(r[3]) / 1024 for r in rows if r[1].strip() == gpu]
        ax.plot(np.arange(len(u)) / 60, u, color=c, lw=0.8, label=f"GPU{gpu} util %")
        ax.plot(np.arange(len(m)) / 60, m, color=c, lw=1.5, ls="--", label=f"GPU{gpu} memory GB")
    ax.set_xlabel("minutes into the 10-episode generation run")
    ax.set_title("GPU utilisation and memory during generation", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "runtime_breakdown.png", dpi=150)
    plt.close(fig)
    util = {g: float(np.mean([float(r[2]) for r in rows if r[1].strip() == g])) for g in ("0", "1")} if rows else {}
    peak_mem = {g: float(max(float(r[3]) for r in rows if r[1].strip() == g)) / 1024 for g in ("0", "1")} if rows else {}
    n_valid = {c: sum(1 for E in EPS for vid, p in res[E]["plans"].items() if vid != "identity" and p["own_scale"][c] is not None) for c in CONDS}
    n_export = exp_val.get("counts", {})
    return {"preprocess_uncached_wall_s": pre_u["wall_s"], "preprocess_cached_wall_s": pre_c["wall_s"],
            "sam2_s_per_episode": [e["recon_runtime_s"]["sam2"] for e in pre_u["episodes"]],
            "sam2_peak_vram_gb": [e["peak_vram_gb"] for e in pre_u["episodes"]],
            "generation_wall_s": gen.get("wall_s"), "per_episode_stage_s": {E: res[E]["runtime_s"] for E in EPS},
            "raft_s_total": sum(res[E]["gpu"]["raft_s"] for E in EPS), "raft_clips": sum(res[E]["gpu"]["n"] for E in EPS),
            "raft_peak_vram_gb": [res[E]["gpu"].get("peak_vram_gb_raft") for E in EPS],
            "gpu_mean_util_pct_during_generation": util, "gpu_peak_mem_gb_during_generation": peak_mem,
            "accepted_candidates": n_valid, "export": n_export}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    res = {E: R(E) for E in EPS if (HERE / f"outputs/ep{E}/results.json").exists()}
    exp = json.loads((HERE / "outputs/dataset_export/export_checks.json").read_text()) if (HERE / "outputs/dataset_export/export_checks.json").exists() else None
    val = json.loads((HERE / "outputs/dataset_export/validation.json").read_text()) if (HERE / "outputs/dataset_export/validation.json").exists() else {}
    rows = counts(res, exp)
    (TAB / "counts.md").write_text(count_tables(rows, exp) + "\n")
    (TAB / "per_variation.md").write_text(per_variation_table(rows) + "\n")
    (TAB / "corruption.md").write_text(corruption_table(res) + "\n")
    fig_counts(rows)
    cov = coverage(res, exp)
    cov["pass_point_displacement_by_family"] = pass_coverage(res)
    for E in (0, 3):
        if E in res:
            fig_trajectory(res, E)
            fig_timing(res, E)
    tr = json.loads((HERE / "outputs/train_screen_runs.json").read_text()) if (HERE / "outputs/train_screen_runs.json").exists() else None
    te = json.loads((HERE / "outputs/train_screen_eval.json").read_text()) if (HERE / "outputs/train_screen_eval.json").exists() else None
    rt = runtime(res, {"counts": {k: v.get("episodes") for k, v in val.get("timing", {}).items()}}, tr, te)
    if val:
        rt["export_timing"] = val.get("timing")
    summary = {"coverage": cov, "runtime": rt, "n_rows": len(rows)}
    (TAB / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(open(TAB / "counts.md").read())
    print(json.dumps(cov, indent=1, default=float))


if __name__ == "__main__":
    main()
