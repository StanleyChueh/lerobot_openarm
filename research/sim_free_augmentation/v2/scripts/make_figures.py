"""Aggregate tables and figures across episodes (reads outputs/ep*/results.json, candidates, cached raw frames).

    cd research/sim_free_augmentation/v2
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/make_figures.py --episodes 0 4 8
"""

from __future__ import annotations

import argparse
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
from sfa2 import metrics2 as M2, planner as pl, recon, sync, verify as vf  # noqa: E402
from sfa2.recon import rb, rd  # noqa: E402

# validated categorical slots 1-3 (all-pairs pass, light surface); source = neutral ink
COL = {"source": "#52514e", "G": "#2a78d6", "P": "#eb6834", "C": "#1baf7a"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
SURF, GRID, INK, INK2 = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2.0,
                     "legend.frameon": False, "axes.titlesize": 10, "axes.titlecolor": INK})
FIG = HERE / "outputs/figures"


def load(E):
    return json.loads((HERE / f"outputs/ep{E}/results.json").read_text())


def cand(E, key):
    z = np.load(HERE / f"outputs/ep{E}/candidates/{key}.npz")
    return {"state": z["state"], "action": z["action"], "valid": z["valid"], "can": z["can"], "tau": z["tau"],
            "q": {"R": z["qR"], "L": z["qL"]}}


# ----------------------------------------------------------------------------- tables
def acceptance_tables(R: dict, variations) -> str:
    lines = []
    conds = pl.CONDITIONS
    hdr = "| variation | " + " | ".join(f"ep{E} {c}" for E in R for c in conds) + " |"
    lines += [hdr, "|" + "---|" * (1 + len(R) * len(conds))]
    for v in variations:
        row = [v["id"]]
        for E, res in R.items():
            for c in conds:
                p = res["candidates"][f"{v['id']}__{c}"]
                row.append(f"✓ ×{p['scale']}" if p["accepted"] else "✗ " + ",".join(f.replace("attach6d_", "6D-") for f in p["verify"]["failed"][:2]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("| condition | " + " | ".join(f"ep{E}" for E in R) + " | total accepted | accepted at full scale |")
    lines.append("|---|" + "---|" * (len(R) + 2))
    for c in conds:
        acc = [sum(res["candidates"][f"{v['id']}__{c}"]["accepted"] for v in variations if v["id"] != "identity") for res in R.values()]
        full = sum(res["candidates"][f"{v['id']}__{c}"]["accepted"] and res["candidates"][f"{v['id']}__{c}"]["scale"] == 1.0
                   for res in R.values() for v in variations if v["id"] != "identity")
        n = sum(1 for v in variations if v["id"] != "identity")
        lines.append(f"| {c} | " + " | ".join(f"{a}/{n}" for a in acc) + f" | {sum(acc)}/{n * len(R)} | {full}/{n * len(R)} |")
    return "\n".join(lines)


def metric_rows(R: dict, variations):
    rows = []
    for E, res in R.items():
        for v in variations:
            for c in pl.CONDITIONS:
                k = f"{v['id']}__{c}"
                p = res["candidates"][k]
                m = res["clips"][p["render_alias"]]["metrics"]
                kp = m.get("keypoints", {}).get("handover", {})
                rows.append({"ep": E, "var": v["id"], "type": v["type"], "cond": c, "accepted": p["accepted"], "scale": p["scale"],
                             "kp_R": (kp.get("R") or {}).get("median_px"), "kp_L": (kp.get("L") or {}).get("median_px"),
                             "kp_can": (kp.get("can") or {}).get("median_px"),
                             "kp_cov": np.nanmean([(kp.get(g) or {}).get("coverage", np.nan) for g in ("R", "L", "can")]),
                             "warp_robot": m["temporal"]["robot_object"]["mean"], "warp_bg": m["temporal"]["background"]["mean"],
                             "warp_hand": m["temporal"]["robot_object_handover"]["mean"],
                             "lid_g": m["lid"]["giver_hold"]["median_px"], "lid_r": m["lid"]["receiver_hold"]["median_px"],
                             "lid_cov_r": m["lid"]["receiver_hold"]["coverage"], "lid_cov_g": m["lid"]["giver_hold"]["coverage"],
                             "lead_expected": res["recon"]["action_lead"]["k_best"],
                             "axis_g": m["can_axis_2d"]["giver_hold"]["median_deg"], "axis_r": m["can_axis_2d"]["receiver_hold"]["median_deg"],
                             "ghost": (m.get("ghost") or {}).get("ghost_frac_median"), "noise": (m.get("ghost") or {}).get("background_noise_frac"),
                             "pd": (m.get("pose_discrimination_vs_source_labels") or {}).get("frac_frames_commanded_better"),
                             "lag": m["lag_scan"]["lag"], "lead": m["action_lead"]["lead"], "lag_expected": res["recon"]["lag_frames"],
                             "pass_mm": p["pass_can_displacement_mm"], "render_s": m["render_s"]})
    return rows


def fmt(x, d=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def metric_table(rows) -> str:
    out = ["| ep | variation | cond | acc | kp R/L/can med px (cov) | warp err robot / bg | lid err giver / recv px (recv cov) | can-axis 2D giver / recv ° | ghost / noise | prefers own labels | lag (exp) | lead |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["var"] == "identity" and r["cond"] != "C":
            continue
        out.append(f"| {r['ep']} | {r['var']} | {r['cond']} | {'✓' if r['accepted'] else '✗'}{'' if r['scale'] == 1 else ' ×' + str(r['scale'])} | "
                   f"{fmt(r['kp_R'])}/{fmt(r['kp_L'])}/{fmt(r['kp_can'])} ({fmt(r['kp_cov'], 2)}) | {fmt(r['warp_robot'], 2)} / {fmt(r['warp_bg'], 2)} | "
                   f"{fmt(r['lid_g'])} / {fmt(r['lid_r'])} ({fmt(r['lid_cov_r'], 2)}) | {fmt(r['axis_g'])} / {fmt(r['axis_r'])} | "
                   f"{fmt(r['ghost'], 2)} / {fmt(r['noise'], 2)} | {fmt(r['pd'], 2)} | {r['lag']} ({r['lag_expected']}) | {r['lead']} |")
    return "\n".join(out)


def image_qa(r, qa):
    return all(r[f"lid_{w}"] is not None and r[f"lid_{w}"] <= qa["lid_median_px_max"] and r[f"lid_cov_{w}"] >= qa["lid_coverage_min"]
               for w in ("g", "r"))


def yield_table(rows, qa, episodes) -> str:
    out = ["| condition | " + " | ".join(f"ep{E} accepted / sync ok / image QA" for E in episodes) + " | overall yield |",
           "|---|" + "---|" * (len(episodes) + 1)]
    for c in pl.CONDITIONS:
        cells, tot, n = [], 0, 0
        for E in episodes:
            rr = [r for r in rows if r["cond"] == c and r["ep"] == E and r["var"] != "identity"]
            acc = [r for r in rr if r["accepted"]]
            syn = [r for r in acc if abs(r["lag"] - r["lag_expected"]) <= 0.5 and r["lead"] == r["lead_expected"]]
            qa_ = [r for r in syn if image_qa(r, qa)]
            cells.append(f"{len(acc)} / {len(syn)} / {len(qa_)} of {len(rr)}")
            tot += len(qa_)
            n += len(rr)
        out.append(f"| {c} | " + " | ".join(cells) + f" | {tot}/{n} |")
    return "\n".join(out)


def summary_by_condition(rows) -> str:
    out = ["| condition | n clips (accepted, non-identity) | kp median px R / L / can | lid err receiver px | can-axis 2D receiver ° | ghost frac | warp err robot region | lag = expected | lead = 2 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in pl.CONDITIONS:
        rr = [r for r in rows if r["cond"] == c and r["accepted"] and r["var"] != "identity"]
        med = lambda k: np.nanmedian([np.nan if r[k] is None else r[k] for r in rr]) if rr else np.nan  # noqa: E731
        out.append(f"| {c} | {len(rr)} | {fmt(med('kp_R'))} / {fmt(med('kp_L'))} / {fmt(med('kp_can'))} | {fmt(med('lid_r'))} | "
                   f"{fmt(med('axis_r'))} | {fmt(med('ghost'), 2)} | {fmt(med('warp_robot'), 2)} | "
                   f"{sum(abs(r['lag'] - r['lag_expected']) <= 0.5 for r in rr)}/{len(rr)} | {sum(r['lead'] == r.get('lead_expected', 2) for r in rr)}/{len(rr)} |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- figures
def fig_acceptance(R, variations):
    vids = [v["id"] for v in variations]
    cols = [(E, c) for E in R for c in pl.CONDITIONS]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 0.34 * len(vids) + 1.2))
    ax.grid(False)
    for i, vid in enumerate(vids):
        for j, (E, c) in enumerate(cols):
            p = R[E]["candidates"][f"{vid}__{c}"]
            if p["accepted"] and p["scale"] == 1.0:
                col, txt = STATUS["good"], "✓"
            elif p["accepted"]:
                col, txt = STATUS["warning"], f"×{p['scale']}"
            else:
                col, txt = STATUS["critical"], "✗"
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=col, alpha=0.85, lw=0))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", color="white" if col != STATUS["warning"] else INK, fontsize=8)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(len(vids), 0)
    ax.set_xticks(np.arange(len(cols)) + 0.5, [f"ep{E}\n{c}" for E, c in cols], fontsize=8)
    ax.set_yticks(np.arange(len(vids)) + 0.5, vids, fontsize=8)
    ax.set_title("Verifier acceptance (✓ full scale, ×s accepted after shrinking to scale s, ✗ rejected after 3 tries)", loc="left")
    for j in range(1, len(R)):
        ax.axvline(3 * j, color=INK2, lw=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "acceptance_grid.png", dpi=150)
    plt.close(fig)


def fig_6d(E, vid, V):
    keys = {c: f"{vid}__{c}" for c in pl.CONDITIONS}
    src = {"state": V.rec.state, "action": V.rec.action, "can": V.src_can, "tau": np.arange(V.rec.T, float)}
    cs = {c: cand(E, k) for c, k in keys.items()}
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    t = np.arange(V.rec.T)
    for i, (lab, ax) in enumerate(zip("xyz", axs[0])):
        ax.plot(t, 100 * src["can"][:, i, 3], color=COL["source"], lw=1.2, ls="--", label="source")
        for c in pl.CONDITIONS:
            ax.plot(t, 100 * cs[c]["can"][:, i, 3], color=COL[c], lw=1.5, label=c)
        ax.set_title(f"can lid {lab} (cm, robot base)", loc="left")
    tilt = lambda C: np.degrees(np.arccos(np.clip(-C["can"][:, 2, 2], -1, 1)))  # noqa: E731
    axs[1, 0].plot(t, tilt(src), color=COL["source"], lw=1.2, ls="--", label="source")
    for c in pl.CONDITIONS:
        axs[1, 0].plot(t, tilt(cs[c]), color=COL[c], lw=1.5, label=c)
    axs[1, 0].set_title("can axis tilt from vertical (°)", loc="left")
    for j, (q, ttl) in enumerate((("mm", "receiver attachment excess drift (mm)"), ("spin", "receiver excess spin about can axis (°)"))):
        ax = axs[1, 1 + j]
        for c in pl.CONDITIONS:
            d = vf.drift_series(V, cs[c])
            ax.plot(t, d["L"][q], color=COL[c], lw=1.5, label=c)
        thr = vf.TOL["attach_mm"] if q == "mm" else vf.TOL["attach_spin_deg"]
        ax.axhline(thr, color=STATUS["critical"], lw=1.0, ls=":")
        ax.text(t[-1], thr, " verifier threshold", color=INK2, va="bottom", ha="right", fontsize=8)
        ax.set_title(ttl, loc="left")
    m = load(E)["candidates"][keys["C"]]["marks"]
    for ax in axs.ravel():
        for ev, ls in (("right_close", "-"), ("left_close", "-"), ("right_open", "-")):
            ax.axvline(m[ev], color=GRID, lw=1.0, zorder=0)
        ax.set_xlabel("frame") if ax in axs[1] else None
    axs[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Episode {E}, {vid}: 6D object motion and receiver-relative attachment (grey verticals: R close, L close, R open)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / f"motion6d_ep{E}_{vid}.png", dpi=150)
    plt.close(fig)


def fig_timeline(E, vids):
    res = load(E)
    fig, axs = plt.subplots(len(vids), 1, figsize=(10, 1.9 * len(vids)), sharex=True, squeeze=False)
    for ax, vid in zip(axs[:, 0], vids):
        y = 0
        ticks = []
        for c in pl.CONDITIONS:
            k = f"{vid}__{c}"
            z = cand(E, k)
            g = rb.Robot.split(z["state"])
            a = rb.Robot.split(np.nan_to_num(z["action"], nan=0))
            for s, nm in (("R", "giver"), ("L", "receiver")):
                cl = g["g" + s] < recon.CLOSED
                for t0, t1 in vf.intervals(cl):
                    ax.add_patch(plt.Rectangle((t0, y + 0.15), t1 - t0, 0.7, color=COL[c], alpha=0.9 if s == "R" else 0.55, lw=0))
                cmd = vf.intervals(a["g" + s] < 0.5)
                for t0, t1 in cmd:
                    ax.plot([t0, t1], [y + 0.5, y + 0.5], color=INK, lw=1.0)
                ticks.append((y + 0.5, f"{c} {nm}"))
                y += 1
            hold = res["candidates"][k]
        ax.set_yticks([p for p, _ in ticks], [l for _, l in ticks], fontsize=8)
        ax.set_ylim(y, 0)
        ax.set_title(f"ep{E} {vid}: gripper closed by STATE (bars) and by COMMAND (black line); "
                     + ", ".join(f"{c}: {'✓' if res['candidates'][f'{vid}__{c}']['accepted'] else '✗'}" for c in pl.CONDITIONS), loc="left")
        ax.grid(axis="y", visible=False)
    axs[-1, 0].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(FIG / f"contact_timeline_ep{E}.png", dpi=150)
    plt.close(fig)


def fig_sync(E, rec, key):
    res = load(E)
    c = cand(E, key)
    alias = res["candidates"][key]["render_alias"]
    frames = np.load(HERE / f"outputs/cache/ep{E}/raw/{alias}.npy", mmap_mode="r")
    robot, cam = rb.Robot(), rb.Camera()
    from sfa2 import render2
    _, _, canv = render2.video_states(rec, c)
    uv, ok = M2.lid_track(np.asarray(frames), cam, canv)
    m = res["candidates"][key]["marks"]
    win = range(m["grasp"] + 10, m["capture"] - 3)
    lags = np.arange(-10.0, 7.01, 0.25)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for lab, st, col, ls in (("exported labels", c["state"], COL["C"], "-"), ("state delayed 3", sync.delay_state(c, 3)["state"], COL["P"], "--"),
                             ("state delayed 6", sync.delay_state(c, 6)["state"], COL["G"], "--"), ("source labels", rec.state, COL["source"], ":")):
        s = M2.lag_scan(robot, cam, st, rec.X["R"], uv, ok, win, lags)
        x = np.array(list(s["curve"]))
        axs[0].plot(x, [s["curve"][k] for k in x], color=col, ls=ls, label=f"{lab} (argmin {s['lag']})")
    axs[0].axvline(rec.lag, color=STATUS["good"], lw=1.0, ls=":")
    axs[0].set_title(f"image-to-state lag scan on the generated clip ({key}); expected {rec.lag}", loc="left")
    axs[0].set_xlabel("assumed lag (frames)")
    axs[0].set_ylabel("median lid reprojection error (px)")
    axs[0].legend(fontsize=8)
    for lab, d, col, ls in (("exported actions", c, COL["C"], "-"), ("actions delayed 6", sync.delay_actions(c, 6), COL["G"], "--"),
                            ("source", {"state": rec.state, "action": rec.action}, COL["source"], ":")):
        r = sync.action_lead(d["state"], d["action"])
        axs[1].plot(list(r["err"]), list(r["err"].values()), color=col, ls=ls, marker="o", ms=4, label=f"{lab} (argmin {r['lead']})")
    axs[1].set_title("command-to-state lead: mean |a(t) - s(t+k)| (rad)", loc="left")
    axs[1].set_xlabel("k (frames)")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"sync_ep{E}_{key}.png", dpi=150)
    plt.close(fig)


def fig_diversity(R, recs):
    fig, axs = plt.subplots(1, len(R), figsize=(3.6 * len(R), 3.4), squeeze=False)
    for ax, (E, res) in zip(axs[0], R.items()):
        for c, mk in zip(pl.CONDITIONS, ("o", "s", "^")):
            P = []
            for k, p in res["candidates"].items():
                if p["cond"] == c and p["accepted"]:
                    z = cand(E, k)
                    P.append(100 * z["can"][p["marks"]["pass"], :3, 3])
            P = np.array(P)
            if len(P):
                ax.scatter(P[:, 1], P[:, 2], s=34 if c != "C" else 60, marker=mk, facecolor="none" if c != "C" else COL[c],
                           edgecolor=COL[c], lw=1.5, label=f"{c} accepted (n={len(P)})", zorder=3 if c == "C" else 2)
        for E2, r2 in recs.items():
            m = R[E2]["candidates"][f"identity__C"]["marks"]
            p = 100 * cand(E2, "identity__C")["can"][m["pass"], :3, 3]
            ax.scatter([p[1]], [p[2]], marker="x", s=60, color=COL["source"], zorder=4)
            ax.annotate(f"src ep{E2}", (p[1], p[2]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK2)
        ax.set_title(f"episode {E}: can lid at the pass frame", loc="left")
        ax.set_xlabel("y (cm, left +)")
        ax.set_ylabel("z (cm, up +)")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "handover_diversity.png", dpi=150)
    plt.close(fig)


def stills(E, vid, frames_at):
    res = load(E)
    F = recon.data.load(E, cams=(recon.data.CAMS[0],)).frames[recon.data.CAMS[0]]
    for t in frames_at:
        row = [rd.label(F[t], f"original ep{E} t={t}")]
        for c in pl.CONDITIONS:
            k = f"{vid}__{c}"
            p = res["candidates"][k]
            a = np.load(HERE / f"outputs/cache/ep{E}/raw/{p['render_alias']}.npy", mmap_mode="r")
            row.append(rd.label(np.array(a[t]), f"{c} {'ACCEPTED' if p['accepted'] else 'REJECTED'}"))
        cv2.imwrite(str(FIG / f"still_ep{E}_{vid}_t{t:03d}.jpg"), cv2.cvtColor(np.concatenate(row, 1), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--config", default=str(HERE / "configs/frozen_v2.json"))
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    variations = cfg["variations"]
    R = {E: load(E) for E in args.episodes if (HERE / f"outputs/ep{E}/results.json").exists()}
    recs = {E: recon.build(E, HERE / f"outputs/cache/ep{E}", with_plate=False, log=lambda *a: None) for E in R}
    robot = rb.Robot()
    fig_acceptance(R, variations)
    rows = metric_rows(R, variations)
    (HERE / "outputs/tables").mkdir(parents=True, exist_ok=True)
    (HERE / "outputs/tables/acceptance.md").write_text(acceptance_tables(R, variations) + "\n")
    (HERE / "outputs/tables/clip_metrics.md").write_text(metric_table(rows) + "\n")
    (HERE / "outputs/tables/summary_by_condition.md").write_text(summary_by_condition(rows) + "\n")
    (HERE / "outputs/tables/yield.md").write_text(yield_table(rows, cfg["image_qa"], list(R)) + "\n")
    (HERE / "outputs/tables/clip_metrics.json").write_text(json.dumps(rows, indent=1, default=float))
    for E in R:
        V = vf.Verifier(recs[E], robot)
        for vid in ("grasp_spin30", "grasp_higher", "approach_above", "pos_right_up"):
            fig_6d(E, vid, V)
        fig_timeline(E, ["pos_right_up", "grasp_spin30", "timing_early"])
        base = R[E]["corruption_tests"].get("base")
        if base:
            fig_sync(E, recs[E], base)
        m = R[E]["candidates"]["grasp_spin30__C"]["marks"]
        stills(E, "grasp_spin30", [m["capture"], m["giver_released"] + 15])
        m = R[E]["candidates"]["approach_above__C"]["marks"]
        stills(E, "approach_above", [m["approach_mid"]])
        m = R[E]["candidates"]["pos_right_up__C"]["marks"]
        stills(E, "pos_right_up", [m["pass"]])
    fig_diversity(R, recs)
    print(open(HERE / "outputs/tables/acceptance.md").read())
    print(open(HERE / "outputs/tables/summary_by_condition.md").read())
    print(open(HERE / "outputs/tables/yield.md").read())


if __name__ == "__main__":
    main()
