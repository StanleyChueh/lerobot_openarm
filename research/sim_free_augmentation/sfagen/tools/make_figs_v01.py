"""Charts for the v01 study README (CPU): coverage vs number of demos (C3), wrist render vs copy (B), can fix (A).
usage: make_figs_v01.py   (writes results/figs/*.png; panels whose inputs are missing are skipped)"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = Path(__file__).resolve().parents[1]; R = S / "results"; OUT = R / "figs"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def coverage():
    d = json.loads((R / "fewsrc/source_budget10.json").read_text()); sel = json.loads((R / "fewsrc/selection.json").read_text())["subsets"]
    rows = {tuple(r["subset"]): r for r in d["all_subsets"]}; Ns = [1, 2, 3, 4, 10]
    cov = lambda r, m="workspace": r[m]["objects"]["can"]["within_2cm"] if m != "real" else r["real"]["can"]["within_2cm"]  # noqa: E731
    best = [max(cov(r) for r in d["all_subsets"] if r["n"] == n) for n in Ns]; med = [np.median([cov(r) for r in d["all_subsets"] if r["n"] == n]) for n in Ns]
    rule = {int(n): cov(rows[tuple(s)]) for n, s in sel.items()}; real = {int(n): cov(rows[tuple(s)], "real") for n, s in sel.items()}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(Ns, best, "^--", color="#9ca3af", label="best subset (post hoc, upper bound)")
    ax.plot(Ns, med, "s:", color="#9ca3af", label="median subset")
    ax.plot(sorted(rule), [rule[n] for n in sorted(rule)], "o-", color="#16a34a", lw=2, label="frozen k-center rule (F1W, <= 11 cm)")
    ax.plot(sorted(real), [real[n] for n in sorted(real)], "o-", color="#2563eb", label="real demos only")
    ax.axhline(24, color="#dc2626", ls="--", lw=1); ax.text(10, 24.6, "published 10-demo v00 (24)", ha="right", color="#dc2626", fontsize=8)
    ax.set_xlabel("real source demos"); ax.set_ylabel("unseen recordings covered (of 40, <= 2 cm)"); ax.set_xticks(Ns); ax.set_ylim(0, 40); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "coverage_vs_demos.png", dpi=150); plt.close(fig)


def wrist():
    runs = [(p.parent.name, json.loads(p.read_text())) for p in sorted(R.glob("wrist/final_*/summary.json"))]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); x = 0; ticks = []
    for name, s in runs:
        for cam, r in s["cameras"].items():
            m, ci = r["T1"]["render_minus_copy_db"], r["T1"]["ci95"]
            ax.errorbar(x, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="o", color="#16a34a" if r["gate"]["pass"] else "#dc2626", capsize=3)
            ticks.append(f"{name.replace('final_', '')}\n{cam.split('.')[-1].replace('_cam', '')}"); x += 1
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(x)); ax.set_xticklabels(ticks, fontsize=7); ax.set_ylabel("T1 render - copy PSNR (dB)")
    ax.set_title("Wrist views: green = gate passed, red = failed", fontsize=9); fig.tight_layout(); fig.savefig(OUT / "wrist_render_vs_copy.png", dpi=150); plt.close(fig)


def canfix():
    p = R / "canfix/heldout_g60_e3_tex/compare_vs_heldout_base.json"
    if not p.exists():
        return
    d = json.loads(p.read_text()); fig, ax = plt.subplots(figsize=(4.6, 3.0))
    lab = ["cross-episode\ncan (registered)", "cross-episode\nwhole image", "same-can lift\ncan", "same-can lift\nwhole image"]
    vals = [(d["can_median_gain_db"], d["can_ci95"]), (d["all_mean_gain_db"], d["all_ci95"]), (d["lift_can_median_gain_db"], d["lift_can_ci95"]), (d["lift_all_mean_gain_db"], d["lift_all_ci95"])]
    for i, (m, ci) in enumerate(vals):
        ax.errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="o", capsize=3, color="#dc2626" if i == 0 else "#2563eb")
    ax.axhline(0, color="k", lw=0.8); ax.axhline(0.5, color="#9ca3af", ls="--", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("gain vs published renderer (dB)"); ax.set_title("A: can fix, held-out (rule needs +0.5 dB on the red point)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "canfix_heldout.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    coverage(); wrist(); canfix(); print(sorted(p.name for p in OUT.glob("*.png")))
