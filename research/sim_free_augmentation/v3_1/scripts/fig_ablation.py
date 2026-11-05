"""Task A figures from outputs/vision_ablation/scores.json."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = Path(__file__).resolve().parents[1]
S = json.loads((HERE / "outputs/vision_ablation/scores.json").read_text())
COL = {"R": "#52514e", "RpGp": "#2a78d6", "RpC": "#1baf7a"}
LAB = {"R": "R", "RpGp": "R+G+", "RpC": "R+C"}
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e1e0d9", "axes.grid": True,
                     "grid.color": "#e1e0d9", "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "legend.frameon": False})
conds = ["blank", "wrong_episode", "object_occluded", "state_only"]
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
for i, a in enumerate(LAB):
    x = np.arange(len(conds)) + (i - 1) * 0.26
    v = [100 * S["per_arm"][a][c]["primary"]["rel_change_vs_original"] for c in conds]
    lo = [100 * S["per_arm"][a][c]["primary"]["rel_change_ci95"][0] for c in conds]
    hi = [100 * S["per_arm"][a][c]["primary"]["rel_change_ci95"][1] for c in conds]
    axs[0].bar(x, v, 0.24, color=COL[a], label=LAB[a])
    axs[0].errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor="#0b0b0b", lw=1)
axs[0].axhline(10, color="#d03b3b", ls=":", lw=1)
axs[0].text(3.4, 11, "pre-declared 10%", fontsize=7, color="#52514e", ha="right")
axs[0].set_xticks(range(len(conds)), ["blank", "wrong episode", "can occluded", "state only"])
axs[0].set_ylabel("% change of phase-balanced error vs original RGB")
axs[0].set_title("Image interventions (10 primary held-out recordings,\n3 seeds; bars = recording-bootstrap 95% CI)", loc="left")
axs[0].legend()
ph = ["approach", "grasp", "carry", "handover", "release"]
for a in LAB:
    axs[1].plot(ph, [S["per_arm"][a]["original"]["per_phase_all"][p] for p in ph], marker="o", color=COL[a], label=LAB[a])
axs[1].set_ylabel("normalised arm-action error (per step)")
axs[1].set_title("Per-step phase error, original RGB (40 held-out recordings)", loc="left")
axs[1].legend()
for a in LAB:
    axs[2].plot(ph, [S["per_arm"][a]["blank"]["per_phase_all"][p] - S["per_arm"][a]["original"]["per_phase_all"][p] for p in ph], marker="o", color=COL[a], label=f"{LAB[a]} blank")
    axs[2].plot(ph, [S["per_arm"][a]["object_occluded"]["per_phase_all"][p] - S["per_arm"][a]["original"]["per_phase_all"][p] for p in ph], marker="s", ls="--", color=COL[a], lw=1, label=f"{LAB[a]} can occluded")
axs[2].axhline(0, color="#52514e", lw=0.8)
axs[2].set_ylabel("error increase vs original")
axs[2].set_title("Where the image matters (40 recordings)", loc="left")
axs[2].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(HERE / "outputs/vision_ablation/vision_ablation.png", dpi=150)
print("ok")
