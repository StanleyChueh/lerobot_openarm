"""R01 figure: source vs synthetic joint angles and grasping-TCP speed for a few object_workspace candidates (evaluation only).
Usage: speed_gate_figure.py CONFIG EPISODE OUT.png KEY [KEY ...]"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
ctx = Context(sys.argv[1]); e = int(sys.argv[2]); out = sys.argv[3]; keys = sys.argv[4:]; r = ctx.robot; fps = ctx.src.fps
arm = ctx.cfg["objects"][ctx.cfg["variations"]["object_targets"]["object"]]["grasped_by"]; ix = r.sidx[arm]
d = ctx.work / "candidates" / f"ep{e}"; ck = json.loads((d / "checks.json").read_text())
env = np.asarray(json.loads((d / "generation_manifest.json").read_text())["replay_envelope"]["speed"])[:6 if arm == "L" else 12][-6:]
fig, ax = plt.subplots(1, 3, figsize=(15, 4)); t = None
for n, k in enumerate(keys):
    z = np.load(d / f"{k}.npz"); A, A0 = z["action"], z["ref_action"]; t = np.arange(len(A)) / fps
    if n == 0:
        for j in range(6):
            ax[0].plot(t, A0[:, ix[j]], color=f"C{j}", lw=2, label=f"{arm}{j + 1} source")
        ax[2].plot(t[1:], np.linalg.norm(np.diff(r.tcp_from_state(z["ref_state"], arm)[:, :3, 3], axis=0), axis=1) * fps, "k", lw=2, label="source")
    sh = 100 * np.hypot(*ck[k]["shift_m"][:2])
    ax[0].plot(t, A[:, ix[4]], "C4--", lw=1, alpha=.8)
    ax[1].plot(t[1:], np.abs(np.diff(A[:, ix[4]])) * fps, lw=1.2, label=f"{k} ({sh:.0f} cm)")
    ax[2].plot(t[1:], np.linalg.norm(np.diff(r.tcp_from_state(z["state"], arm)[:, :3, 3], axis=0), axis=1) * fps, lw=1, label=f"{sh:.0f} cm")
ax[1].plot(t[1:], np.abs(np.diff(np.load(d / f"{keys[0]}.npz")["ref_action"][:, ix[4]])) * fps, "k", lw=2, label="source")
ax[1].axhline(1.25 * env[4], color="r", ls=":", label=f"gate 1.25 x envelope ({env[4]:.2f} rad/s)")
ax[0].set(title=f"{arm} joint angles (solid = source; dashed = synthetic {arm}5)", xlabel="s", ylabel="rad"); ax[0].legend(fontsize=6, ncol=2)
ax[1].set(title=f"{arm}5 (wrist) speed", xlabel="s", ylabel="rad/s"); ax[1].legend(fontsize=7)
ax[2].set(title=f"{arm} gripper (TCP) speed", xlabel="s", ylabel="m/s"); ax[2].legend(fontsize=7)
fig.suptitle(f"{Path(sys.argv[1]).stem} ep{e}: the synthetic gripper moves no faster, but the wrist joint must turn"); fig.tight_layout(); fig.savefig(out, dpi=110)
