"""Evaluation-only diagnosis of speed/accel gate failures (R01, P8). For every object_workspace candidate of the given
group work dirs: shift, per-joint speed ratio to the replay envelope (sfagen/checks.py:151, same formula), the joint
that sets the maximum, and the TCP peak speed of the grasping arm relative to the same source episode (Cartesian
speed-up, independent of how IK distributes the motion over joints). Reads saved candidates only; changes nothing.
Usage: speed_gate_diag.py OUT.json CONFIG [CONFIG ...]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context

out, rows, envs = Path(sys.argv[1]), [], {}
for c in sys.argv[2:]:
    ctx = Context(c); r = ctx.robot; fps = ctx.src.fps; idx = [i for s in r.arms for i in r.sidx[s]]
    names = [f"{s}{k + 1}" for s in r.arms for k in range(len(r.sidx[s]))]
    obj = ctx.cfg["objects"][ctx.cfg["variations"]["object_targets"]["object"]]; arm = obj["grasped_by"]
    for cj in sorted((ctx.work / "candidates").glob("ep*/checks.json")):
        e = int(cj.parent.name[2:]); env = json.loads((cj.parent / "generation_manifest.json").read_text())["replay_envelope"]
        sp = np.maximum(np.asarray(env["speed"]), 1e-9); envs[ctx.work.name] = dict(zip(names, np.round(env["speed"], 3).tolist()))
        ck = json.loads(cj.read_text())
        for k, v in ck.items():
            f = cj.parent / f"{k}.npz"
            if v.get("family") != "object_workspace" or not f.exists():
                continue
            z = np.load(f); A, A0 = z["action"], z["ref_action"]
            ratio = np.abs(np.diff(A[:, idx], axis=0)).max(0) * fps / sp; j = int(ratio.argmax())
            own = np.abs(np.diff(A0[:, idx], axis=0)).max(0) * fps / sp      # source episode vs the group envelope
            tcp = lambda X: np.linalg.norm(np.diff(r.tcp_from_state(X, arm)[:, :3, 3], axis=0), axis=1) * fps  # noqa: E731
            rows.append({"group": ctx.work.name, "episode": e, "key": k, "pass": v["pass"],
                         "shift_cm": 100 * float(np.hypot(*v["shift_m"][:2])), "speed_ratio_max": float(ratio.max()),
                         "speed_ratio_joint": names[j], "source_own_ratio_max": float(own.max()),
                         "tcp_peak_ratio": float(tcp(z["state"]).max() / max(tcp(z["ref_state"]).max(), 1e-9)),
                         "frames": int(len(A))})
out.write_text(json.dumps({"envelope_speed_rad_s": envs, "rows": rows}, indent=1))
R = rows; n = len(R)
print(f"{n} object_workspace candidates")
for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 99)):
    s = [x for x in R if lo <= x["shift_cm"] < hi]
    if s:
        print(f"shift {lo:>2}-{hi:<2} cm: n={len(s):3d}  speed ratio median {np.median([x['speed_ratio_max'] for x in s]):6.2f}"
              f"  <=1.25: {sum(x['speed_ratio_max'] <= 1.25 for x in s):3d}  TCP peak ratio median {np.median([x['tcp_peak_ratio'] for x in s]):.2f}")
from collections import Counter
print("joint setting the max:", Counter(x["speed_ratio_joint"] for x in R).most_common())
print("envelope (rad/s):", json.dumps(envs))
