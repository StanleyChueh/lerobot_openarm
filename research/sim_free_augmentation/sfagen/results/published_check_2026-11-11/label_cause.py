"""For each label-mismatch episode found by published_check.py: is the exported action exactly the expected
(a + dq(t+lead)) clipped to the planner's action range [min(URDF lo, source min), max(URDF hi, source max)]?
If yes for every mismatching value, the cause is the clip in plan.py (the action was clipped, the state was not)."""
import json, sys
from pathlib import Path
import numpy as np, yaml
REPO = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation/sfagen"); sys.path.insert(0, str(REPO))
from sfagen.context import Context
from sfagen.data import Source
WORK = Path("/home/stanley/Stanley_ws/sfa_hf_export/sfagen/configs/work"); H = Path(__file__).parent
d = json.loads((H / "published_check.json").read_text()); out = {}
for stem, repo in (("openarm_pringles_handover", "ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00"),
                   ("openarm_battery_insertion", "ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00")):
    ctx = Context(WORK.parent / f"{stem}.yaml"); r = ctx.robot
    exp = Source({**ctx.cfg["source"], "root": str(WORK / stem / "dataset")})
    rows = []
    for m in d[repo]["labels"]["mismatches"]:
        S, A = exp.arrays(m["episode"]); S0, A0 = ctx.src.arrays(m["source"])
        lead = json.loads((WORK / stem / f"candidates/ep{m['source']}/checks.json").read_text())[m["variation"]]["lead"]
        fut = np.minimum(np.arange(len(S)) + lead, len(S) - 1); want = A0 + (S[fut] - S0[fut])
        clipped = want.copy()
        for s in r.arms:
            i = r.sidx[s]; alo = np.minimum(r.lo[s], A0[:, i].min(0)); ahi = np.maximum(r.hi[s], A0[:, i].max(0))
            clipped[:, i] = np.clip(want[:, i], alo, ahi)
        bad = np.abs(A - want) > 2e-6
        explained = bool(np.all(np.abs(A - clipped)[bad] <= 2e-6))
        joints = sorted({int(j) for j in np.argwhere(bad)[:, 1]})
        rows.append({**{k: m[k] for k in ("episode", "variation", "family", "max_err_rad", "frames_over_tol")}, "explained_by_action_clip": explained, "joints": joints})
    out[repo] = {"mismatch_episodes": len(rows), "all_explained_by_clip": all(x["explained_by_action_clip"] for x in rows), "rows": rows}
    print(repo, len(rows), "episodes; all explained by the action clip:", out[repo]["all_explained_by_clip"],
          "| joints involved:", sorted({j for x in rows for j in x["joints"]}), flush=True)
(H / "label_cause.json").write_text(json.dumps(out, indent=1))
