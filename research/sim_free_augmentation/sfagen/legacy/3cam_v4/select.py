"""v4 selection = the 55 v3 episodes (unchanged trajectories) + hand-over-LOCATION variants (frozen v3.1 condition C,
grasp-relative SE(3) coordination of both arms) that pass (1) the same replay checks and (2) the render gate: per
episode and inside the hand-over window, label-image agreement ('placed') >= the 10th percentile and ghost residue
<= the 90th percentile of the published v3 episodes (target/pr_handover.py, target/pr_published_window.py)."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; EXP = HERE.parent
V31 = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation/v3_1/outputs")
v3 = json.loads((EXP / "3cam_v3/selection.json").read_text())["selected"]
for e in v3:
    e["body_src"] = str(EXP / f"3cam_v3/work/body/{e['E']}_{e['key']}.npy")
h = json.loads((EXP / "target/pr_handover.json").read_text()); p = json.loads((EXP / "target/pr_published_window.json").read_text())
ref = {(w, m): [v[w][m] for v in p.values() if v[w][m] is not None] for w in ("all", "handover") for m in ("placed", "ghost")}
gate = {f"{w}_{m}": float(np.percentile(x, 10 if m == "placed" else 90)) for (w, m), x in ref.items()}
man = {(e["E"], e["key"]): e for e in json.loads((V31 / "dataset_export/C_all_manifest.json").read_text())}
add, rej = [], []
for k, v in h.items():
    E = int(k.split("_")[0]); key = k[len(str(E)) + 1:]; e = dict(man[(E, key)])
    r = v["render"]; why = []
    if not v["pass"]:
        why.append("replay checks")
    for w in ("all", "handover"):
        if r[w]["placed"] < gate[f"{w}_placed"]:
            why.append(f"{w} placed {r[w]['placed']:.3f} < {gate[f'{w}_placed']:.3f}")
        if r[w]["ghost"] is not None and r[w]["ghost"] > gate[f"{w}_ghost"]:
            why.append(f"{w} ghost {r[w]['ghost']:.3f} > {gate[f'{w}_ghost']:.3f}")
    e.update({"replay": {kk: v[kk] for kk in v if kk not in ("render", "pass")}, "render_gate": r, "handover_shift_mm": v["handover_tcp_shift_mm"],
              "left_tcp_dev_mm": None, "body_src": str(EXP / f"target/pr_handover/{E}_{key}.npy"), "reject": why})
    (rej if why else add).append(e)
print("v3 kept", len(v3), "hand-over added", len(add), "rejected", [(e["E"], e["key"], e["reject"]) for e in rej])
(HERE / "selection.json").write_text(json.dumps({"gate": gate, "selected": v3 + add, "rejected_handover": rej}, indent=1, default=float))
