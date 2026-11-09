"""Per-frame choice between the v3 render (cached) and the MGW render using the validated no-reference score
(render_v4/proxy.py; agrees with the ground-truth PSNR winner 80 % on the benchmark). Decisions are smoothed with a
9-frame majority filter to avoid flicker. Writes work/body_final/{E}_{key}.npy + per-frame choice/scores."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "render_v4"))
import mgw, proxy
from sfa3 import recon3
V31 = mgw.SFA / "v3_1/outputs"
OUT = HERE / "work/body_final"; OUT.mkdir(parents=True, exist_ok=True)
eps = [int(x) for x in sys.argv[1].split(",")]; dev = sys.argv[2]
sel = json.loads((HERE / "selection.json").read_text())["selected"]
for E in eps:
    todo = [e for e in sel if e["E"] == E and not (OUT / f"{E}_{e['key']}.npy").exists()]
    if not todo:
        continue
    rec = recon3.build(E, with_plate=True, log=lambda *a: None)
    R = mgw.MGWForward(rec, device=dev)
    for e in todo:
        a, b = e["rows"]
        z = np.load(V31 / f"ep{E}/candidates/{e['key']}.npz"); vid = mgw.candidate_video(rec, z)
        new = np.load(HERE / f"work/body/{E}_{e['key']}.npy"); old = np.load(V31 / f"cache/ep{E}/raw/{e['alias']}.npy", mmap_mode="r")
        sv3, smg = [], []
        for k, i in enumerate(range(a, b)):
            j, q, g, can = vid[i]
            _, lab, _ = R.zbuf(R.placements(q, g, can)); lab = lab.cpu().numpy()
            tm = (lab >= 0) & (R.part_side[np.clip(lab, 0, None)] != "B")
            mv = (rec.masks["R"][j] > 0) | (rec.masks["L"][j] > 0) | (rec.masks["can"][j] > 0)
            sv3.append(proxy.score(np.asarray(old[i]), rec.plate, tm, mv)); smg.append(proxy.score(new[k], rec.plate, tm, mv))
        sv3, smg = np.array(sv3), np.array(smg)
        use_mgw = uniform_filter1d((smg >= sv3).astype(float), 9, mode="nearest") >= 0.5
        final = np.where(use_mgw[:, None, None, None], new, np.asarray(old[a:b]))
        np.save(OUT / f"{E}_{e['key']}.npy", final)
        (OUT / f"{E}_{e['key']}.json").write_text(json.dumps({"use_mgw": use_mgw.astype(int).tolist(), "score_v3": sv3.tolist(), "score_mgw": smg.tolist()}))
        print(E, e["key"], "mgw frames", round(float(use_mgw.mean()), 3), "score v3", round(float(sv3.mean()), 3), "mgw", round(float(smg.mean()), 3), flush=True)
