"""Render every passing battery candidate (body_cam, all T frames). MGW forward renderer, battery configuration =
Pringles configuration + part-aware hole filling (selected on battery dev episode 1, bbench_dev.json)."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import brender, brecon
brender.mgw.MGWForward.PART_FILL = True
OUT = ROOT / "work/body"; OUT.mkdir(parents=True, exist_ok=True)
eps = [int(x) for x in sys.argv[1].split(",")]; dev = sys.argv[2]
for E in eps:
    cp = ROOT / f"candidates/ep{E}/checks.json"
    if not cp.exists():
        print("no candidates yet", E, flush=True); continue
    checks = json.loads(cp.read_text())
    todo = [k for k, v in checks.items() if v["pass"] and not (OUT / f"{E}_{k}.npy").exists()]
    if not todo:
        continue
    rec = brecon.BRecon(E, device=dev); R = brender.renderer(rec, dev)
    for key in todo:
        t0 = time.time()
        z = np.load(ROOT / f"candidates/ep{E}/{key}.npz"); vid = brender.candidate_video(rec, z)
        tj = checks[key]["marks"]["t_rejoin"]
        frames, infos = [], []
        for i in range(rec.T):
            j, q, g, can = vid[i]
            if i > tj + 3 and j == i:          # after the rejoin the candidate equals the recording
                frames.append(rec.F[j]); infos.append({"source_frame": j, "copied": True}); continue
            img, info = R.render(j, q, g, can); frames.append(img); infos.append({"source_frame": j, **info})
        np.save(OUT / f"{E}_{key}.npy", np.stack(frames)); (OUT / f"{E}_{key}.json").write_text(json.dumps(infos))
        print(E, key, rec.T, round(time.time() - t0, 1), "s", flush=True)
    del R, rec
