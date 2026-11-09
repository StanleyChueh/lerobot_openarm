"""Render every passing battery-v01 candidate (body_cam): MGW forward renderer on the calibrated reconstruction,
battery config (part-aware fill) + PLATE_ADAPT. Families: 'bat' battery-position (target/bat2/candidates),
'st' station-position (target/candidates_station). Frames after the rejoin equal the recording and are copied."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; T = HERE.parent / "target"
sys.path.insert(0, str(T))
import srender, srecon  # noqa: E402
OUT = HERE / "work/body"
FAM = {"bat": (T / "bat2/candidates", "battery"), "st": (T / "candidates_station", "station")}
ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 else list(FAM)
eps = [int(x) for x in sys.argv[1].split(",")]; dev = sys.argv[2]; pa = int(sys.argv[3])
srender.mgw.MGWForward.PLATE_ADAPT = pa
for E in eps:
    for fam, (cdir, obj) in ((f, FAM[f]) for f in ONLY):
        cp = cdir / f"ep{E}/checks.json"
        if not cp.exists():
            print("missing", fam, E, flush=True); continue
        checks = json.loads(cp.read_text())
        todo = [k for k, v in checks.items() if v["pass"] and not (OUT / f"{E}_{k}.npy").exists()]
        if not todo:
            continue
        rec = srecon.StationRecon(E, device=dev) if obj == "station" else srecon.BRecon2(E, device=dev)
        R = srender.renderer(rec, dev, obj)
        for key in todo:
            t0 = time.time(); z = np.load(cdir / f"ep{E}/{key}.npz"); vid = srender.mgw.candidate_video(rec, z)
            tj = checks[key]["marks"]["t_rejoin"]; frames, infos = [], []
            for i in range(rec.T):
                j, q, g, can = vid[i]
                if i > tj + 3 and j == i:
                    frames.append(rec.F[j]); infos.append({"source_frame": j, "copied": True}); continue
                img, info = R.render(j, q, g, can); frames.append(img); infos.append({"source_frame": j, **info})
            np.save(OUT / f"{E}_{key}.npy", np.stack(frames)); (OUT / f"{E}_{key}.json").write_text(json.dumps(infos))
            print(E, key, rec.T, round(time.time() - t0, 1), "s", flush=True)
        del R, rec; torch.cuda.empty_cache()
