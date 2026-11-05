"""Temporal consistency of generated wrist videos: RAFT flow-warp error (half resolution, FB-consistent pixels) of
generated vs the source wrist video on the same frames (lower = smoother)."""
import json, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
from gen_wrist import CACHE
from sfa3.recon3 import data
from sfa32 import wristcam as wc
from sfa2 import metrics2 as M2
from sfa import track as trk
half = lambda F: np.stack([cv2.resize(np.asarray(f), (320, 240), interpolation=cv2.INTER_AREA) for f in F])
out = {}
for E in (0, 3, 5, 8):
    ep = data.load(E)
    keys = sorted(p.name[len("wristR_"):-4] for p in (CACHE / f"ep{E}").glob("wristR_*.npy") if "identity" not in p.name)[:3]
    for key in keys:
        for s in "RL":
            g = half(np.load(CACHE / f"ep{E}" / f"wrist{s}_{key}.npy", mmap_mode="r"))
            src = half(ep.frames[wc.CAM_OF[s]])
            r = {}
            for nm, F in (("generated", g), ("source", src)):
                fl = trk.video_flows(F, "cuda:1")
                r[nm] = M2.warp_error_regions(F, fl, {"all": np.ones(F.shape[:3], bool)})["all"]["mean"]
            out[f"ep{E}/{key}/{s}"] = r
            print(E, key, s, {k: round(v, 3) for k, v in r.items()}, flush=True)
agg = {s: {k: float(np.mean([v[k] for kk, v in out.items() if kk.endswith("/" + s)])) for k in ("generated", "source")} for s in "RL"}
(HERE / "outputs/wrist_temporal.json").write_text(json.dumps({"per_clip": out, "mean": agg}, indent=1))
print(agg)
