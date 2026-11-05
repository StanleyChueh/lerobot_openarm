"""Full-episode original vs generated three-camera videos (with wrist provenance row) from the caches."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
from gen_wrist import CACHE, V3, panel
from sfa3.recon3 import data
from sfa.render import write_mp4
out = HERE / "outputs/episodes"
for E in (0, 3, 8, 5):
    ep = data.load(E)
    res = json.loads((V3 / f"outputs/ep{E}/results.json").read_text())
    keys = sorted(p.name[len("wristR_"):-4] for p in (CACHE / f"ep{E}").glob("wristR_*.npy"))
    for key in [k for k in keys if k != "identity__C__s1.0"][:2]:
        body = np.load(V3 / f"outputs/cache/ep{E}/raw/{res['render_alias'][key]}.npy", mmap_mode="r")
        lab = np.load(CACHE / f"ep{E}" / f"wristlab_{key}.npz")
        gen = {s: (np.load(CACHE / f"ep{E}" / f"wrist{s}_{key}.npy", mmap_mode="r"), lab[s]) for s in "RL"}
        vid = [panel(ep.frames, gen, None, body, t, t, E, key) for t in range(len(ep.state))]
        (out / f"ep{E}").mkdir(parents=True, exist_ok=True)
        write_mp4(out / f"ep{E}" / f"{key}_full_3cam.mp4", vid, crf=24)
        print(E, key, flush=True)
