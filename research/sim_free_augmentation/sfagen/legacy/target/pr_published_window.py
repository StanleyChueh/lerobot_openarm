"""Same render metrics (pr_handover.metrics) on the PUBLISHED v3 episodes: reference distribution for the render gate."""
import json, sys, types
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
src = (HERE / "pr_handover.py").read_text().rsplit("\nmain()", 1)[0]
m = types.ModuleType("prh"); m.__file__ = str(HERE / "pr_handover.py"); sys.argv = [sys.argv[0], sys.argv[1]]
exec(compile(src, "pr_handover", "exec"), m.__dict__)
sel = json.loads((ROOT / "3cam_v3/selection.json").read_text())["selected"]
out = {}
for E in sorted({e["E"] for e in sel}):
    s = m.data.load(E, cams=()); evs = m.data.command_events(s.action)
    rec = m.recon3.build(E, with_plate=True, log=lambda *a: None); R = m.mgw.MGWForward(rec, device=sys.argv[1])
    for e in [x for x in sel if x["E"] == E]:
        a, b = e["rows"]; z = np.load(m.V31 / f"ep{E}/candidates/{e['key']}.npz"); vid = m.mgw.candidate_video(rec, z)
        frames = np.load(ROOT / f"3cam_v3/work/body/{E}_{e['key']}.npy", mmap_mode="r")
        out[f"{E}_{e['key']}"] = m.metrics(R, rec, vid, frames, a, b, (evs["left_close"] - 30 + a, evs["right_open"] + 20 + a))
        print(E, e["key"], out[f"{E}_{e['key']}"], flush=True)
    (HERE / "pr_published_window.json").write_text(json.dumps(out, indent=1))
    del R, rec; torch.cuda.empty_cache()
