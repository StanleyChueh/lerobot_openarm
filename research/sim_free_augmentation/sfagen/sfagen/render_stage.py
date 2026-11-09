"""Render every passing candidate of one episode (body camera). Frames whose scene equals the recording are copied."""
from __future__ import annotations
import json, time
import numpy as np
from .generate import load_candidate
from .render import Renderer, candidate_frames


def unchanged_rows(rec, c):
    same = np.all(np.abs(c["state"] - rec.state) < 1e-9, axis=1)
    for k, P in enumerate(c["objects"]):
        src = np.stack([rec.obj_pose(k, t) for t in range(rec.T)]); same &= np.abs(P - src).reshape(rec.T, -1).max(1) < 1e-9
    return same


def run_episode(ctx, rec, device="cuda:0", log=print, keys=None):
    out = ctx.work / "body"; out.mkdir(exist_ok=True)
    checks = json.loads((ctx.work / f"candidates/ep{rec.e}/checks.json").read_text())
    todo = [k for k, v in checks.items() if v["pass"] and (keys is None or k in keys) and not (out / f"{rec.e}_{k}.npy").exists()]
    if not todo:
        return 0
    R = Renderer(rec, device, ctx.cfg.get("render", {})); n_rendered = 0
    for key in todo:
        t0 = time.time(); c = load_candidate(ctx, rec, key); same = unchanged_rows(rec, c); frames, infos = [], []
        for (i, q, g, objs) in candidate_frames(rec, c):
            ts = i + rec.lag; a, b = int(np.clip(np.floor(ts), 0, rec.T - 1)), int(np.clip(np.ceil(ts), 0, rec.T - 1))
            if same[a] and same[b]:
                frames.append(rec.F[i]); infos.append({"copied": True}); continue
            img, info = R.render(i, q, g, objs); frames.append(img); infos.append(info); n_rendered += 1
        np.save(out / f"{rec.e}_{key}.npy", np.stack(frames)); (out / f"{rec.e}_{key}.json").write_text(json.dumps(infos))
        log(rec.e, key, rec.T, "frames", sum(1 for x in infos if not x.get("copied")), "rendered", round(time.time() - t0, 1), "s")
    return n_rendered
