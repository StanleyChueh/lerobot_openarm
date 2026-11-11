"""Side-by-side comparison video of a built synthetic dataset: real source episode (left) vs synthetic episode (right),
body camera, with a status bar: EDITED (scene differs from the recording) or SAME AS REAL (recorded motion)."""
from __future__ import annotations
import json
import av, cv2, numpy as np, pandas as pd


def _label(img, text, sub, col):
    bar = np.full((52, img.shape[1], 3), 18, np.uint8)
    cv2.putText(bar, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
    cv2.putText(bar, sub, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def run(ctx, out, episodes=None, speed=1, log=print):
    root = ctx.work / "dataset"; prov = {e["export_episode_index"]: e for e in json.loads((root / "meta/sfa_provenance.json").read_text())["episodes"]}
    df = pd.read_parquet(sorted((root / "data").rglob("*.parquet")), columns=["episode_index", "frame_index", "wrist_view_valid"])
    epi = df.episode_index.to_numpy(); fi = df.frame_index.to_numpy(); valid = np.stack(df.wrist_view_valid.to_numpy())[:, 0] > 0.5
    want = set(prov) if episodes is None else set(episodes); cam = ctx.src.body
    W, H = ctx.cam.W, ctx.cam.H
    c = av.open(str(out), "w"); vs = c.add_stream("libx264", rate=ctx.src.fps); vs.width, vs.height = 2 * W, H + 52; vs.pix_fmt = "yuv420p"; vs.options = {"crf": "20"}
    cur, R, k = None, None, 0; g = 0; render_info = {}
    for p in sorted((root / f"videos/{cam}").rglob("*.mp4")):
        with av.open(str(p)) as f:
            for fr in f.decode(video=0):
                n = int(epi[g]); t = int(fi[g]); ok = valid[g]; g += 1
                if n not in want or t % speed:
                    continue
                e = prov[n]; E = e["source_real_episode"]
                if n not in render_info:
                    render_info[n]=json.loads((ctx.work/f"body/{E}_{e['variation']}.json").read_text())
                ok=bool(render_info[n][t].get('copied',False))
                if E != cur:
                    R = ctx.src.decode(cam, E); cur = E
                sh = e.get("shift_m") or [0, 0]
                source_t=int(e.get("source_frame_index",list(range(len(R))))[t])
                left = _label(R[source_t], f"REAL  source episode {E}", f"source frame {source_t}/{len(R) - 1}", (255, 255, 255))
                right = _label(fr.to_ndarray(format="rgb24"), f"SYNTHETIC  episode {n}  {e['variation']}",
                               f"frame {t}  {e['family'].replace('_', ' ')} ({100 * sh[0]:+.0f}, {100 * sh[1]:+.0f}) cm  |  " + ("SAME AS REAL (recorded motion)" if ok else "EDITED"),
                               (120, 200, 255) if ok else (255, 170, 60))
                for pkt in vs.encode(av.VideoFrame.from_ndarray(np.concatenate([left, right], 1), format="rgb24")):
                    c.mux(pkt)
                k += 1
    for pkt in vs.encode():
        c.mux(pkt)
    c.close(); log(out, k, "frames", round(k / ctx.src.fps, 1), "s")
