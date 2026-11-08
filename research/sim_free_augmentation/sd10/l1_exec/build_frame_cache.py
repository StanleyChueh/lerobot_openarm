"""Decode native pool body RGB (PyAV rgb24) for all 23 training-source captures and the 834 locked evaluation inputs.

Frame index convention identical to learning_v1/freeze_inputs.body_frames: k = round((pts_s - episode_t0) * 30).
Every evaluation frame is checked against the locked manifest RGB hash, PTS and time base. Cache is regenerable
and not committed. Writes outputs/frame_cache/*.npy and outputs/frame_cache_report.json.
"""
from __future__ import annotations

import json

import av
import numpy as np

import l1lib as L

PT, C = L.PT, L.C


def decode_episode(ds, e, want=None):
    p, t0, _ = ds.video(C.BODY, e)
    T = len(ds.rows(e))
    want = set(range(T)) if want is None else set(want)
    out, meta = {}, {}
    with av.open(str(p)) as cont:
        st = cont.streams.video[0]
        cont.seek(int(max(0, t0 - .5) / st.time_base), stream=st)
        for fr in cont.decode(st):
            if fr.time is None or fr.time < t0 - .5 / 30:
                continue
            k = int(round((fr.time - t0) * 30))
            if k > max(want):
                break
            if k in want and k not in out:
                rgb = fr.to_ndarray(format="rgb24")
                assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
                out[k] = rgb
                meta[k] = {"pts": int(fr.pts), "time_base_numerator": int(st.time_base.numerator),
                           "time_base_denominator": int(st.time_base.denominator), "native_pts_s": float(fr.time)}
    assert set(out) == want, (e, sorted(want - set(out))[:5])
    return out, meta, str(p)


def main():
    L.CACHE.mkdir(parents=True, exist_ok=True)
    ds = C.pool()
    rep = {"train": {}, "eval": {}}
    for e in L.TRAIN_EPS:
        fr, meta, path = decode_episode(ds, e)
        T = len(ds.rows(e))
        arr = np.stack([fr[k] for k in range(T)])
        np.save(L.cache_path(e), arr)
        rep["train"][str(e)] = {"frames": T, "video": path, "content_sha256": PT.array_sha(arr)}
        print(f"train ep{e}: {T} frames", flush=True)
    man = json.loads((L.L1 / "locked_evaluation_manifest.json").read_text())
    z = np.load(L.L1 / "evaluation_arrays.npz", allow_pickle=False)
    items = z["items"]
    ev = np.zeros((len(items), 480, 640, 3), np.uint8)
    mism = []
    for e in L.EVAL:
        idx = np.flatnonzero(items[:, 0] == e)
        fr, meta, _ = decode_episode(ds, e, set(int(items[i, 1]) for i in idx))
        for i in idx:
            t = int(items[i, 1]); en = man["entries"][i]
            assert en["episode"] == e and en["input_frame"] == t
            got = {"rgb_sha256": PT.array_sha(fr[t]), **meta[t]}
            if got != en["native_video"]:
                mism.append(int(i))
            ev[i] = fr[t]
        rep["eval"][str(e)] = len(idx)
        print(f"eval ep{e}: {len(idx)} inputs", flush=True)
    assert not mism, f"locked RGB hash/PTS mismatch at items {mism[:10]}"
    np.save(L.CACHE / "eval_frames.npy", ev)
    rep["eval_items"] = int(len(items))
    rep["eval_rgb_pts_timebase_all_match_manifest"] = True
    rep["eval_frames_content_sha256"] = PT.array_sha(ev)
    (L.OUT / "frame_cache_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"train_eps": len(rep["train"]), "eval_items": len(items), "rgb_match": True}))


if __name__ == "__main__":
    main()
