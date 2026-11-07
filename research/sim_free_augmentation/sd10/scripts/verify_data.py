"""STEP 1 - pin and verify the 50-episode pool and the original-ten dataset; test duplication/provenance.

Read-only. Network use is limited to Hub metadata reads (HfApi.dataset_info) unless --offline.

    env -u PYTHONPATH -u LD_LIBRARY_PATH ../../../.venv/bin/python scripts/verify_data.py [--offline]

Writes outputs/data_verification.json and outputs/duplication.json.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sd10 import common as C  # noqa: E402


def git_blob_sha1(p: Path) -> str:
    b = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def hub_files(repo: str, rev: str, snap: Path, extra, offline: bool) -> dict:
    out = {"repo": repo, "pinned_revision": rev, "files": []}
    if offline:
        out["hub"] = "offline: local hashes only"
        sib = [{"rfilename": str(p.relative_to(snap)), "lfs": None, "blob_id": None, "size": None}
               for p in sorted(snap.rglob("*")) if p.is_file() or p.is_symlink()]
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info(repo, revision=rev, files_metadata=True)
        out["hub_main_sha_now"] = api.dataset_info(repo).sha
        out["hub_revision_sha"] = info.sha
        out["main_equals_pinned"] = out["hub_main_sha_now"] == rev
        sib = [{"rfilename": s.rfilename, "lfs": (s.lfs.sha256 if s.lfs else None), "blob_id": s.blob_id, "size": s.size}
               for s in info.siblings]
    for s in sib:
        p = C.resolve(snap, s["rfilename"], extra)
        rec = {"path": s["rfilename"], "local": str(p) if p else None, "hub_lfs_sha256": s["lfs"],
               "hub_blob_id": s["blob_id"], "hub_size": s["size"]}
        if p is not None:
            rec["bytes"] = p.stat().st_size
            if s["lfs"]:
                rec["sha256"] = C.sha256(p)
                rec["match"] = rec["sha256"] == s["lfs"]
            else:
                rec["sha256"] = C.sha256(p)
                rec["git_blob_sha1"] = git_blob_sha1(p)
                rec["match"] = (s["blob_id"] is None) or rec["git_blob_sha1"] == s["blob_id"]
        else:
            rec["match"] = None   # not present locally
        out["files"].append(rec)
    return out


def tabular_checks(ds: C.Dataset) -> dict:
    info = ds.info
    feats = info["features"]
    res = {"total_episodes": info["total_episodes"], "total_frames": info["total_frames"], "fps": info["fps"],
           "robot_type": info["robot_type"], "codebase_version": info["codebase_version"],
           "cameras": [k for k, v in feats.items() if v["dtype"] == "video"],
           "camera_shapes": {k: v["shape"] for k, v in feats.items() if v["dtype"] == "video"},
           "camera_codecs": {k: v.get("info", {}).get("video.codec") for k, v in feats.items() if v["dtype"] == "video"},
           "state_dim": feats["observation.state"]["shape"], "action_dim": feats["action"]["shape"],
           "action_names": feats["action"]["names"], "state_names": feats["observation.state"].get("names"),
           "episodes_in_meta": len(ds.ep), "rows_in_data": len(ds.df), "per_episode": {}}
    g = 0
    for e in ds.episodes:
        r = ds.rows(e)
        n = int(ds.ep.loc[e]["length"])
        S, A = ds.vec(e, "observation.state"), ds.vec(e, "action")
        fi, ts = r["frame_index"].to_numpy(), r["timestamp"].to_numpy().astype(float)
        chk = {"length": n, "rows": len(r),
               "frame_index_contiguous": bool((fi == np.arange(n)).all()),
               "timestamp_regular": bool(np.abs(ts - np.arange(n) / C.FPS).max() < 1e-3),
               "timestamp_max_err_s": float(np.abs(ts - np.arange(n) / C.FPS).max()),
               "episode_index_ok": bool((r["episode_index"].to_numpy() == e).all()),
               "global_index_ok": bool((r["index"].to_numpy() == np.arange(g, g + n)).all()),
               "state_shape": list(S.shape), "action_shape": list(A.shape),
               "finite": bool(np.isfinite(S).all() and np.isfinite(A).all()),
               "state_sha256": C.array_sha256(S), "action_sha256": C.array_sha256(A)}
        for cam in res["cameras"]:
            _, t0, t1 = ds.video(cam, e)
            chk[f"{cam}_window_frames"] = round((t1 - t0) * C.FPS, 3)
        chk["ok"] = all(chk[k] for k in ("frame_index_contiguous", "timestamp_regular", "episode_index_ok",
                                          "global_index_ok", "finite")) and chk["rows"] == n \
            and S.shape[1] == 16 and A.shape[1] == 16 \
            and all(abs(chk[f"{c}_window_frames"] - n) < 0.01 for c in res["cameras"])
        res["per_episode"][e] = chk
        g += n
    res["sum_lengths"] = g
    res["all_ok"] = all(v["ok"] for v in res["per_episode"].values()) and g == info["total_frames"] == len(ds.df)
    return res


def packet_scan(path: str) -> tuple[list[float], list[str]]:
    """Demux only (no pixel decoding): PTS seconds and SHA-256 of each packet's compressed bytes."""
    import av
    pts, hs = [], []
    with av.open(path) as c:
        st = c.streams.video[0]
        for pk in c.demux(st):
            if pk.pts is None or pk.size == 0:
                continue
            pts.append(float(pk.pts * st.time_base))
            hs.append(hashlib.sha256(bytes(pk)).hexdigest())
    o = np.argsort(pts, kind="stable")
    return [pts[i] for i in o], [hs[i] for i in o]


def packet_checks(ds: C.Dataset) -> dict:
    out = {}
    for cam in C.CAMS:
        path = ds.video(cam, ds.episodes[0])[0]
        pts, hs = packet_scan(str(path))
        pts = np.array(pts)
        out[cam] = {}
        for e in ds.episodes:
            _, t0, t1 = ds.video(cam, e)
            m = (pts >= t0 - 0.5 / C.FPS) & (pts < t1 - 0.5 / C.FPS)
            p = pts[m]
            d = np.diff(p) * C.FPS if len(p) > 1 else np.array([1.0])
            h = hashlib.sha256("".join(np.array(hs)[m]).encode()).hexdigest()
            n = int(ds.ep.loc[e]["length"])
            out[cam][e] = {"packets": int(m.sum()), "expected": n, "count_ok": int(m.sum()) == n,
                           "pts_step_min": float(d.min()), "pts_step_max": float(d.max()),
                           "pts_regular": bool(np.abs(d - 1).max() < 0.05), "packet_stream_sha256": h}
    return out


def _pixel_job(args):
    which, cam, e = args
    ds = C.pool() if which == "pool" else C.ten()
    F = ds.decode(cam, e)
    g = F.reshape(len(F), -1)
    same = (np.abs(np.diff(g.astype(np.int16), axis=0)).max(1) == 0)
    mean = F.mean(axis=(1, 2, 3))
    small = F[:, ::4, ::4].astype(np.float32).mean(-1)
    lap = [float(np.var(np.diff(s, 2, axis=0)) + np.var(np.diff(s, 2, axis=1))) for s in small]
    return which, cam, e, {"decoded": len(F), "frozen_consecutive": int(same.sum()), "longest_frozen_run": _run(same),
                           "black_frames": int((mean < 8).sum()), "white_frames": int((mean > 247).sum()),
                           "mean_intensity": float(mean.mean()), "sharpness_median": float(np.median(lap)),
                           "sharpness_p10": float(np.percentile(lap, 10))}


def _run(b):
    best = cur = 0
    for x in b:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def pixel_checks(workers: int) -> dict:
    pdsd, tds = C.pool(), C.ten()
    jobs = [("pool", cam, e) for cam in C.CAMS for e in pdsd.episodes if not (cam != C.BODY and e in C.V4A_LOCKED)]
    jobs += [("ten", cam, e) for cam in C.CAMS for e in tds.episodes]
    out = {"pool": {c: {} for c in C.CAMS}, "ten": {c: {} for c in C.CAMS}}
    with ProcessPoolExecutor(workers) as ex:
        for which, cam, e, r in ex.map(_pixel_job, jobs):
            out[which][cam][e] = r
    for e in C.V4A_LOCKED:
        for cam in (C.LEFT_WRIST, C.RIGHT_WRIST):
            out["pool"][cam][e] = {"decoded": None, "note": "v4A whole-stream exclusion: pixels not decoded (packet/PTS checks only)"}
    return out


def _frame_diff_job(args):
    cam, i, j = args
    a = C.ten().decode(cam, i).astype(np.int16)
    b = C.pool().decode(cam, j).astype(np.int16)
    if len(a) != len(b):
        return cam, i, j, {"len_ten": len(a), "len_pool": len(b)}
    d = np.abs(a - b).reshape(len(a), -1)
    return cam, i, j, {"frames": len(a), "max_abs_px": int(d.max()), "mean_abs_px": float(d.mean()),
                       "identical_frames": int((d.max(1) == 0).sum())}


def duplication(tab_pool, tab_ten, pk_pool, pk_ten, workers: int) -> dict:
    pdsd, tds = C.pool(), C.ten()
    out = {"ten_vs_pool": {}, "pool_internal": {}}
    # state/action equality, all 10 x 50 pairs
    Sp = {e: pdsd.vec(e, "observation.state") for e in pdsd.episodes}
    Ap = {e: pdsd.vec(e, "action") for e in pdsd.episodes}
    match = {}
    for i in tds.episodes:
        S, A = tds.vec(i, "observation.state"), tds.vec(i, "action")
        cands = []
        for j in pdsd.episodes:
            if len(Sp[j]) != len(S):
                continue
            cands.append({"pool": j, "state_max_abs": float(np.abs(Sp[j] - S).max()),
                          "action_max_abs": float(np.abs(Ap[j] - A).max())})
        exact = [c["pool"] for c in cands if c["state_max_abs"] == 0 and c["action_max_abs"] == 0]
        hash_eq = [j for j in pdsd.episodes if tab_pool["per_episode"][j]["state_sha256"] == tab_ten["per_episode"][i]["state_sha256"]
                   and tab_pool["per_episode"][j]["action_sha256"] == tab_ten["per_episode"][i]["action_sha256"]]
        pk_eq = {cam: [j for j in pdsd.episodes if pk_pool[cam][j]["packet_stream_sha256"] == pk_ten[cam][i]["packet_stream_sha256"]]
                 for cam in C.CAMS}
        out["ten_vs_pool"][i] = {"length": len(S), "same_length_pool_episodes": [c["pool"] for c in cands],
                                 "exact_state_action_matches": exact, "hash_matches": hash_eq,
                                 "closest_same_length": sorted(cands, key=lambda c: c["state_max_abs"])[:2],
                                 "compressed_video_packet_matches": pk_eq}
        match[i] = exact[0] if len(exact) == 1 else None
    # decoded-pixel comparison for each matched pair, all three cameras (pool eps 0-9 are not v4A-locked)
    jobs = [(cam, i, j) for i, j in match.items() if j is not None and j not in C.V4A_LOCKED for cam in C.CAMS]
    with ProcessPoolExecutor(workers) as ex:
        for cam, i, j, r in ex.map(_frame_diff_job, jobs):
            out["ten_vs_pool"][i].setdefault("decoded_pixels", {})[cam] = r
    out["mapping_ten_to_pool"] = match
    # pool-internal duplicates: exact hashes and nearest trajectories (state resampled to 100 points)
    hs = {}
    for e in pdsd.episodes:
        hs.setdefault((tab_pool["per_episode"][e]["state_sha256"], tab_pool["per_episode"][e]["action_sha256"]), []).append(e)
    out["pool_internal"]["exact_duplicate_groups"] = [v for v in hs.values() if len(v) > 1]
    R = {e: np.stack([np.interp(np.linspace(0, len(s) - 1, 100), np.arange(len(s)), s[:, k]) for k in range(16)], 1)
         for e, s in Sp.items()}
    eps = pdsd.episodes
    D = np.array([[np.sqrt(((R[a] - R[b]) ** 2).mean()) if a != b else np.inf for b in eps] for a in eps])
    nn = {e: {"nearest": int(eps[int(D[k].argmin())]), "rms_rad": float(D[k].min())} for k, e in enumerate(eps)}
    out["pool_internal"]["nearest_trajectory"] = nn
    out["pool_internal"]["min_pair_rms_rad"] = float(D.min())
    out["pool_internal"]["median_nn_rms_rad"] = float(np.median(D.min(1)))
    pk_groups = {}
    for cam in C.CAMS:
        g = {}
        for e in eps:
            g.setdefault(pk_pool[cam][e]["packet_stream_sha256"], []).append(e)
        pk_groups[cam] = [v for v in g.values() if len(v) > 1]
    out["pool_internal"]["identical_compressed_video_groups"] = pk_groups
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    pdsd, tds = C.pool(), C.ten()
    rep = {"hub": {"pool": hub_files(C.POOL_REPO, C.POOL_REV, C.POOL_SNAP, C.POOL_EXTRA_DIRS, a.offline),
                   "ten": hub_files(C.TEN_REPO, C.TEN_REV, C.TEN_SNAP, (), a.offline)}}
    for k, v in rep["hub"].items():
        v["all_present_match"] = all(f["match"] for f in v["files"] if f["match"] is not None)
        v["missing_local"] = [f["path"] for f in v["files"] if f["local"] is None]
        print(k, "hub files", len(v["files"]), "match", v["all_present_match"], "missing", v["missing_local"], flush=True)
    rep["tabular"] = {"pool": tabular_checks(pdsd), "ten": tabular_checks(tds)}
    print("tabular ok", {k: v["all_ok"] for k, v in rep["tabular"].items()}, flush=True)
    rep["packets"] = {"pool": packet_checks(pdsd), "ten": packet_checks(tds)}
    for k, v in rep["packets"].items():
        print("packets", k, {c: all(r["count_ok"] and r["pts_regular"] for r in v[c].values()) for c in v}, flush=True)
    rep["pixels"] = pixel_checks(a.workers)
    print("pixels decoded", flush=True)
    dup = duplication(rep["tabular"]["pool"], rep["tabular"]["ten"], rep["packets"]["pool"], rep["packets"]["ten"], a.workers)
    C.dump(dup, C.OUT / "duplication.json")
    print("ten->pool mapping", dup["mapping_ten_to_pool"], flush=True)
    C.dump(rep, C.OUT / "data_verification.json")


if __name__ == "__main__":
    main()
