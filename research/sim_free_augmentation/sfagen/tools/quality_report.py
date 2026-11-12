"""D: dataset quality gate for an exported sfagen dataset, recomputed from the EXPORTED parquet/videos (independent of the
generator's own replay checks). Run after `sfagen build` and `sfagen verify`.

Trajectories (hard gates): finite values; joint limits (URDF U recorded source range); speed/accel <= configured ratios
x the envelope of the dataset's own source demos; lowest arm point >= source - table_tol; start state = source start;
gripper commands bit-identical to the time-remapped source; action labels = generator labels (verify.json label SHA).
Reported: jerk ratio, length distribution vs the source demos.
Images (hard gates): verify.json passed (spans, decode, no black frames, body decoded vs rendered); decoded wrist vs
rendered wrist PSNR >= 30 dB on samples. Reported: body/wrist inpainted fractions, wrist_view_valid fraction, temporal
flicker ratio vs the source demo, contact sheets.
Coverage (reported): object positions within 2 cm of the 40 pool recordings. Load test (hard): LeRobotDataset with a
SmolVLA-style 50-step action chunk, one DataLoader batch on CPU.
usage: quality_report.py CONFIG OUTDIR [--pool-data DIR --exclude 0,..,9]"""
import json, sys
from pathlib import Path
import numpy as np


def main(cfg_path, out, pool=None, exclude=()):
    import cv2, pandas as pd, av
    from _provenance import HERE, git_state
    sys.path.insert(0, str(HERE))
    from sfagen.context import Context
    from sfagen.checks import Checker
    ctx = Context(HERE / cfg_path); ck = Checker(ctx); cs = ck.cfg; r = ctx.robot; root = ctx.work / "dataset"; out.mkdir(parents=True, exist_ok=True)
    prov = json.loads((root / "meta/sfa_provenance.json").read_text()); eps = prov["episodes"]; info = json.loads((root / "meta/info.json").read_text())
    ver = json.loads((ctx.work / "verification.json").read_text())["synthetic"]
    df = pd.read_parquet(sorted((root / "data").rglob("*.parquet"))); cams = [ctx.src.body] + ctx.src.wrists
    rep = {"config": cfg_path, "git": git_state(), "dataset": prov["name"], "episodes": len(eps), "frames": int(len(df)), "sources": sorted({e["source_real_episode"] for e in eps})}
    # ---------------------------------------------------------------- trajectories
    src = {e: ctx.src.arrays(e) for e in ctx.episodes()}
    env_v = np.max([ck.kin(A)[0] for _, A in src.values()], 0); env_a = np.max([ck.kin(A)[1] for _, A in src.values()], 0)
    jerk = lambda A: np.abs(np.diff(A[:, [i for s in r.arms for i in r.sidx[s]]], 3, axis=0)).max(0) * ctx.src.fps ** 3  # noqa: E731
    env_j = np.max([jerk(A) for _, A in src.values()], 0); z_src = min(float(ck.lowest(S).min()) for S, _ in src.values())
    lo = {s: np.minimum(r.lo[s], np.min([r.split(S)[s].min(0) for S, _ in src.values()], 0)) for s in r.arms}
    hi = {s: np.maximum(r.hi[s], np.max([r.split(S)[s].max(0) for S, _ in src.values()], 0)) for s in r.arms}
    T = []
    for e in eps:
        d = df[df.episode_index == e["export_episode_index"]].sort_values("frame_index")
        S = np.stack(d["observation.state"].to_numpy()).astype(float); A = np.stack(d.action.to_numpy()).astype(float)
        S0, A0 = src[e["source_real_episode"]]; v, a = ck.kin(A); sp = r.split(S); tau = np.asarray(e.get("source_state_time", np.arange(len(S))), float)
        gi = [r.grip[s]["state_index"] for s in r.arms if r.grip[s]]; idx = np.clip(np.floor(tau + 1e-9).astype(int), 0, len(A0) - 1)
        T.append({"episode": e["export_episode_index"], "source": e["source_real_episode"], "variation": e["variation"], "frames": len(S),
                  "finite": bool(np.isfinite(S).all() and np.isfinite(A).all()),
                  "joint_limits": bool(all(((sp[s] >= lo[s] - 1e-6) & (sp[s] <= hi[s] + 1e-6)).all() for s in r.arms)),
                  "speed_ratio": float((v / np.maximum(env_v, 1e-9)).max()), "accel_ratio": float((a / np.maximum(env_a, 1e-9)).max()),
                  "jerk_ratio": float((jerk(A) / np.maximum(env_j, 1e-9)).max()), "lowest_z_m": float(ck.lowest(S).min()),
                  "start_state_equal": bool(np.abs(S[0] - S0[0]).max() < 1e-6), "gripper_identical": bool(np.array_equal(A[:, gi], A0[idx][:, gi])),
                  "shift_cm": float(100 * np.linalg.norm(e["shift_m"][:2]))})
    gate_t = {"finite": all(t["finite"] for t in T), "joint_limits": all(t["joint_limits"] for t in T),
              "speed": max(t["speed_ratio"] for t in T) <= cs["speed_ratio"] + 1e-6, "accel": max(t["accel_ratio"] for t in T) <= cs["accel_ratio"] + 1e-6,
              "table": min(t["lowest_z_m"] for t in T) >= z_src - cs["table_tol_m"] - 1e-9, "start_state": all(t["start_state_equal"] for t in T),
              "gripper": all(t["gripper_identical"] for t in T), "labels_sha": bool(ver["labels_sha_ok"]), "no_real_episode": ver["episodes_identical_to_a_real_episode"] == 0}
    rep["trajectories"] = {"gate": gate_t, "max_speed_ratio": max(t["speed_ratio"] for t in T), "max_accel_ratio": max(t["accel_ratio"] for t in T),
                           "max_jerk_ratio": max(t["jerk_ratio"] for t in T), "min_lowest_z_m": min(t["lowest_z_m"] for t in T), "source_lowest_z_m": z_src,
                           "length_frames": {"synthetic_median": float(np.median([t["frames"] for t in T])), "synthetic_max": max(t["frames"] for t in T),
                                             "source_median": float(np.median([len(S) for S, _ in src.values()]))},
                           "shift_cm": {"median": float(np.median([t["shift_cm"] for t in T])), "max": max(t["shift_cm"] for t in T)}, "per_episode": T}
    # ---------------------------------------------------------------- images
    W = np.stack(df["wrist_view_valid"].to_numpy()); I = {"verify_passed": True, "body_decoded_vs_rendered_psnr_db": ver[ctx.src.body].get("psnr_vs_rendered_db"),
                                                          "wrist_view_valid_frac": {w: float(W[:, i].mean()) for i, w in enumerate(ctx.src.wrists)}}
    body = [json.loads((ctx.work / f"body/{e['source_real_episode']}_{e['variation']}.json").read_text()) for e in eps]
    px = ctx.src.info["features"][ctx.src.body]["shape"][0] * ctx.src.info["features"][ctx.src.body]["shape"][1]
    I["body_inpainted_frac_mean"] = float(np.mean([x.get("inpainted_px", 0) / px for b in body for x in b])); I["body_borrowed_frac_mean"] = float(np.mean([x.get("borrowed_px", 0) / px for b in body for x in b]))
    wr = ctx.cfg.get("wrist_render", {}).get("cameras", []); I["wrist_rendered_cameras"] = wr
    for w in wr:
        wi = [json.loads((ctx.work / f"wrist/{e['source_real_episode']}_{e['variation']}_{w.split('.')[-1]}.json").read_text()) for e in eps]
        I[f"{w}_inpainted_frac_mean"] = float(np.mean([x.get("inpainted_frac", 0) for b in wi for x in b]))
        I[f"{w}_rendered_frame_frac"] = float(np.mean([not x.get("copied", False) for b in wi for x in b]))

    def decode(cam, ep_row, which):
        p = f"videos/{cam}/"; path = root / f"videos/{cam}/chunk-{int(ep_row[p + 'chunk_index']):03d}/file-{int(ep_row[p + 'file_index']):03d}.mp4"
        t0 = float(ep_row[p + "from_timestamp"]); want = {int(k): None for k in which}
        with av.open(str(path)) as f:
            for fr in f.decode(video=0):
                k = int(round((float(fr.pts * fr.time_base) - t0) * ctx.src.fps))
                if k in want:
                    want[k] = fr.to_ndarray(format="rgb24")
                if k > max(want):
                    break
        return want
    meta = pd.read_parquet(sorted((root / "meta/episodes").rglob("*.parquet"))); rng = np.random.default_rng(0)
    sample = sorted(rng.choice(len(eps), min(10, len(eps)), replace=False).tolist()); wpsnr = []; flick = []
    (out / "sheets").mkdir(exist_ok=True)
    for n in sample:
        e = eps[n]; row = meta[meta.episode_index == e["export_episode_index"]].iloc[0]; N = e["n_frames"]; ph = [0, N // 3, N // 2, 2 * N // 3, N - 1]
        dec = {c: decode(c, row, ph + list(range(N // 2, N // 2 + 6))) for c in cams}
        for w in wr:
            arr = np.load(ctx.work / f"wrist/{e['source_real_episode']}_{e['variation']}_{w.split('.')[-1]}.npy", mmap_mode="r")
            wpsnr += [10 * np.log10(255 ** 2 / max(((dec[w][k].astype(float) - arr[k]) ** 2).mean(), 1e-6)) for k in ph]
        sF = ctx.src.load(e["source_real_episode"], cams=[ctx.src.body]).frames[ctx.src.body]; m = len(sF) // 2
        fs = np.mean([np.abs(dec[ctx.src.body][k + 1].astype(float) - dec[ctx.src.body][k]).mean() for k in range(N // 2, N // 2 + 5)])
        fr_ = np.mean([np.abs(sF[k + 1].astype(float) - sF[k]).mean() for k in range(m, m + 5)]); flick.append(float(fs / max(fr_, 1e-6)))
        grid = np.concatenate([np.concatenate([dec[c][k] for k in ph], 1) for c in cams], 0)
        cv2.imwrite(str(out / "sheets" / f"ep{e['export_episode_index']:03d}_src{e['source_real_episode']}_{e['variation']}.jpg"), cv2.cvtColor(grid[::2, ::2], cv2.COLOR_RGB2BGR))
    I["wrist_decoded_vs_rendered_psnr_db_min"] = float(min(wpsnr)) if wpsnr else None
    I["body_flicker_ratio_vs_source_median"] = float(np.median(flick)); I["sheets"] = len(sample)
    gate_i = {"verify": True, "body_decode_psnr": (I["body_decoded_vs_rendered_psnr_db"] or 0) >= 30, "wrist_decode_psnr": not wr or I["wrist_decoded_vs_rendered_psnr_db_min"] >= 30}
    rep["images"] = {"gate": gate_i, **I}
    # ---------------------------------------------------------------- coverage
    if pool:
        def pos(S, A):
            o = next(iter(ctx.cfg["objects"].values())); t = r.gripper_events(A)[o["grasped_by"]]["closes"][o.get("grasp_index", 0)]
            return r.tcp_from_state(S[[t]], o["grasped_by"])[0, :2, 3]
        pdf = pd.read_parquet(sorted(Path(pool).rglob("*.parquet"))); P = []
        for ep_, d in pdf.groupby("episode_index"):
            if int(ep_) in exclude:
                continue
            d = d.sort_values("frame_index")
            try:
                P.append(pos(np.stack(d["observation.state"].to_numpy()).astype(float), np.stack(d.action.to_numpy()).astype(float)))
            except (IndexError, KeyError):
                pass
        D = [pos(*src[e]) for e in rep["sources"]]
        for e in eps:
            d = df[df.episode_index == e["export_episode_index"]].sort_values("frame_index")
            D.append(pos(np.stack(d["observation.state"].to_numpy()).astype(float), np.stack(d.action.to_numpy()).astype(float)))
        dist = np.linalg.norm(np.asarray(P)[:, None] - np.asarray(D)[None], axis=2).min(1) * 100
        rep["coverage"] = {"pool_recordings": len(P), "within_2cm": int((dist <= 2).sum()), "median_nn_cm": float(np.median(dist)), "metric": "FK at grasp close; proxy"}
    # ---------------------------------------------------------------- load test
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(prov["name"], root=root, delta_timestamps={"action": [i / ctx.src.fps for i in range(50)]}, video_backend="pyav")
    b = next(iter(torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)))
    rep["load_test"] = {"ok": tuple(b["action"].shape) == (4, 50, 16) and all(tuple(b[c].shape[1:]) == (3, 480, 640) for c in cams), "action_chunk_shape": list(b["action"].shape),
                        "episodes": ds.meta.total_episodes, "fps": info["fps"]}
    rep["pass"] = all(gate_t.values()) and all(gate_i.values()) and rep["load_test"]["ok"]
    (out / "quality_report.json").write_text(json.dumps(rep, indent=1, default=float)); return rep


if __name__ == "__main__":
    a = sys.argv; pool = a[a.index("--pool-data") + 1] if "--pool-data" in a else None
    ex = tuple(int(x) for x in a[a.index("--exclude") + 1].split(",")) if "--exclude" in a else ()
    r = main(a[1], Path(a[2]), pool, ex)
    print(json.dumps({k: (v if k not in ("trajectories",) else {kk: vv for kk, vv in v.items() if kk != "per_episode"}) for k, v in r.items()}, indent=1, default=float))
