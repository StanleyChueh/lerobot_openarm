"""Export (LeRobot v3.0): synthetic-only dataset of every passing+rendered candidate, and optionally a mixed dataset
(source real episodes first, original videos copied, + synthetic) with a per-frame `is_synthetic` flag."""
from __future__ import annotations
import hashlib, json, shutil
import numpy as np
from .generate import load_candidate
from .recon import Recon
from .render_stage import unchanged_rows


def _features(ctx):
    inf = ctx.src.info["features"]; cams = [ctx.src.body] + ctx.src.wrists; nw = len(ctx.src.wrists)
    f = {k: {kk: inf[k][kk] for kk in ("dtype", "shape", "names") if kk in inf[k]} for k in ["observation.state", "action"] + cams}
    for c in cams:
        f[c]["dtype"] = "video"
    f["wrist_view_valid"] = {"dtype": "float32", "shape": (nw,), "names": ctx.src.wrists}
    f["wrist_view_pose_offset"] = {"dtype": "float32", "shape": (2 * nw,), "names": [f"{w}_{q}" for w in ctx.src.wrists for q in ("trans_m", "rot_rad")]}
    return f


def run(ctx, out_repo, device="cuda:0", mixed_repo=None, log=print):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out = ctx.work / "dataset"
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=out_repo, fps=ctx.src.fps, features=_features(ctx), root=out,
                               robot_type=ctx.src.info.get("robot_type"), use_videos=True, streaming_encoding=True)
    wa = ctx.cfg["source"].get("wrist_camera_arms", {}); eps = []; n = 0
    for e in ctx.episodes():
        p = ctx.work / f"candidates/ep{e}/checks.json"
        if not p.exists():
            continue
        checks = json.loads(p.read_text()); keys = [k for k, v in sorted(checks.items()) if v["pass"] and (ctx.work / f"body/{e}_{k}.npy").exists()]
        if not keys:
            continue
        rec = Recon(ctx, e, device); ep = ctx.src.load(e)
        for key in keys:
            c = load_candidate(ctx, rec, key); ck = checks[key]
            body = np.load(ctx.work / f"body/{e}_{key}.npy", mmap_mode="r"); same = unchanged_rows(rec, c)
            off = np.zeros((rec.T, 2 * len(ctx.src.wrists)), np.float32)
            for i, w in enumerate(ctx.src.wrists):
                arm = wa.get(w)
                if arm:
                    Tn = ctx.robot.tcp_from_state(c["state"], arm); Ts = ctx.robot.tcp_from_state(rec.state, arm)
                    off[:, 2 * i] = np.linalg.norm(Tn[:, :3, 3] - Ts[:, :3, 3], axis=1)
                    off[:, 2 * i + 1] = np.arccos(np.clip((np.einsum("nij,nij->n", Tn[:, :3, :3], Ts[:, :3, :3]) - 1) / 2, -1, 1))
            st, ac = c["state"].astype(np.float32), c["action"].astype(np.float32)
            for t in range(rec.T):
                fr = {"observation.state": st[t], "action": ac[t], ctx.src.body: np.ascontiguousarray(body[t]),
                      "wrist_view_valid": np.full(len(ctx.src.wrists), float(same[t]), np.float32), "wrist_view_pose_offset": off[t], "task": ctx.src.task}
                for w in ctx.src.wrists:
                    fr[w] = ep.frames[w][t]
                ds.add_frame(fr)
            ds.save_episode()
            eps.append({"export_episode_index": n, "synthetic": True, "source_dataset": ctx.cfg["source"].get("repo_id"), "source_revision": ctx.cfg["source"].get("revision"),
                        "source_real_episode": e, "variation": key, "family": ck["family"], "object": ck.get("object"), "shift_m": ck["shift_m"],
                        "n_frames": rec.T, "frames_edited": int((~same).sum()), "t_rejoin": ck["t_rejoin"],
                        "state_sha256": hashlib.sha256(st.tobytes()).hexdigest(), "action_sha256": hashlib.sha256(ac.tobytes()).hexdigest(),
                        "replay_checks": {k: v for k, v in ck.items() if k not in ("marks",)}, "schedule": ck["marks"]})
            log(n, e, key); n += 1
    ds.finalize()
    prov = {"name": out_repo, "task": ctx.src.task, "contains_real_episodes": False, "physics_used": False, "generator": "sfagen",
            "config": ctx.cfg, "camera_calibration": ctx.calib, "table_height_m": ctx.table_height, "episodes": eps}
    (out / "meta/sfa_provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    log("built", ds.meta.total_episodes, ds.meta.total_frames)
    if mixed_repo:
        build_mixed(ctx, out_repo, mixed_repo, log)
    return out


def build_mixed(ctx, syn_repo, mixed_repo, log=print):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import add_features, merge_datasets
    W = ctx.work / "mixed_tmp"; shutil.rmtree(W, ignore_errors=True); W.mkdir()
    real = LeRobotDataset(ctx.cfg["source"]["repo_id"], root=ctx.src.root, episodes=ctx.episodes())
    syn = LeRobotDataset(syn_repo, root=ctx.work / "dataset"); nr, ns = real.meta.total_frames, syn.meta.total_frames
    if set(ctx.episodes()) != set(ctx.src.episode_ids):
        raise NotImplementedError("mixed export currently requires all source episodes to be used")
    f1 = {"dtype": "float32", "shape": [1], "names": ["is_synthetic"]}
    r2 = add_features(real, {"wrist_view_valid": (np.ones((nr, len(ctx.src.wrists)), np.float32), syn.meta.features["wrist_view_valid"]),
                             "wrist_view_pose_offset": (np.zeros((nr, 2 * len(ctx.src.wrists)), np.float32), syn.meta.features["wrist_view_pose_offset"]),
                             "is_synthetic": (np.zeros((nr, 1), np.float32), f1)}, output_dir=W / "real", repo_id="tmp/real")
    s2 = add_features(syn, {"is_synthetic": (np.ones((ns, 1), np.float32), f1)}, output_dir=W / "syn", repo_id="tmp/syn")
    out = ctx.work / "dataset_mixed"; shutil.rmtree(out, ignore_errors=True)
    m = merge_datasets([r2, s2], output_repo_id=mixed_repo, output_dir=out, concatenate_videos=False)
    sp = json.loads((ctx.work / "dataset/meta/sfa_provenance.json").read_text())
    nre = real.meta.total_episodes
    sp.update({"name": mixed_repo, "contains_real_episodes": True, "real_episodes": list(range(nre)), "per_frame_flag": "is_synthetic"})
    sp["episodes"] = [{"episode_index": i, "synthetic": False, "source_real_episode": e} for i, e in enumerate(ctx.src.episode_ids)] + \
                     [{**e, "episode_index": nre + e["export_episode_index"]} for e in sp["episodes"]]
    (out / "meta/sfa_provenance.json").write_text(json.dumps(sp, indent=1, default=float)); shutil.rmtree(W)
    log("mixed", m.meta.total_episodes, m.meta.total_frames)
