"""Export (LeRobot v3.0): synthetic-only dataset of every passing+rendered candidate, and optionally a mixed dataset
(source real episodes first, original videos copied, + synthetic) with a per-frame `is_synthetic` flag."""
from __future__ import annotations
import hashlib, json, shutil
import numpy as np
from .generate import load_candidate
from .recon import Recon
from .render_stage import unchanged_rows,render_fingerprint
from .timing import source_frame_map,time_coordinates,source_time_at
from .recon import interp_rows


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
        raise FileExistsError(f"Refusing to replace an existing export: {out}; choose a new work directory")
    ds = LeRobotDataset.create(repo_id=out_repo, fps=ctx.src.fps, features=_features(ctx), root=out,
                               robot_type=ctx.src.info.get("robot_type"), use_videos=True, streaming_encoding=False, image_writer_threads=4, encoder_threads=2)
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
            from .checks import nominal_action_error
            if nominal_action_error(rec,c)>2e-6:raise ValueError(f'Action labels do not match planned states: {e}/{key}; regenerate without clipping')
            body = np.load(ctx.work / f"body/{e}_{key}.npy", mmap_mode="r"); same = unchanged_rows(rec, c)
            N=len(c['state']);tau=time_coordinates(rec,c)
            if body.shape!=(N,rec.H,rec.W,3):raise ValueError(f'Wrong rendered shape for {e}/{key}: {body.shape}')
            fingerprint=ctx.work/f'body/{e}_{key}.fingerprint'
            if not fingerprint.exists() or fingerprint.read_text()!=render_fingerprint(ctx,rec,key):raise ValueError(f'Stale rendering: {e}/{key}')
            off = np.zeros((N, 2 * len(ctx.src.wrists)), np.float32)
            valid=np.zeros((N,len(ctx.src.wrists)),np.float32);wrist_maps={}

            wr=ctx.cfg.get('wrist_render',{}).get('cameras',[]);wrist_img={}
            if wr:
                from .wrist import load_spec
                from .render_stage import wrist_fingerprint
                wspec=load_spec(ctx)
            for i,w in enumerate(ctx.src.wrists):
                # Body-camera lag is not an independently calibrated wrist-camera lag.
                lag=float(wspec[w]['lag']) if w in wr else float(ctx.cfg['source'].get('wrist_camera_lags',{}).get(w,0.))
                jm=source_frame_map(rec,c,lag);wrist_maps[w]=jm
                x=np.clip(np.arange(N)+lag,0,N-1);a=np.floor(x).astype(int);b=np.ceil(x).astype(int)
                exact=np.abs(source_time_at(rec,c,np.arange(N)+lag)-lag-jm)<1e-8
                valid[:,i]=(same[a]&same[b]&exact).astype(np.float32)
                if w in wr:   # rendered wrist view (camera passed the wrist gate): valid if copied or <= wrist_valid_max_inpaint
                    stem=ctx.work/f"wrist/{e}_{key}_{w.split('.')[-1]}"
                    if not stem.with_suffix('.fingerprint').exists() or stem.with_suffix('.fingerprint').read_text()!=wrist_fingerprint(ctx,rec,key,w,wspec[w]):raise ValueError(f'Stale wrist rendering: {stem}')
                    wrist_img[w]=np.load(stem.with_suffix('.npy'),mmap_mode='r');winfo=json.loads(stem.with_suffix('.json').read_text())
                    if wrist_img[w].shape!=(N,)+ep.frames[w].shape[1:] or [x['source_frame'] for x in winfo]!=jm.tolist():raise ValueError(f'Wrist render does not match the export clock: {stem}')
                    thr=ctx.cfg['wrist_render'].get('valid_max_inpaint',0.05)
                    valid[:,i]=np.array([x.get('copied',False) or x.get('inpainted_frac',1.)<=thr for x in winfo],np.float32)
                arm=wa.get(w)
                if arm:
                    generated=interp_rows(c['state'],np.arange(N)+lag)
                    reference=interp_rows(rec.state,jm+lag)
                    Tn=ctx.robot.tcp_from_state(generated,arm);Ts=ctx.robot.tcp_from_state(reference,arm)
                    off[:,2*i]=np.linalg.norm(Tn[:,:3,3]-Ts[:,:3,3],axis=1)
                    off[:,2*i+1]=np.arccos(np.clip((np.einsum('nij,nij->n',Tn[:,:3,:3],Ts[:,:3,:3])-1)/2,-1,1))
            st, ac = c["state"].astype(np.float32), c["action"].astype(np.float32)
            for t in range(N):
                fr = {"observation.state": st[t], "action": ac[t], ctx.src.body: np.ascontiguousarray(body[t]),
                      "wrist_view_valid": valid[t], "wrist_view_pose_offset": off[t], "task": ctx.src.task}
                for w in ctx.src.wrists:
                    fr[w] = np.ascontiguousarray(wrist_img[w][t]) if w in wrist_img else ep.frames[w][wrist_maps[w][t]]
                ds.add_frame(fr)
            ds.save_episode()
            eps.append({"export_episode_index": n, "synthetic": True, "source_dataset": ctx.cfg["source"].get("repo_id"), "source_revision": ctx.cfg["source"].get("revision"),
                        "source_real_episode": e, "variation": key, "family": ck["family"], "object": ck.get("object"), "shift_m": ck["shift_m"],
                        "render_fingerprint":fingerprint.read_text(),"n_frames": N,"source_frames":rec.T,"frames_added":N-rec.T,"time_scale":ck['marks'].get('time_scale',1.),
                        "source_state_time":tau.tolist(),"source_frame_index":source_frame_map(rec,c).tolist(),
                        "wrist_source_frame_index":{w:j.tolist() for w,j in wrist_maps.items()},
                        "wrist_rendering":{w:("sfagen.wrist re-rendered (moving-camera MGW); valid = copied or inpainted <= %g"%ctx.cfg.get('wrist_render',{}).get('valid_max_inpaint',0.05) if w in wrist_img else "time-remapped source RGB; changed scenes are explicitly invalid, not generated wrist views") for w in ctx.src.wrists},
                        "wrist_lag_status":{w:('configured' if w in ctx.cfg['source'].get('wrist_camera_lags',{}) else 'assumed zero; unvalidated') for w in ctx.src.wrists}, "frames_edited": int((~same).sum()), "t_rejoin": ck["t_rejoin"],
                        "state_sha256": hashlib.sha256(st.tobytes()).hexdigest(), "action_sha256": hashlib.sha256(ac.tobytes()).hexdigest(),
                        "replay_checks": {k: v for k, v in ck.items() if k not in ("marks",)}, "schedule": ck["marks"]})
            log(n, e, key); n += 1
    if not eps:raise ValueError('No passing, current rendered candidates were available for export')
    ds.finalize()
    prov = {"name": out_repo, "task": ctx.src.task, "contains_real_episodes": False, "physics_used": False, "generator": "sfagen",
            "validation_scope":{"state_action_labels":"nominal kinematic consistency; not measured execution","body_rgb":"pixel warping; fidelity not independently validated","wrist_rgb":"re-rendered for wrist_render.cameras (geometry + depth warping, see per-episode wrist_rendering), else copied source frames; mask invalid wrist_view_valid entries","robot_success":"not tested","force_contact":"not validated"},
            "config": ctx.cfg, "camera_calibration": ctx.calib, "table_height_m": ctx.table_height, "episodes": eps}
    (out / "meta/sfa_provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    from .verify import video_span_report
    spans=video_span_report(out,[ctx.src.body]+ctx.src.wrists,ctx.src.fps)
    (out/'meta/video_span_verification.json').write_text(json.dumps(spans,indent=1))
    if not spans['pass']:raise ValueError('Export has missing, duplicated or misassigned video frames; inspect video_span_verification.json')
    log("built", ds.meta.total_episodes, ds.meta.total_frames)
    if mixed_repo:
        build_mixed(ctx, out_repo, mixed_repo, log)
    return out


def build_mixed(ctx, syn_repo, mixed_repo, log=print):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import add_features, merge_datasets
    W=ctx.work/'mixed_tmp'
    if W.exists() or (ctx.work/'dataset_mixed').exists():raise FileExistsError('Mixed output/temp already exists; choose a new work directory')
    W.mkdir()
    real = LeRobotDataset(ctx.cfg["source"]["repo_id"], root=ctx.src.root, episodes=ctx.episodes())
    syn = LeRobotDataset(syn_repo, root=ctx.work / "dataset"); nr, ns = real.meta.total_frames, syn.meta.total_frames
    if set(ctx.episodes()) != set(ctx.src.episode_ids):
        raise NotImplementedError("mixed export currently requires all source episodes to be used")
    f1 = {"dtype": "float32", "shape": [1], "names": ["is_synthetic"]}
    r2 = add_features(real, {"wrist_view_valid": (np.ones((nr, len(ctx.src.wrists)), np.float32), syn.meta.features["wrist_view_valid"]),
                             "wrist_view_pose_offset": (np.zeros((nr, 2 * len(ctx.src.wrists)), np.float32), syn.meta.features["wrist_view_pose_offset"]),
                             "is_synthetic": (np.zeros((nr, 1), np.float32), f1)}, output_dir=W / "real", repo_id="tmp/real")
    s2 = add_features(syn, {"is_synthetic": (np.ones((ns, 1), np.float32), f1)}, output_dir=W / "syn", repo_id="tmp/syn")
    out = ctx.work / "dataset_mixed"
    m = merge_datasets([r2, s2], output_repo_id=mixed_repo, output_dir=out, concatenate_videos=False)
    sp = json.loads((ctx.work / "dataset/meta/sfa_provenance.json").read_text())
    nre = real.meta.total_episodes
    sp.update({"name": mixed_repo, "contains_real_episodes": True, "real_episodes": list(range(nre)), "per_frame_flag": "is_synthetic"})
    sp["episodes"] = [{"episode_index": i, "synthetic": False, "source_real_episode": e} for i, e in enumerate(ctx.src.episode_ids)] + \
                     [{**e, "episode_index": nre + e["export_episode_index"]} for e in sp["episodes"]]
    (out / "meta/sfa_provenance.json").write_text(json.dumps(sp, indent=1, default=float)); shutil.rmtree(W)
    from .verify import video_span_report
    spans=video_span_report(out,[ctx.src.body]+ctx.src.wrists,ctx.src.fps)
    (out/'meta/video_span_verification.json').write_text(json.dumps(spans,indent=1))
    if not spans['pass']:raise ValueError('Mixed export camera spans failed verification')
    log("mixed", m.meta.total_episodes, m.meta.total_frames)
