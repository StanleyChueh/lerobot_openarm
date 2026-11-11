"""Verify built datasets: loads with LeRobot, label SHA-256 = planned candidate, no synthetic episode equals a real one,
every camera decodes without black frames, body camera matches the rendered frames, mixed: is_synthetic flags."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import av, numpy as np, pandas as pd


def _frames(root, cam):
    for p in sorted((Path(root) / f"videos/{cam}").rglob("*.mp4")):
        with av.open(str(p)) as f:
            for fr in f.decode(video=0):
                yield fr.to_ndarray(format="rgb24")


def video_span_report(root,cameras,fps):
    """Count and align actual PTS in EACH episode/camera span, including boundaries and trailing frames."""
    root=Path(root);eps=pd.read_parquet(sorted((root/'meta/episodes').rglob('*.parquet')));rows=[]
    for cam in cameras:
        prefix=f'videos/{cam}/';pts={}
        for path in sorted((root/f'videos/{cam}').rglob('*.mp4')):
            times=[];black=0
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    times.append(float(frame.pts*frame.time_base));black+=int(frame.to_ndarray(format='rgb24').max()==0)
            pts[str(path.resolve())]=(np.asarray(times),black)
        used={key:np.zeros(len(values[0]),int) for key,values in pts.items()}
        for _,e in eps.iterrows():
            a=float(e[prefix+'from_timestamp']);b=float(e[prefix+'to_timestamp']);n=int(e['length'])
            path=root/f"videos/{cam}/chunk-{int(e[prefix+'chunk_index']):03d}/file-{int(e[prefix+'file_index']):03d}.mp4"
            key=str(path.resolve());times,black=pts.get(key,(np.zeros(0),0));sel=(times>=a-1e-6)&(times<b-1e-6)
            if key in used:used[key][sel]+=1
            indices=np.round((times[sel]-a)*fps).astype(int)
            aligned=len(indices)==n and np.array_equal(indices,np.arange(n)) and np.max(np.abs((times[sel]-a)*fps-indices),initial=0)<1e-3
            ok=abs((b-a)*fps-n)<1e-3 and aligned and black==0
            rows.append({'camera':cam,'episode':int(e['episode_index']),'length':n,'metadata_span_frames':(b-a)*fps,'decoded_span_frames':int(sel.sum()),'pts_aligned':bool(aligned),'pass':bool(ok)})
        for key,count in used.items():
            if not np.all(count==1):rows.append({'camera':cam,'file':key,'unassigned_or_multiply_assigned_frames':int(np.sum(count!=1)),'pass':False})
    return {'pass':bool(rows) and all(r['pass'] for r in rows),'spans':rows}


def run(ctx, log=print):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out = {}
    root = ctx.work / "dataset"; prov = json.loads((root / "meta/sfa_provenance.json").read_text()); eps = prov["episodes"]
    ds = LeRobotDataset(prov["name"], root=root,video_backend="pyav"); df = pd.read_parquet(sorted((root / "data").rglob("*.parquet")))
    epi = df.episode_index.to_numpy(); S = np.stack(df["observation.state"].to_numpy()).astype(np.float32); A = np.stack(df["action"].to_numpy()).astype(np.float32)
    real = {e: ctx.src.arrays(e)[0].astype(np.float32) for e in ctx.episodes()}
    rep = {"episodes": len(eps), "frames": int(len(df)), "loads_in_lerobot": ds.meta.total_episodes == len(eps),
           "labels_sha_ok": all(hashlib.sha256(S[epi == e["export_episode_index"]].tobytes()).hexdigest() == e["state_sha256"] and
                                hashlib.sha256(A[epi == e["export_episode_index"]].tobytes()).hexdigest() == e["action_sha256"] for e in eps),
           "episodes_identical_to_a_real_episode": int(sum(any(len(r) == (epi == e["export_episode_index"]).sum() and np.array_equal(S[epi == e["export_episode_index"]], r) for r in real.values()) for e in eps)),
           "sources_used": sorted({e["source_real_episode"] for e in eps}), "families": {f: sum(e["family"] == f for e in eps) for f in {e["family"] for e in eps}},
           "all_replay_checks_pass": all(e["replay_checks"]["pass"] for e in eps)}
    order = [(e["source_real_episode"], e["variation"]) for e in eps]
    for cam in [ctx.src.body] + ctx.src.wrists:
        n, mn, se = 0, 255, []
        ref = (fr for s_, k in order for fr in np.load(ctx.work / f"body/{s_}_{k}.npy", mmap_mode="r")) if cam == ctx.src.body else None
        for i, f in enumerate(_frames(root, cam)):
            if ref is not None:
                r = next(ref)
                if i % 5 == 0:
                    se.append(float(((f.astype(np.float32) - r) ** 2).mean()))
            mn = min(mn, int(f.max())); n += 1
        rep[cam] = {"decoded": n, "any_all_black_frame": mn == 0, **({"psnr_vs_rendered_db": round(float(10 * np.log10(255 ** 2 / np.mean(se))), 2)} if se else {})}
    rep['video_spans']=video_span_report(root,[ctx.src.body]+ctx.src.wrists,ctx.src.fps)
    sample_indices=[];offset=0
    for episode in eps:
        n=episode['n_frames'];sample_indices.extend([offset,offset+n//2,offset+n-1]);offset+=n
    for i in sorted(set(sample_indices)):
        item=ds[i]
        np.testing.assert_array_equal(item['observation.state'].numpy(),S[i]);np.testing.assert_array_equal(item['action'].numpy(),A[i])
        for cam in [ctx.src.body]+ctx.src.wrists:
            h,w=ctx.src.info['features'][cam]['shape'][:2]
            assert tuple(item[cam].shape)==(3,h,w),(i,cam,item[cam].shape)
            assert np.isfinite(item[cam].numpy()).all()
    rep['lerobot_getitem_boundary_midpoint_samples']=len(set(sample_indices))
    out["synthetic"] = rep
    mroot = ctx.work / "dataset_mixed"
    if mroot.exists():
        mp = json.loads((mroot / "meta/sfa_provenance.json").read_text()); md = pd.read_parquet(sorted((mroot / "data").rglob("*.parquet")))
        flag = np.stack(md["is_synthetic"].to_numpy()).reshape(-1); me = md.episode_index.to_numpy(); nre = len(mp["real_episodes"])
        LeRobotDataset(mp["name"], root=mroot)
        out["mixed"] = {"episodes": int(me.max() + 1), "frames": int(len(md)), "real_episodes": nre,
                        "is_synthetic_flag_correct": bool(np.all(flag == (me >= nre))),
                        "real_labels_identical_to_source": all(np.array_equal(np.stack(md[me == i]["observation.state"].to_numpy()).astype(np.float32), real[e]) for i, e in enumerate(ctx.src.episode_ids))}
    (ctx.work / "verification.json").write_text(json.dumps(out, indent=1)); log(json.dumps(out, indent=1))
    passed=rep['loads_in_lerobot'] and rep['labels_sha_ok'] and rep['all_replay_checks_pass'] and rep['video_spans']['pass'] and rep['episodes_identical_to_a_real_episode']==0
    for cam in [ctx.src.body]+ctx.src.wrists:passed &= rep[cam]['decoded']==len(df) and not rep[cam]['any_all_black_frame']
    if not passed:raise ValueError('Dataset verification failed; inspect verification.json')
    return out
