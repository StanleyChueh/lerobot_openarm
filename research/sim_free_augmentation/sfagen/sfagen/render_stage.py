"""Render passing candidates; source-copy only exact unchanged scenes on the continuous camera clock."""
from __future__ import annotations
import hashlib,json,time
import numpy as np
from .generate import load_candidate
from .render import Renderer,candidate_frames_tw
from .timing import source_time_at,source_frame_map,time_coordinates
from .recon import interp_rows


def unchanged_rows(rec,c):
    tau=time_coordinates(rec,c);ref=interp_rows(rec.state,tau)
    same=np.all(np.abs(c['state']-ref)<2e-7,axis=1)
    for k,P in enumerate(c['objects']):
        src=np.stack([rec.obj_pose(k,t) for t in tau]);same &= np.abs(P-src).reshape(len(tau),-1).max(1)<1e-8
    return same


def render_fingerprint(ctx,rec,key):
    h=hashlib.sha256((ctx.work/f'candidates/ep{rec.e}/{key}.npz').read_bytes())
    h.update(json.dumps({'render':ctx.cfg.get('render',{}),'calibration':ctx.calib,'lag':rec.lag},sort_keys=True).encode())
    for name in ('render.py','render_stage.py','timing.py'):
        h.update((__import__('pathlib').Path(__file__).parent/name).read_bytes())
    return h.hexdigest()


def run_episode(ctx,rec,device='cuda:0',log=print,keys=None):
    out=ctx.work/'body';out.mkdir(exist_ok=True)
    checks=json.loads((ctx.work/f'candidates/ep{rec.e}/checks.json').read_text());todo=[]
    for key,ck in checks.items():
        if not ck['pass'] or (keys is not None and key not in keys):continue
        fp=out/f'{rec.e}_{key}.fingerprint';digest=render_fingerprint(ctx,rec,key)
        if (out/f'{rec.e}_{key}.npy').exists() and fp.exists() and fp.read_text()==digest:continue
        todo.append((key,digest))
    if not todo:return 0
    R=Renderer(rec,device,ctx.cfg.get('render',{}));n_rendered=0
    for key,digest in todo:
        t0=time.time();c=load_candidate(ctx,rec,key);same=unchanged_rows(rec,c);frames=[];infos=[];N=len(same)
        for i,j,q,g,objs in candidate_frames_tw(rec,c):
            x=np.clip(i+rec.lag,0,N-1);a=int(np.floor(x));b=int(np.ceil(x))
            source_rgb=float(source_time_at(rec,c,i+rec.lag)-rec.lag)
            exact_scene=same[a] and same[b] and abs(source_rgb-j)<1e-8
            if exact_scene:
                exact_scene=all(np.max(np.abs(q[arm]-rec.gv[arm][j]))<1e-9 for arm in rec.arms)
                exact_scene &= all(abs(g[arm]-rec.grip_v[arm][j])<1e-9 for arm in g)
                exact_scene &= all(np.max(np.abs(T-rec.obj_pose(k,j+rec.lag)))<1e-8 for k,T in enumerate(objs))
            if exact_scene:
                frames.append(rec.F[j]);infos.append({'copied':True,'source_frame':j});continue
            img,info=R.render(j,q,g,objs);frames.append(img);infos.append({**info,'source_frame':j});n_rendered+=1
        np.save(out/f'{rec.e}_{key}.npy',np.stack(frames));(out/f'{rec.e}_{key}.json').write_text(json.dumps(infos))
        (out/f'{rec.e}_{key}.fingerprint').write_text(digest)
        log(rec.e,key,N,'frames',sum(not x.get('copied',False) for x in infos),'rendered',round(time.time()-t0,1),'s')
    return n_rendered


def wrist_fingerprint(ctx,rec,key,cam,spec):
    h=hashlib.sha256((ctx.work/f'candidates/ep{rec.e}/{key}.npz').read_bytes())
    h.update(json.dumps({'camera':cam,'spec':spec,'renderer':ctx.cfg.get('wrist_render',{}).get('renderer',{}),'lag':rec.lag},sort_keys=True,default=float).encode())
    for name in ('wrist.py','render.py','render_stage.py','timing.py','mesh.py'):
        h.update((__import__('pathlib').Path(__file__).parent/name).read_bytes())
    return h.hexdigest()


def run_wrist_episode(ctx,rec,device='cuda:0',log=print,keys=None):
    """Render the wrist cameras listed in wrist_render.cameras (those that passed the wrist gate) for passing candidates."""
    cams=ctx.cfg.get('wrist_render',{}).get('cameras',[])
    if not cams:return 0
    from .wrist import load_spec,WristCam,WristRenderer
    spec=load_spec(ctx);out=ctx.work/'wrist';out.mkdir(exist_ok=True);n=0
    checks=json.loads((ctx.work/f'candidates/ep{rec.e}/checks.json').read_text())
    for cam in cams:
        wc=WristCam(cam,spec[cam],ctx.robot);short=cam.split('.')[-1];todo=[]
        for key,ck in checks.items():
            if not ck['pass'] or (keys is not None and key not in keys):continue
            stem=out/f'{rec.e}_{key}_{short}';digest=wrist_fingerprint(ctx,rec,key,cam,spec[cam])
            if stem.with_suffix('.npy').exists() and stem.with_suffix('.fingerprint').exists() and stem.with_suffix('.fingerprint').read_text()==digest:continue
            todo.append((key,stem,digest))
        if not todo:continue
        R=WristRenderer(rec,ctx.src.load(rec.e,cams=[cam]).frames[cam],wc,ctx.cfg['wrist_render'].get('renderer',{}),device)
        for key,stem,digest in todo:
            t0=time.time();c=load_candidate(ctx,rec,key);frames=[];infos=[]
            for i,j,q,g,objs in candidate_frames_tw(rec,c,wc.lag):
                img,info=R.render(j,q,g,objs);frames.append(img);infos.append({**info,'source_frame':int(j)})
            np.save(stem.with_suffix('.npy'),np.stack(frames));stem.with_suffix('.json').write_text(json.dumps(infos));stem.with_suffix('.fingerprint').write_text(digest);n+=1
            log(rec.e,key,short,len(frames),'frames',sum(not x.get('copied',False) for x in infos),'rendered',round(time.time()-t0,1),'s')
    return n
