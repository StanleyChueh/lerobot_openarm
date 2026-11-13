"""Annotated matched raw-RGB examples; no generated illustration or image retouching."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw,ImageFont
S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S))
from sfagen.context import Context
from sfagen.timing import source_frame_map

ctx=Context(S/'configs/openarm_pringles_v02.yaml');out=S/'results/v02/examples';out.mkdir(exist_ok=True)
allrows=[]
for e in ctx.episodes():
    for k,v in json.loads((ctx.work/f'candidates/ep{e}/checks.json').read_text()).items():
        if v['pass']:allrows.append((e,k,v))
position=sorted([r for r in allrows if r[2]['family']=='object_position'],key=lambda r:(r[2]['marks']['target_xy_m'][0],r[2]['marks']['target_xy_m'][1],r[0]))
chosen=[position[0],next(r for r in allrows if r[2]['family']=='object_position'),
        next(r for r in allrows if r[2]['family']=='interaction_location'),next(r for r in allrows if r[2]['family']=='interaction_timing')]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13);records=[]
for e,key,ck in chosen:
    body=ctx.work/f'body/{e}_{key}.npy';wrist=ctx.work.parent/f'right_wrist/wrist/{e}_{key}_right_wrist_cam.npy'
    if not body.exists() or not wrist.exists():continue
    ep=ctx.src.load(e);T=len(ep.state);z=np.load(ctx.work/f'candidates/ep{e}/{key}.npz');lag=float(np.load(ctx.cache_dir/f'recon_ep{e}.npz',allow_pickle=True)['lag'])
    rec=SimpleNamespace(T=T,lag=lag);candidate={'state':z['state']}
    if 'tau' in z:candidate['tau']=z['tau']
    events=ctx.robot.gripper_events(ep.action);grasp=events['R']['closes'][0];left=next(t for t in events['L']['closes'] if t>grasp);release=next(t for t in events['R']['opens'] if t>left)
    tau=candidate.get('tau',np.arange(T));source_times=np.array([max(0,grasp-15),grasp,left,release,T-1]);times=np.round(np.interp(source_times,tau,np.arange(T))).astype(int)
    bm=source_frame_map(rec,candidate);wm=source_frame_map(rec,candidate,-1.)
    B=np.load(body,mmap_mode='r');W=np.load(wrist,mmap_mode='r');right='observation.images.right_wrist_cam'
    canvas=Image.new('RGB',(150+5*256,95+4*192),'white');d=ImageDraw.Draw(canvas)
    title=f'Source {e} | {key} | {ck["family"]} | raw generated RGB (physical fidelity unknown)';d.text((12,8),title,font=font,fill='#24334a')
    for c,(t,label) in enumerate(zip(times,['Approach','Grasp','Receiver close','Giver opens','End'])):
        x=150+c*256;d.text((x+5,42),label,font=small,fill='#24334a');d.text((x+5,62),f'Output {t/30:.2f}s; source {tau[t]/30:.2f}s',font=small,fill='#24334a')
        imgs=[ep.frames[ctx.src.body][bm[t]],B[t],ep.frames[right][wm[t]],W[t]]
        for row,im in enumerate(imgs):canvas.paste(Image.fromarray(im).resize((256,192),Image.Resampling.LANCZOS),(x,95+row*192))
    for r,label in enumerate(['Real body\nclock-matched','Generated\nbody','Copied right\nwrist','Rendered right\nwrist']):d.multiline_text((8,155+r*192),label,font=font,fill='#24334a',spacing=5)
    name=f'src{e}_{key}.png';canvas.save(out/name)
    records.append(dict(source=e,variation=key,family=ck['family'],output_frames=times.tolist(),source_state_times=tau[times].tolist(),path=f'examples/{name}'))
(out/'manifest.json').write_text(json.dumps(records,indent=2));print('Matched example sheets:',len(records))
