"""Offline 1..N source-budget coverage using only subset-specific replay envelopes.
Positions are FK proxies at recorded command closes, NOT measured object poses or task-success rates.
Rig calibration may use additional recordings; disclose it separately from motion-source count.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial import ConvexHull,QhullError
from sfagen.context import Context
from sfagen.checks import Checker
from sfagen.why import reasons

def main():
 ap=argparse.ArgumentParser();ap.add_argument('config');ap.add_argument('--pool-data',required=True);ap.add_argument('--exclude',required=True);ap.add_argument('--pool-revision',required=True);ap.add_argument('--comparison-work')
 ap.add_argument('--max-shift-cm',type=float,help='keep only candidates whose object moved at most this far (R1 shift limit)')
 ap.add_argument('--out',help='output JSON (default: <work>/source_budget.json)');a=ap.parse_args()
 ctx=Context(a.config);ck=Checker(ctx);r=ctx.robot;E=ctx.episodes();names=list(ctx.cfg['objects']);exclude=set(map(int,a.exclude.split(',')))
 def positions(S,A):
  ev=r.gripper_events(A);out={}
  for name,o in ctx.cfg['objects'].items():
   arm=o['grasped_by'];t=ev[arm]['closes'][o.get('grasp_index',0)];out[name]=r.tcp_from_state(S[[t]],arm)[0,:2,3]
  return out
 src={}
 for e in E:
  S,A=ctx.src.arrays(e);v,acc=ck.kin(A);src[e]={'v':v,'a':acc,'z':float(ck.lowest(S).min()),'pos':positions(S,A)}
 pool={n:[] for n in names};pool_ids=[]
 df=pd.read_parquet(sorted(Path(a.pool_data).rglob('*.parquet')))
 for ep,d in df.groupby('episode_index'):
  if int(ep) in exclude:continue
  d=d.sort_values('frame_index')
  try:pos=positions(np.stack(d['observation.state']).astype(float),np.stack(d.action).astype(float))
  except (IndexError,KeyError):continue
  pool_ids.append(int(ep))
  for n in names:pool[n].append(pos[n])
 pool={n:np.asarray(P) for n,P in pool.items()}
 def candidates(work,legacy=False):
  out=[]
  for e in E:
   mp=work/f'candidates/ep{e}/generation_manifest.json'
   if mp.exists() and not json.loads(mp.read_text())['complete']:raise ValueError(f'Incomplete source episode {e}: {work}')
   if not (work/f'candidates/ep{e}/checks.json').exists():raise FileNotFoundError(f'Missing checks for source episode {e}: {work}')
  for e in E:
   file=work/f'candidates/ep{e}/checks.json'
   if not file.exists():continue
   for key,c in json.loads(file.read_text()).items():
    if c.get('family') not in ('object_workspace','object_position') or 'error' in c:continue
    # Historical dense checks keep their original thresholds; do not rewrite their outcome.
    cfg=({k:v for k,v in ctx.cfg.get('checks',{}).items() if k!='hand_floor_m'} if legacy else ctx.cfg.get('checks',{}))
    failed=set(reasons(c,cfg,c.get('real_lowest_arm_z_m',0)))
    if failed-{'speed','accel','table'}:continue
    z=np.load(work/f'candidates/ep{e}/{key}.npz');v,acc=ck.kin(z['action'].astype(float))
    label_error=0.
    if legacy:
     sourceS,sourceA=ctx.src.arrays(e);future=np.minimum(np.arange(len(sourceS))+c['lead'],len(sourceS)-1);arm=ctx.cfg['objects'][c['object']]['grasped_by'];cols=r.sidx[arm]
     expected=sourceA.copy();expected[:,cols]+=(z['state'].astype(float)-sourceS)[future][:,cols];label_error=float(np.abs(expected-z['action'].astype(float)).max())
    out.append({'legacy_label_error':label_error,'e':e,'key':key,'v':v,'a':acc,'z':c['lowest_arm_z_m'],'object':c['object'],'shift_cm':100*np.linalg.norm(c['shift_m']),'pos':positions(z['state'].astype(float),z['action'].astype(float))})
  return out
 def score(P,n):
  P=np.asarray(P);dist=np.linalg.norm(pool[n][:,None]-P[None],axis=2).min(1)*100
  try:area=float(ConvexHull(P*100).volume) if len(P)>=3 else 0.
  except QhullError:area=0.
  return {'within_2cm':int((dist<=2).sum()),'median_nn_cm':float(np.median(dist)),'hull_cm2':area}
 methods={'workspace':candidates(ctx.work)}
 if a.max_shift_cm is not None:methods[f'workspace_max{a.max_shift_cm:g}cm']=[c for c in methods['workspace'] if c['shift_cm']<=a.max_shift_cm+1e-9]
 if a.comparison_work:
  methods['legacy_dense']=candidates(Path(a.comparison_work),True)
  methods['legacy_dense_label_checked']=[c for c in methods['legacy_dense'] if c['legacy_label_error']<=2e-6]
 rows=[]
 for n in range(1,len(E)+1):
  for sub in itertools.combinations(E,n):
   V=np.max([src[e]['v'] for e in sub],0);Acc=np.max([src[e]['a'] for e in sub],0);Z=min(src[e]['z'] for e in sub)
   row={'n':n,'subset':list(sub),'real':{name:score([src[e]['pos'][name] for e in sub],name) for name in names}}
   for method,cands in methods.items():
    ok=[c for c in cands if c['e'] in sub and (c['v']/np.maximum(V,1e-9)).max()<=ck.cfg['speed_ratio']+1e-6 and (c['a']/np.maximum(Acc,1e-9)).max()<=ck.cfg['accel_ratio']+1e-6 and c['z']>=Z-ck.cfg['table_tol_m']]
    row[method]={'accepted_episodes':len(ok),'objects':{name:{'accepted_object_variants':sum(c['object']==name for c in ok),'max_shift_cm':max([c['shift_cm'] for c in ok if c['object']==name]+[0]),**score([src[e]['pos'][name] for e in sub]+[c['pos'][name] for c in ok if c['object']==name],name)} for name in names}}
   rows.append(row)
 out={'source_dataset':ctx.cfg['source'],'pool_revision':a.pool_revision,'pool_episode_ids':pool_ids,'metric':'FK at gripper close; geometric coverage proxy, not real object-pose truth','comparison_caveat':f"Different target locations (configured workspace budget {ctx.cfg.get('variations',{}).get('object_workspace',{}).get('n_targets')} vs 48 dense offsets per object) and hand-distance policies; coverage comparison is engineering, not isolated algorithm effect. Minimal retiming chosen with all configured sources, then stricter subset envelopes rechecked; small-N yield is conservative. Rig calibration reused.",'nested_first_n':[r for r in rows if r['subset']==E[:r['n']]],'all_subsets':rows}
 out['max_shift_cm']=a.max_shift_cm
 Path(a.out or ctx.work/'source_budget.json').write_text(json.dumps(out,indent=1));print(json.dumps(out['nested_first_n'],indent=1))
if __name__=='__main__':main()
