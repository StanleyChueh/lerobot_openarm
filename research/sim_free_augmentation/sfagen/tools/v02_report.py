"""Recompute v02 coverage/timing/label evidence and build an exportable visual report (CPU)."""
import hashlib, html, json, sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S))
from sfagen.context import Context
from sfagen.checks import nominal_action_error


def save(fig,name,out):
    for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    out=S/'results/v02';ctx=Context(S/'configs/openarm_pringles_v02.yaml');w=ctx.work
    frozen=json.loads((out/'FREEZE.json').read_text())
    drift=[p for p,h in frozen['files'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    amendment=out/'PACKAGING_AMENDMENT.json'
    if amendment.exists():
        amendments=json.loads(amendment.read_text())['files']
        drift=[p for p in drift if p not in amendments or amendments[p]['frozen_sha256']!=frozen['files'][p] or amendments[p]['current_sha256']!=hashlib.sha256(Path(p).read_bytes()).hexdigest()]
    assert not drift, f'Frozen input changed: {drift}'
    planned=json.loads((out/'planned_targets.json').read_text());grid=np.array(planned['targets']['positions_xy_m']);P=np.array(planned['eval_xy_m'])
    pos=np.array(list(planned['targets']['source_positions_xy_m'].values()))
    rows=[];independent=[];G=Counter();families=Counter();failures=Counter();clocks=[];durations=[];limit_rows=[]
    for e in ctx.episodes():
        root=w/f'candidates/ep{e}';path=root/'checks.json'
        if not path.exists():continue
        m=json.loads((root/'generation_manifest.json').read_text());assert m['complete']
        checks=json.loads(path.read_text());S0,A0=ctx.src.arrays(e);T=len(S0)
        rec=SimpleNamespace(T=T,state=S0,action=A0,robot=ctx.robot,arms=ctx.robot.arms)
        idx=[j for arm in ctx.robot.arms for j in ctx.robot.sidx[arm]]
        lo=np.concatenate([ctx.robot.lo[a] for a in ctx.robot.arms]);hi=np.concatenate([ctx.robot.hi[a] for a in ctx.robot.arms])
        joint_names=[name for arm in ctx.robot.arms for name in ctx.cfg['robot']['arms'][arm]['joints']]
        for key,ck in checks.items():
            rows.append(dict(source=e,variation=key,**ck));families[(ck['family'],ck['pass'])]+=1
            if not ck['pass']:
                candidate_path=root/f'{key}.npz'
                if ck['family']=='object_position' and candidate_path.exists():
                    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest()==m['candidate_sha256'][key]
                    with np.load(candidate_path) as z:
                        for field,Y in [('state',S0),('action',A0)]:
                            X=z[field][:,idx];excess=np.maximum(0,np.maximum(np.minimum(lo,Y[:,idx].min(0))-X,X-np.maximum(hi,Y[:,idx].max(0))))
                            for j in np.flatnonzero(excess.max(0)>1e-6):limit_rows.append(dict(source=e,variation=key,field=field,joint=joint_names[j],max_excess_deg=float(np.rad2deg(excess[:,j].max()))))
                reasons=ck.get('fast_rejection_reasons',[])
                if not reasons and ck.get('error'):reasons=[ck['error']]
                if not reasons:
                    tests={'mesh clearance':ck['min_mesh_distance_mm'] < min(ck['source_min_mesh_distance_same_frames_mm'],20)-1,
                           'hand clearance':ck.get('min_hand_distance_mm',1e9)<ck.get('source_min_hand_distance_same_frames_mm',0)-5,
                           'table':ck['lowest_arm_z_m']<ck['real_lowest_arm_z_m']-.01,
                           'placement':not ck.get('placement',{}).get('ok',True),
                           'object floor':ck['object_lowest_point_vs_source_mm'] < -5}
                    reasons=[k for k,v in tests.items() if v] or ['other replay gate']
                failures.update(reasons);continue
            p=root/f'{key}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==m['candidate_sha256'][key]
            with np.load(p) as z:c={k:z[k].astype(float) for k in ['state','action']+(['tau','ref_state','ref_action'] if 'tau' in z else [])}
            tau=c.get('tau',np.arange(len(c['state'])));timing=ck['family']=='interaction_timing'
            durations.append((T,len(tau),timing,float(np.diff(tau).min()),float(np.diff(tau).max())))
            c.update(lead=ck['lead'],moved_arms=['R'] if ck['object'] else ctx.robot.arms)
            limits=all(((X[:,idx]>=np.minimum(lo,Y[:,idx].min(0))-1e-6)&(X[:,idx]<=np.maximum(hi,Y[:,idx].max(0))+1e-6)).all() for X,Y in [(c['state'],S0),(c['action'],A0)])
            good=dict(source=e,variation=key,fixed_duration=len(tau)==T,endpoints=bool(tau[0]==0 and tau[-1]==T-1),
                      monotone=bool((np.diff(tau)>0).all()),spatial_unit_slope=timing or bool(np.allclose(tau,np.arange(T),atol=1e-10)),
                      original_limits=bool(limits),label_error=nominal_action_error(rec,c))
            if timing:
                knots=np.array(ck['marks']['knots_source']);outs=np.array(ck['marks']['knots_output']);offset=ck['marks']['dt_frames']
                good['initial_grasp_clock']=bool(np.allclose(tau[:int(knots[1])+1],np.arange(int(knots[1])+1)))
                good['handover_shift']=bool(np.allclose(tau[int(outs[2]):int(outs[3])+1],np.arange(int(knots[2]),int(knots[3])+1)))
                clocks.append((e,key,tau))
            good['pass']=all(v for k,v in good.items() if k not in ['source','variation','label_error']) and good['label_error']<=2e-6
            independent.append(good)
            if key.startswith('can_target_'):G[int(key.rsplit('_',1)[1])]+=1
    (out/'rejected_joint_limits.json').write_text(json.dumps(limit_rows,indent=2))
    complete=len(list(w.glob('candidates/ep*/checks.json')))==10
    accepted=grid[sorted(G)] if G else np.empty((0,2))
    cov=json.loads((S/'results/eval_v00/coverage.json').read_text());realpos=np.array(cov['real_xy_cm'])/100
    D=np.concatenate([realpos,accepted])
    nn=lambda A:np.linalg.norm(P[:,None]-A[None],axis=2).min(1)*100
    realdist=nn(realpos);newdist=nn(D);plandist=nn(np.concatenate([realpos,grid]))
    v00dist=nn(np.concatenate([realpos,np.array(cov['v00_xy_cm'])/100]))
    checks={}
    for condition in ['copied','right_wrist']:
        p=out/f'quality_{condition}/quality_report.json'
        checks[condition]=json.loads(p.read_text()) if p.exists() else None
    match=None
    copies=[];export_checks={}
    for condition in ['copied','right_wrist']:
        p=w.parent/condition/'dataset/meta/sfa_provenance.json'
        if p.exists():
            import pandas as pd
            prov=json.loads(p.read_text());copies.append(prov);root=p.parents[1]
            df=pd.read_parquet(sorted((root/'data').rglob('*.parquet')))
            expected={(r['source'],r['variation']) for r in rows if r['pass']}
            actual={(e['source_real_episode'],e['variation']) for e in prov['episodes']}
            exp=dict(all_accepted_exported=actual==expected and len(prov['episodes'])==len(expected),synthetic_only=not prov['contains_real_episodes'],
                labels=all(hashlib.sha256(np.stack(df[df.episode_index==e['export_episode_index']][key].to_numpy()).astype(np.float32).tobytes()).hexdigest()==e[sha] for e in prov['episodes'] for key,sha in [('observation.state','state_sha256'),('action','action_sha256')]))
            mixed=root.parent/'dataset_mixed'
            if mixed.exists() and (mixed/'meta/sfa_provenance.json').exists():
                md=pd.read_parquet(sorted((mixed/'data').rglob('*.parquet')));flag=np.stack(md.is_synthetic.to_numpy()).reshape(-1)
                exp['mixed_flags']=bool(np.array_equal(flag,(md.episode_index.to_numpy()>=10).astype(float)))
                exp['mixed_episode_count']=md.episode_index.nunique()==10+len(prov['episodes'])
                exp['mixed_real_labels']=all(np.array_equal(np.stack(md[md.episode_index==e][key].to_numpy()).astype(np.float32),ctx.src.arrays(e)[i].astype(np.float32)) for e in ctx.episodes() for i,key in enumerate(['observation.state','action']))
                exp['mixed_synthetic_labels']=all(np.array_equal(np.stack(md[md.episode_index==e['export_episode_index']+10][key].to_numpy()).astype(np.float32),np.stack(df[df.episode_index==e['export_episode_index']][key].to_numpy()).astype(np.float32)) for e in prov['episodes'] for key in ['observation.state','action'])
                video_hashes=lambda base:{hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'videos').rglob('*.mp4')}
                mix_hashes=video_hashes(mixed)
                exp['mixed_original_video_bytes']=video_hashes(ctx.src.root)<=mix_hashes
                exp['mixed_synthetic_video_bytes']=video_hashes(root)<=mix_hashes
            else:exp['mixed_export_exists']=False
            export_checks[condition]={k:bool(v) for k,v in exp.items()}
    if len(copies)==2:
        fields=['source_real_episode','variation','n_frames','state_sha256','action_sha256','source_state_time','wrist_source_frame_index','render_fingerprint']
        match=dict(episodes=len(copies[0]['episodes']),pass_all=[{k:e[k] for k in fields} for e in copies[0]['episodes']]==[{k:e[k] for k in fields} for e in copies[1]['episodes']])
    rep=dict(complete=complete,attempts=len(rows),accepted=sum(r['pass'] for r in rows),targets_planned=35,targets_accepted=len(G),
             family_counts=[dict(family=f,passed=families[(f,True)],attempts=families[(f,True)]+families[(f,False)]) for f in sorted({f for f,b in families})],
             failures=dict(failures),spatial_limit_exceedances=limit_rows,independent_candidate_checks=independent,all_independent_pass=all(r['pass'] for r in independent),
             matched_exports=match,export_checks=export_checks,coverage=dict(eval_starts=40,real_median_cm=float(np.median(realdist)),v02_median_cm=float(np.median(newdist)),
               v02_max_cm=float(newdist.max()),v00_median_cm=float(np.median(v00dist)),v00_max_cm=float(v00dist.max()),v02_within_2cm=int((newdist<=2).sum())),
             quality={c:None if q is None else q['pass'] for c,q in checks.items()},rows=rows)
    (out/'summary.json').write_text(json.dumps(rep,indent=2))
    evidence=[Path(__file__),out/'FREEZE.json',S/'results/eval_v00/coverage.json']
    evidence += [p for p in w.glob('candidates/ep*/generation_manifest.json')]
    evidence += [p for p in w.parent.glob('*/dataset*/meta/sfa_provenance.json')]
    (out/'REPORT_PROVENANCE.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},indent=2))
    fig,ax=plt.subplots(figsize=(8,6));ax.scatter(P[:,1]*100,P[:,0]*100,facecolors='none',edgecolors='#df8434',label='40 measured starts')
    ax.scatter(realpos[:,1]*100,realpos[:,0]*100,marker='*',s=90,c='#34415d',label='10 measured real sources')
    ax.scatter(grid[:,1]*100,grid[:,0]*100,c='#cecece',marker='x',s=65,label='Rejected grid targets')
    if len(accepted):
        ax.scatter(accepted[:,1]*100,accepted[:,0]*100,c='#087e8b',s=80,marker='s',label='Accepted grid targets')
        for i,count in G.items():ax.annotate(str(count),(grid[i,1]*100,grid[i,0]*100),ha='center',va='center',color='white',fontsize=7)
    ax.set(xlabel='World y (cm; robot left →)',ylabel='World x (cm; away from robot →)',title=f'{len(G)}/35 planned positions survived the fixed gates')
    ax.set_aspect('equal');ax.legend(fontsize=8);ax.grid(alpha=.15);save(fig,'accepted_coverage',out)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');F=rep['family_counts'];labels=[r['family'].replace('interaction_','handover\n').replace('object_position','can position') for r in F]
    passed=[r['passed'] for r in F];rejected=[r['attempts']-r['passed'] for r in F]
    axs[0].bar(labels,passed,label='Pass',color='#087e8b');axs[0].bar(labels,rejected,bottom=passed,label='Reject',color='#d4a78a');axs[0].set(ylabel='Candidate attempts',title='No slowing or gate relaxation');axs[0].legend()
    for i,r in enumerate(F):axs[0].text(i,r['attempts']+.5,f"{r['passed']}/{r['attempts']}",ha='center')
    fail=failures.most_common();axs[1].barh([x[0].replace('no valid schedule (see plan.py rules)','No valid schedule') for x in fail],[x[1] for x in fail],color='#ad6d50');axs[1].invert_yaxis();axs[1].set(xlabel='Rejected attempts (may have several causes)',title='Why candidates were rejected')
    save(fig,'rejections',out)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for label,d in [('Real10',realdist),('+v00',v00dist),('+v02',newdist),('Planned grid',plandist)]:axs[0].plot(np.sort(d),np.arange(1,41)/40,label=label)
    axs[0].axvline(2,color='grey',ls=':');axs[0].set(xlabel='Nearest can start (cm)',ylabel='Fraction of 40 evaluation starts',title='Approximate spatial coverage');axs[0].legend()
    for e,key,tau in clocks:axs[1].plot(np.arange(len(tau))/30,(tau-np.arange(len(tau)))/30,lw=1,alpha=.5)
    axs[1].axhline(0,color='#34415d',ls=':');axs[1].set(xlabel='Output time (s)',ylabel='Source time − output time (s)',title='Timing variants: local warp, fixed duration')
    save(fig,'coverage_timing',out)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');d=np.array(durations)
    for timing,color,label in [(False,'#087e8b','Spatial'),(True,'#c67b3d','Timing')]:
        sub=d[d[:,2]==timing];axs[0].scatter(sub[:,0]/30,sub[:,1]/30,c=color,label=label,alpha=.6)
        axs[1].scatter(np.flatnonzero(d[:,2]==timing),sub[:,3],c=color,marker='v',s=15)
        axs[1].scatter(np.flatnonzero(d[:,2]==timing),sub[:,4],c=color,marker='^',s=15)
    axs[0].plot([7.5,9.5],[7.5,9.5],ls=':',c='grey');axs[0].set(xlabel='Source duration (s)',ylabel='Generated duration (s)',title='All accepted durations equal their source');axs[0].legend()
    axs[1].axhline(1,c='grey',ls=':');axs[1].set(xlabel='Accepted candidate index',ylabel='Source-time slope (frames/frame)',title='Min/max slope: timing exception is explicit')
    save(fig,'duration_gate',out)
    status='Generation complete' if complete else 'Generation in progress'
    qtext='; '.join(f'{c}: '+('pending' if q is None else ('PASS' if q['pass'] else 'FAIL')) for c,q in checks.items())
    matched_text='pending' if match is None else f"{match['episodes']}/104 episodes match on candidate IDs, state/action hashes, body fingerprints and wrist source clocks: {'PASS' if match['pass_all'] else 'FAIL'}"
    body=f'''<p class="eyebrow">STEP 2 · LOCAL EXPERIMENT · BASE 77e52d4</p><h1>v02: coverage without longer episodes</h1>
    <p>{status}. {rep['accepted']}/{rep['attempts']} attempted variants passed generation checks; {len(G)}/35 grid positions survived.
    Rejected rear targets remain visible. This is kinematic evidence; robot success and physical RGB fidelity remain untested.</p>
    <div class="stats"><strong>{rep['accepted']}<small>accepted episodes</small></strong><strong>{len(G)}/35<small>grid positions accepted</small></strong><strong>{rep['coverage']['v02_within_2cm']}/40<small>eval starts within 2 cm</small></strong><strong>{len(independent)}<small>independent clock/label checks</small></strong></div>
    <h2>Did the grid survive?</h2><p>Squares mark accepted positions; numbers count source demos. Grey crosses expose gaps.
    The arena was designed from prior evaluation recordings and is development-informed, not a held-out test.</p><img src="accepted_coverage.svg" alt="Accepted and rejected grid targets with measured evaluation starts">
    <h2>What was rejected?</h2><p>Existing duration, speed, acceleration, IK, placement and clearance rules were fixed before generation. Failed variants were not slowed or clipped.</p><img src="rejections.svg" alt="Candidate acceptance by family and rejection causes">
    <p>Detailed inspection found 30 rejected spatial attempts with right joint 6 action labels outside the original allowed bounds, by up to 0.97°. These labels were preserved and rejected. Other gates were not all evaluated after early rejection, so correcting this bound alone would not establish feasibility. <a href="rejected_joint_limits.json">Exact joint-limit records</a>.</p>
    <h2>Coverage and timing</h2><p>Nearest positions are approximate world-plane reconstructions. Both baselines use the same measured source centers from Step 1; v02 adds modeled target centers. Fitted source-object centers differ from those measurements by 0.39–2.10 cm, so these distances are proxies, not surveyed accuracy.</p><img src="coverage_timing.svg" alt="Coverage distribution and local handover time warps">
    <p>Spatial variants retain the exact source clock. Only the approved handover timing variants change local slope; all retain source frame count and fps. Overlapping points represent repeated source durations.</p><img src="duration_gate.svg" alt="Source versus generated durations and minimum maximum source-time slopes">
    <h2>Export quality</h2><p>{qtext}. {matched_text}.</p>
    <p>Right-wrist reconstruction remains an ablation: the prior Pringles fidelity test failed. The ≥30 dB export gate only measures codec agreement with generated images.</p>
    <h2>What the images show</h2><p>The inspected raw examples show local warping around the can/background in the rendered wrist view, and block-like fill artifacts in a handover-location body view. Preserve these as limitations; the codec test does not measure realism.</p>
    <div id="examples"></div><h2>Evidence and reproduction</h2><p><a href="PROTOCOL.md">Frozen protocol</a> · <a href="FREEZE.json">Exact hashes</a> · <a href="summary.json">All checks and rejection rows</a> · <a href="LITERATURE.md">Literature refresh</a> · <a href="../eval_v00/index.html">Step 1 diagnosis</a></p>
    <p>No training, robot/CAN, physics, commits or dataset publication performed. The chart SVG/PNG files can be reused in a paper with these limitations.</p>'''
    examples=out/'examples/manifest.json'
    if examples.exists():
        gallery='<p>Rows: clock-matched real body, generated body, copied right wrist, rendered right wrist. Columns follow approach → grasp → receiver close → giver open → end. These are raw rendered frames, not decoded export frames.</p>'
        for e in json.loads(examples.read_text()):gallery+=f'<a href="{e["path"]}"><img src="{e["path"]}" alt="Source {e["source"]}, {e["variation"]}: matched body and right wrist across task phases" loading="lazy"></a>'
        body=body.replace('<div id="examples"></div>',gallery)
    for c,q in checks.items():
        if q:
            body+=f'<h2>{c.replace("_"," ").title()} decoded examples</h2><p>Rows: body, left wrist, right wrist. Columns: start, one-third, midpoint, two-thirds, end. Deterministic quality sample; click to inspect.</p>'
            for p in sorted((out/f'quality_{c}/sheets').glob('*.jpg'))[:4]:body+=f'<a href="{p.relative_to(out)}"><img src="{p.relative_to(out)}" alt="{p.stem}" loading="lazy"></a>'
    if all(q is not None for q in checks.values()):
        table='<div style="overflow-x:auto"><table style="min-width:600px;border-spacing:18px 10px"><tr><th>Condition</th><th>Speed max<br>≤1.25× real</th><th>Acceleration max<br>≤1.5× real</th><th>Body codec<br>≥30 dB</th><th>Rendered wrist codec<br>≥30 dB (sample minimum)</th></tr>'
        for c,q in checks.items():
            t=q['trajectories'];im=q['images'];wp=im['wrist_decoded_vs_rendered_psnr_db_min']
            table+=f"<tr><td>{c.replace('_',' ')}</td><td>{t['max_speed_ratio']:.3f}×</td><td>{t['max_accel_ratio']:.3f}×</td><td>{im['body_decoded_vs_rendered_psnr_db']:.2f} dB</td><td>{f'{wp:.2f} dB' if wp is not None else 'copied'}</td></tr>"
        table+='</table></div><p>Both conditions also pass finite values, joint limits, table clearance, start/gripper rules, exact label hashes and 50-action CPU load tests. The default TorchCodec backend additionally loads real, first-synthetic and final mixed samples. <a href="RESULTS.md">Full result table and local dataset paths</a> · <a href="PACKAGING_AMENDMENT.md">Packaging-only codec correction</a>.</p>'
        body=body.replace('<h2>What the images show</h2>',table+'<h2>What the images show</h2>')
    md=f'''# v02 results — local, uncommitted

The bounded Step 2 batch completed generation with {rep['accepted']}/{rep['attempts']} passing candidates, but **only {len(G)}/35 arena grid positions survived**.
It does not establish full rear-arena coverage or policy improvement. [Visual report](index.html).

| Family | Accepted / attempted |
|---|---:|
'''+''.join(f"| {r['family']} | {r['passed']} / {r['attempts']} |\n" for r in rep['family_counts'])+f'''
![Accepted and rejected target positions](accepted_coverage.png)

Coverage of the same 40 prior measured starts: {rep['coverage']['v02_within_2cm']}/40 within 2 cm; median nearest distance {rep['coverage']['v02_median_cm']:.2f} cm;
worst {rep['coverage']['v02_max_cm']:.2f} cm, versus v00 {rep['coverage']['v00_max_cm']:.2f} cm. These are approximate geometric proxies from reused development/evaluation recordings.
The model-fitted source positions differ from the measured positions by 0.39–2.10 cm. No new held-out policy evaluation occurred.

![Rejections with unchanged gates](rejections.png)

Thirty rejected spatial attempts exceed original right-joint-6 action bounds by up to 0.97°. They were neither clipped
nor admitted; remaining necessary/full gates were not all evaluated after early rejection. [Exact records](rejected_joint_limits.json).

![Coverage distributions and fixed-duration local timing warp](coverage_timing.png)

![Every accepted episode retains source duration](duration_gate.png)

All 104 accepted candidates pass independent fixed-duration, monotone clock, endpoint, spatial-unit-slope,
original-source state/action-limit and nominal-label checks. The approved ±10-frame timing exception changes local
slope while keeping the initial grasp clock and unit-slope handover interval. The existing F1/F2 planners are unchanged.

## Local exports and quality

Each condition has 104 synthetic episodes / 26,781 frames and a separate mixture of 10 real + 104 synthetic episodes
/ 29,356 frames. Conditions differ only in right-wrist image reconstruction and associated wrist-validity flags.
{matched_text}.

Quality status: {qtext}. The independent export checks compare labels, flags and exact original/synthetic MP4 bytes.
53 CPU tests pass. Full verifier passes for both conditions, including 312 boundary/midpoint samples each.
The default TorchCodec backend additionally loads real, first-synthetic and final mixed frames with 50-action chunks in each condition.

| Condition | Max speed / real | Max acceleration / real | Max jerk / real (reported) | Body codec PSNR (dB) | Min sampled rendered-wrist codec PSNR (dB) | Gate |
|---|---:|---:|---:|---:|---:|---|
'''
    for condition,q in checks.items():
        if q:
            t=q['trajectories'];im=q['images'];wp=im['wrist_decoded_vs_rendered_psnr_db_min']
            md+=f"| {condition} | {t['max_speed_ratio']:.3f} | {t['max_accel_ratio']:.3f} | {t['max_jerk_ratio']:.3f} | {im['body_decoded_vs_rendered_psnr_db']:.2f} | {wp if wp is not None else 'copied'} | {'PASS' if q['pass'] else 'FAIL'} |\n"
    md+='''
Thresholds remain speed ≤1.25×, acceleration ≤1.5×, decoded body/rendered-wrist PSNR ≥30 dB; all other original gates remain.
Jerk is descriptive, not an added acceptance gate. Codec PSNR compares decoded video to generated RGB, not physical truth.
The inspected images show wrist texture/background warping and body fill artifacts. The previously failed Pringles wrist
fidelity gate remains a limitation. Kinematic consistency does not establish contact force, physical replay or task success.

## Evidence and reproduction

[Protocol](PROTOCOL.md) · [Frozen hashes](FREEZE.json) · [Packaging-only amendment](PACKAGING_AMENDMENT.md) ·
[Runtime/source assets](SOURCE_ASSETS.json) · [All machine-readable results](summary.json) · [Literature](LITERATURE.md).
Chart PNG/SVG files are exportable; annotated examples are in examples/, and 10 decoded sheets per condition are saved in quality_*/sheets/.
Use the branch root README for exact commands. Failed temporary mixture outputs are retained; no gates were loosened.
No training, robot/CAN, physics, commits, pushes or dataset uploads were performed.
'''
    for condition in ['copied','right_wrist']:
        for dataset in ['dataset','dataset_mixed']:
            path=w.parent/condition/dataset;md+=f'\n- {condition} {dataset}: [{path}]({path})\n'
    (out/'RESULTS.md').write_text(md)
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>v02 generation evidence</title><style>body{font:17px/1.6 system-ui;color:#24334a;background:#f6f8fa;margin:0}main{max-width:1100px;margin:auto;padding:36px 24px}h1{font-size:42px;line-height:1.15}h2{margin-top:44px}.eyebrow{font-size:12px;letter-spacing:.15em;color:#087e8b}img{max-width:100%;background:white;border:1px solid #dde3ea;border-radius:10px;margin:12px 0}.stats{display:flex;flex-wrap:wrap;gap:18px}.stats strong{background:white;padding:20px;border-radius:12px;flex:1;font-size:34px;min-width:150px}.stats small{display:block;font-size:13px;font-weight:500}a{color:#087e8b}@media(max-width:600px){h1{font-size:30px}main{padding:20px 16px}}</style><main>'''+body+'</main></html>')
    print(json.dumps({k:v for k,v in rep.items() if k not in ['rows','independent_candidate_checks','spatial_limit_exceedances']},indent=2))


if __name__=='__main__':main()
