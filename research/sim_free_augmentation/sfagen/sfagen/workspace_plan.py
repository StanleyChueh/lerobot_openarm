"""Task-independent initial-object workspace augmentation, using kinematics and recorded RGB only.

Support targets come from an estimated source support surface, camera visibility and translated-grasp IK.
The near-object reach is translated rigidly; transit may take a vertical detour. A finite clock bank selects
short retimings that fit the source replay envelope. Protected transit clocks, a smoothly slowed pickup
window, and a slower whole-recording fallback are distinguished in provenance. Gripper command order is
preserved by zero-order hold. For command lead k, labels preserve the source action/future-state residual:
  a_out(n) = A(tau_n) + q_out(n+k) - S(tau_n+k).
The same reference is computed independently for unchanged arms. These are nominal labels, not measured
execution or physical contact evidence. Source-to-output exposure times are handled in timing.py.
"""
from __future__ import annotations
import cv2, numpy as np
from .plan import ik, smoothstep, action_lead, _attach, state_bounds
from .recon import interp_rows
from .timing import replay_reference


def _centre(o, T):
    return (T[:3, :3] @ o["points"].mean(0)) + T[:3, 3]


def _halfwidth(o):
    P = o["points"] @ o['table'][:3,:3].T; c = P.mean(0)
    return float(np.max(np.linalg.norm((P - c)[:, :2], axis=1)))


def support_model(rec, k, checks_cfg):
    """Conservative appearance-derived support proposal, not a calibrated table segmentation."""
    o=rec.objects[k];hsv=cv2.cvtColor(rec.plate,cv2.COLOR_RGB2HSV).astype(int)
    orig=rec.obj_sil(k,o['table']);robot0=np.isin(rec.robot_sil(0,sides=True),rec.arms)
    ring=(cv2.dilate(orig.astype('uint8'),np.ones((41,41),'uint8'))>0)&~(cv2.dilate(orig.astype('uint8'),np.ones((15,15),'uint8'))>0)&~robot0
    if ring.sum()<20:raise ValueError('Insufficient unoccluded support evidence around source object')
    ref=np.median(hsv[ring],0);tol=checks_cfg.get('plate_appearance_tol',30)
    mask=(np.abs(hsv[...,1]-ref[1])<=tol)&(np.abs(hsv[...,2]-ref[2])<=tol)
    mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((9,9),'uint8'))
    _,labels=cv2.connectedComponents(mask)
    ids,counts=np.unique(labels[ring],return_counts=True);counts[ids==0]=0
    if not counts.max():raise ValueError('No connected support surface found')
    mask=labels==ids[counts.argmax()]
    return mask,ref


def footprint_coverage(rec,o,T,support):
    P=o['points']@T[:3,:3].T+T[:3,3];P[:,2]=P[:,2].min()
    uv,z=rec.cam.project(P)
    if not np.isfinite(uv).all() or (z<=0).any():return 0.
    if (uv[:,0]<0).any() or (uv[:,0]>=rec.W).any() or (uv[:,1]<0).any() or (uv[:,1]>=rec.H).any():return 0.
    hull=cv2.convexHull(np.round(uv).astype('int32'));m=np.zeros((rec.H,rec.W),'uint8');cv2.fillConvexPoly(m,hull,1)
    return float(support[m>0].mean()) if m.any() else 0.


def support_targets(rec,k,checks_cfg,n_targets=48,step=.01,min_shift=.02,log=print,near_fraction=.5,max_shift=None):
    if not (0<step<=.1 and int(n_targets)==n_targets and n_targets>=1 and min_shift>=0 and 0<=near_fraction<=1):raise ValueError('Invalid workspace grid settings')
    o=rec.objects[k];T0=o['table'];arm=o['arm'];tg=o['grasp_frame'];cam=rec.cam
    support,ref=support_model(rec,k,checks_cfg);c0=_centre(o,T0)
    # Only front-facing rays on the support plane. Bound by the arm-chain length so a horizon cannot create a huge grid.
    yy,xx=np.nonzero(support);uv=np.c_[xx[::16],yy[::16]]
    z=(o['points']@T0[:3,:3].T+T0[:3,3])[:,2].min()
    with np.errstate(divide='ignore',invalid='ignore'):P=cam.backproject_plane(uv,z)
    jid=rec.robot.m.frames[rec.robot.tcp[arm]].parentJoint;reach=0.
    while jid:reach+=np.linalg.norm(rec.robot.m.jointPlacements[jid].translation);jid=rec.robot.m.parents[jid]
    reach=max(float(reach),.5)
    good=np.isfinite(P).all(1)&(np.linalg.norm(P[:,:2]-c0[:2],axis=1)<2*reach)
    P=P[good];_,depth=cam.project(P);P=P[depth>0]
    if not len(P):raise ValueError('No front-facing support plane intersections')
    lo,hi=P[:,:2].min(0),P[:,:2].max(0)
    xs=np.arange(np.floor(lo[0]/step)*step,hi[0],step);ys=np.arange(np.floor(lo[1]/step)*step,hi[1],step)
    others=[oo['points'][::5]@oo['table'][:3,:3].T+oo['table'][:3,3] for j,oo in enumerate(rec.objects) if j!=k]
    sp=rec.robot.split(rec.state[:1]);rec.robot.fk({a:sp[a][0] for a in rec.arms},{a:sp['g'+a][0] for a in rec.arms if 'g'+a in sp})
    hull0=np.concatenate([Q for n,a,Q in rec.robot.link_points() if a!='B'])
    plo,phi=o['points'].min(0)-.01,o['points'].max(0)+.01
    TG=rec.tcp(arm,tg);qg=rec.q_at(tg);lo_q=np.minimum(rec.robot.lo[arm],qg[arm]);hi_q=np.maximum(rec.robot.hi[arm],qg[arm])
    keep=[];why={x:0 for x in ('grid','view','surface','gap','robot_start','ik','near_original','beyond_max_shift')}
    for x in xs:
      for y in ys:
        why['grid']+=1;d=np.array([x-c0[0],y-c0[1],0.])
        if np.linalg.norm(d)<min_shift:why['near_original']+=1;continue
        if max_shift is not None and np.linalg.norm(d)>max_shift+1e-9:why['beyond_max_shift']+=1;continue  # R1 shift limit
        Tn=T0.copy();Tn[:3,3]+=d;Pn=o['points']@Tn[:3,:3].T+Tn[:3,3];u,depth=cam.project(Pn)
        if not (np.isfinite(u).all() and (depth>0).all() and (u[:,0]>10).all() and (u[:,0]<rec.W-10).all() and (u[:,1]>10).all() and (u[:,1]<rec.H-10).all()):why['view']+=1;continue
        if footprint_coverage(rec,o,Tn,support)<checks_cfg.get('support_coverage_min',.95):why['surface']+=1;continue
        if others and min(float(np.linalg.norm(Pn[::5,None,:2]-Q[None,:,:2],axis=2).min()) for Q in others)<checks_cfg.get('object_gap_m',.01):why['gap']+=1;continue
        ql=(hull0-Tn[:3,3])@Tn[:3,:3]
        if np.all((ql>plo)&(ql<phi),axis=1).any():why['robot_start']+=1;continue
        Tt=TG.copy();Tt[:3,3]+=d;q=ik(rec.robot,arm,Tt,qg[arm],qg,lo_q,hi_q)
        rec.robot.fk({**qg,arm:q});Tq=rec.robot.frame(rec.robot.tcp[arm])
        if np.linalg.norm(Tq[:3,3]-Tt[:3,3])>1e-3 or np.degrees(np.arccos(np.clip((np.trace(Tq[:3,:3].T@Tt[:3,:3])-1)/2,-1,1)))>.5:why['ik']+=1;continue
        keep.append(d)
    R=np.asarray(keep).reshape(-1,3)
    info={'rejected':why,'region_points':len(R),'original_centre_xy':c0[:2].tolist(),'support_hsv_reference':ref.tolist(),'support_pixels':int(support.sum()),'coordinate_status':'estimated from source camera/object fit; not hardware validated'}
    if not len(R):return np.zeros((0,3)),info,R
    total=min(int(n_targets),len(R));broad=max(1,int(np.ceil(total*(1-near_fraction))))
    sel=[int(np.argmin(np.linalg.norm(R[:,:2]-R[:,:2].mean(0),axis=1)))];dmin=np.linalg.norm(R[:,:2]-R[sel[0],:2],axis=1)
    def add(i):
        sel.append(int(i));np.minimum(dmin,np.linalg.norm(R[:,:2]-R[i,:2],axis=1),out=dmin)
    while len(sel)<broad:add(int(np.argmax(dmin)))
    # Preserve near-source examples as well as extremes; every choice is derived from source geometry.
    near=np.linalg.norm(R[:,:2],axis=1)<=np.quantile(np.linalg.norm(R[:,:2],axis=1),.25)
    remaining=np.flatnonzero(near & ~np.isin(np.arange(len(R)),sel))
    if len(sel)<total and len(remaining):add(remaining[np.argmin(np.linalg.norm(R[remaining,:2],axis=1))])
    while len(sel)<total:
        remaining=np.flatnonzero(near & ~np.isin(np.arange(len(R)),sel))
        if not len(remaining):break
        add(remaining[np.argmax(dmin[remaining])])
    while len(sel)<total:add(int(np.argmax(dmin)))
    info.update(sampling_strategy='centroid/extremes plus nearest-quartile coverage',broad_targets=broad,near_fraction=near_fraction)
    info.update(region_area_cm2=float(len(R)*(100*step)**2),targets_world_xy=(R[sel,:2]+c0[:2]).tolist(),region_world_xy=(R[:,:2]+c0[:2]).tolist())
    return R[sel],info,R


def approach_distances(o,P):
    """Distance to the object's local 3D bounds, including height (an overhead hand is not at the object)."""
    local=(P-o['table'][:3,3])@o['table'][:3,:3]
    lo,hi=o['points'].min(0),o['points'].max(0)
    return np.linalg.norm(np.maximum(0,np.maximum(lo-local,local-hi)),axis=1)


def f1w_schedule(rec, k, approach_margin=0.10):
    o = rec.objects[k]; arm = o["arm"]; tg = o["grasp_frame"]
    P = rec.robot.tcp_from_state(rec.state, arm)[:, :3, 3]
    v = np.r_[0, np.linalg.norm(np.diff(P, axis=0), axis=1)] * rec.ctx.src.fps
    mov = np.flatnonzero(v[: tg - 15] > 0.02); t0 = int(mov[0]) if len(mov) else 0
    r_app=approach_margin
    far=np.flatnonzero(approach_distances(o,P[:tg])>r_app)
    ta=int(far[-1]+1) if len(far) else t0+10
    # Earlier unrelated arm movement must not spread the relocation across another object's pickup.
    # Use the final travel bout after a sustained rest, with enough frames left for a smooth transit.
    low=v[:ta]<.015;edges=np.diff(np.r_[False,low,False].astype(int))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    rests=[int(b) for a,b in zip(starts,ends) if b-a>=6 and b<=ta-10]
    if rests:t0=max(t0,rests[-1])
    # Do not clamp the rigid approach to an earlier frame where the hand is still far from the object.
    # If no sufficiently long transit exists, retain the failure instead of disguising a swept approach.
    ta = max(ta,t0+10)
    nxt = min([e for s in rec.ev for kind in ("closes", "opens") for e in rec.ev[s][kind] if e > tg] + [rec.T])
    win = np.arange(tg + 30, min(tg + 4 * int(rec.ctx.src.fps), nxt - 10))
    if len(win) == 0 or not (t0 + 10 <= ta <= tg - 3):
        return None
    tr = int(win[np.argmax(P[win, 2])])
    events=sorted({e for s in rec.ev.values() for key in ('closes','opens') for e in s[key]})
    return {"t_motion": t0, "t_approach": ta, "t_grasp": tg, "t_rejoin": tr,
            "next_event": int(nxt), "r_app_m": r_app,"approach_metric":"3D distance to object bounds; final travel bout",
            "protected_windows":[[max(0,e-8),min(rec.T-1,e+12)] for e in events]}


def _w_src(m, t):
    t = np.asarray(t, float); w = np.zeros_like(t)
    a = (t >= m["t_motion"]) & (t < m["t_approach"]); w[a] = smoothstep((t[a] - m["t_motion"]) / (m["t_approach"] - m["t_motion"]))
    w[(t >= m["t_approach"]) & (t <= m["t_grasp"] + 10)] = 1.0
    b = (t > m["t_grasp"] + 10) & (t < m["t_rejoin"]); w[b] = 1 - smoothstep((t[b] - m["t_grasp"] - 10) / (m["t_rejoin"] - m["t_grasp"] - 10))
    return w


def time_map(m, T, s):
    """C2 monotone time stretch outside every gripper-event window; integer offsets elsewhere."""
    if not np.isfinite(s) or s<1:raise ValueError('time scale must be finite and >=1')
    spans=[(m['t_motion'],m['t_approach']),(m['t_grasp']+10,m['t_rejoin'])]
    protected=m.get('protected_windows',[])
    knots=sorted({0,T-1,*[x for ab in spans for x in ab],*[x for ab in protected for x in ab]})
    knots=[int(x) for x in knots if 0<=x<T]
    output=[0];tau_parts=[]
    for a,b in zip(knots[:-1],knots[1:]):
        L=b-a;mid=(a+b)/2
        stretch=any(x<=mid<=y for x,y in spans) and not any(x<=mid<=y for x,y in protected)
        extra=int(round((s-1)*L)) if stretch else 0
        M=L+extra;y=np.arange(M,dtype=float)
        if extra:
            lo=np.zeros(M);hi=np.ones(M)
            for _ in range(40):
                u=(lo+hi)/2;ease=u*u*u*(10+u*(-15+6*u))
                below=L*u+extra*ease<y;lo=np.where(below,u,lo);hi=np.where(below,hi,u)
            values=a+L*(lo+hi)/2;values[0]=a
        else:values=a+y
        tau_parts.append(values);output.append(output[-1]+M)
    tau=np.r_[np.concatenate(tau_parts),T-1.]
    return tau,{'src_knots':knots,'out_knots':output}


def window_time_map(m,T,s,fps):
    """Constant slowed pickup/transit with C2 speed ramps in adjacent source context; later interaction retains pace."""
    a=float(m['t_motion']);b=float(m['t_rejoin']);left=max(0.,a-fps);right=min(T-1.,b+fps,float(m['next_event']-8))
    right=max(right,b);La=a-left;Lb=right-b;K=b-a+.5*(La+Lb);extra=round((s-1)*K);gain=extra/max(K,1e-9)
    def integral(u):return u**6-3*u**5+2.5*u**4
    def out(x):
        u=np.clip((x-left)/max(La,1e-9),0,1);v=np.clip((x-b)/max(Lb,1e-9),0,1)
        return x+gain*(La*integral(u)+np.clip(x-a,0,b-a)+Lb*(v-integral(v)))
    y=np.arange(T+extra,dtype=float);lo=np.zeros_like(y);hi=np.full_like(y,T-1.)
    for _ in range(42):
        mid=(lo+hi)/2;below=out(mid)<y;lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    tau=(lo+hi)/2;tau[0]=0.;tau[-1]=T-1.
    knots=sorted(set([0.,left,a,b,right,T-1.]))
    return tau,{'src_knots':knots,'out_knots':[float(out(x)) for x in knots]}


def _clock(m,T,scale,clock,fps):
    if clock=='uniform':
        tau=np.linspace(0,T-1,int(round((T-1)*scale))+1);return tau,{'src_knots':[0,T-1],'out_knots':[0,len(tau)-1]}
    if clock=='window':return window_time_map(m,T,scale,fps)
    return time_map(m,T,scale)


def _native_spatial(rec,k,d,m,action_bounds=False,margin=0.0):
    """Retarget a Cartesian path; route transit above a relocated object if a straight blend crosses it."""
    r=rec.robot;arm=rec.objects[k]['arm'];sp=r.split(rec.state);q={a:sp[a].copy() for a in rec.arms}
    lo,hi=state_bounds(rec,arm,sp[arm],action_bounds,margin)
    P=r.tcp_traj(q,arm);t=np.arange(rec.T);W=_w_src(m,t);field=W[:,None]*d
    obj=rec.objects[k];centre=_centre(obj,obj['table'])+d
    world=obj['points']@obj['table'][:3,:3].T+obj['table'][:3,3]+d
    u=np.clip((t-m['t_motion'])/(m['t_approach']-m['t_motion']),0,1);bump=np.sin(np.pi*u)**2
    xy=P[:,:2,3]+field[:,:2]
    risk=(np.linalg.norm(xy-centre[:2],axis=1)<_halfwidth(obj)+.065)&(bump>.05)&(W<.999)
    need=np.maximum(world[:,2].max()+.04-P[:,2,3],0)
    lift=float(np.max(need[risk]/bump[risk],initial=0))
    field[:,2]+=lift*bump
    act=np.linalg.norm(field,axis=1)>1e-9
    for i in np.flatnonzero(act):
        target=P[i].copy();target[:3,3]+=field[i]
        q0=q[arm][i-1]+sp[arm][i]-sp[arm][i-1] if i>0 and act[i-1] else sp[arm][i]
        q[arm][i]=ik(r,arm,target,q0,{a:sp[a][i] for a in rec.arms},lo[i],hi[i])
    m={**m,'transit_lift_m':lift}
    return m,q,field


def f1w_plan(rec, k, d, s=1.0, approach_margin=0.10, spatial=None, clock='local'):

    m = f1w_schedule(rec, k, approach_margin) if spatial is None else spatial[0]
    if m is None:
        return None
    robot, S, A = rec.robot, rec.state, rec.action; o = rec.objects[k]; arm = o["arm"]; d = np.asarray(d, float)
    tau,kn=_clock(m,rec.T,s,clock,rec.ctx.src.fps)
    Tn = len(tau); w = _w_src(m, tau)
    kl=action_lead(S,A);Sw,Aref=replay_reference(rec,tau,kl);sp=robot.split(Sw)
    q = {a: sp[a].copy() for a in robot.arms}; qn = {a: sp[a].copy() for a in robot.arms}
    if spatial is None:m,base_q,field=_native_spatial(rec,k,d,m)
    else:_,base_q,field=spatial
    offsets=interp_rows(field,tau)
    qn[arm]=interp_rows(base_q[arm],tau)
    TT = robot.tcp_traj(q,arm);act=w>1e-9;err=np.zeros((Tn,2))
    for t in np.flatnonzero(act):
        robot.fk({a: qn[a][t] for a in robot.arms}); cur = robot.frame(robot.tcp[arm]); Tt = TT[t].copy(); Tt[:3, 3] += offsets[t]
        err[t] = [np.linalg.norm(cur[:3, 3] - Tt[:3, 3]), np.degrees(np.arccos(np.clip((np.trace(cur[:3, :3].T @ Tt[:3, :3]) - 1) / 2, -1, 1)))]
    # actions with the recorded command lead k: a'(n) = A(tau_n) + [q'(n+k) - S(tau_n + k)]
    idx = np.minimum(np.arange(Tn) + kl, Tn - 1)
    Sn = Sw.copy(); An = Aref.copy()
    for a in robot.arms:
        cols = robot.sidx[a]; qa = qn[a] if a == arm else q[a]
        corr = qa[idx] - q[a][idx]; corr[np.abs(corr) < 1e-12] = 0.0
        An[:, cols] += corr
        if a == arm:
            Sn[:, cols] = qn[a]
    # object tracks in output time
    objs = []
    for j, oo in enumerate(rec.objects):
        P = np.zeros((Tn, 4, 4))
        for n in range(Tn):
            h = rec.holder_at(j, tau[n])
            if h == "table":
                P[n] = oo["table"].copy()
                if j == k:
                    P[n][:3, 3] += d
            elif h == "placed":
                P[n] = rec.obj_pose(j, tau[n])
            else:
                robot.fk({a: qn[a][n] for a in robot.arms}); P[n] = robot.frame(robot.tcp[h]) @ _attach(oo, rec, h, tau[n])
        objs.append(P)
    out_idx = lambda t_src: int(np.searchsorted(tau, t_src - 1e-9))  # noqa: E731
    marks = {**m, **{f"{key}_out": out_idx(m[key]) for key in ("t_motion", "t_approach", "t_grasp", "t_rejoin")}, "time_scale": s,"retiming_mode":clock,
             "source_t_grasp":m["t_grasp"],"frames_out": Tn, "frames_added": Tn - rec.T, **kn}
    marks_check = {**marks, "t_grasp": marks["t_grasp_out"]}              # placement check works in output frames
    return {"family": "object_workspace", "object": o["name"], "shift_m": d.tolist(), "marks": marks_check, "w": w, "state": Sn, "action": An,
            "q": qn, "ik_err": err, "lead": kl, "moved_arms": [arm], "objects": objs, "edited": act, "t_rejoin": marks["t_rejoin_out"],
            "tau": tau, "ref_state": Sw, "ref_action": Aref,"native_q":base_q,"native_offset":field}


def plan_with_time_scale(rec, k, d, checker, scales=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0), approach_margin=0.10,spatial=None,action_bounds=False,margin=0.0):
    """Shortest tested clock meeting commanded speed/acceleration limits; other checks remain mandatory."""
    cfg, env = checker.cfg, checker.env; last = None
    m=f1w_schedule(rec,k,approach_margin) if spatial is None else spatial[0]
    if m is None:return None
    if spatial is None:spatial=_native_spatial(rec,k,np.asarray(d,float),m,action_bounds,margin)
    trials=[(s,'local') for s in scales if s<=3]+[(s,'window') for s in scales if s>1]+[(s,'uniform') for s in scales if s>1]
    trials.sort(key=lambda trial:len(_clock(m,rec.T,*trial,rec.ctx.src.fps)[0]))
    chosen=trials[-1]
    arm=rec.objects[k]['arm'];lead=action_lead(rec.state,rec.action)
    for scale,clock in trials:
        tau,_=_clock(m,rec.T,scale,clock,rec.ctx.src.fps)
        Sw,Aref=replay_reference(rec,tau,lead);future=np.minimum(np.arange(len(tau))+lead,len(tau)-1)
        dq=interp_rows(spatial[1][arm],tau)-Sw[:,rec.robot.sidx[arm]]
        A=Aref.copy();A[:,rec.robot.sidx[arm]]+=dq[future]
        v,a=checker.kin(A);chosen=(scale,clock)
        if (v/np.maximum(env['speed'],1e-9)).max()<=cfg['speed_ratio'] and (a/np.maximum(env['accel'],1e-9)).max()<=cfg['accel_ratio']:break
    return f1w_plan(rec,k,d,chosen[0],approach_margin,spatial=spatial,clock=chosen[1])
