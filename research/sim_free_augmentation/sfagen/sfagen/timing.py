"""Continuous source/output clocks for variable-duration candidates (no robot execution)."""
from __future__ import annotations
import numpy as np
import pinocchio as pin
from .recon import interp_rows


def time_coordinates(rec, candidate):
    n=len(candidate['state'])
    tau=np.asarray(candidate.get('tau',np.arange(n)),dtype=float)
    if tau.shape!=(n,) or not np.isfinite(tau).all() or np.any(np.diff(tau)<=0):
        raise ValueError('tau must be finite, strictly increasing, one source time per output row')
    if abs(tau[0])>1e-8 or abs(tau[-1]-(rec.T-1))>1e-8:
        raise ValueError('tau must retain both source endpoints')
    return tau


def source_time_at(rec,candidate,output_time):
    tau=time_coordinates(rec,candidate);x=np.asarray(output_time,float)
    # Unit-slope extension preserves the first/last camera frame when lag reaches outside state support.
    return np.interp(x,np.arange(len(tau)),tau)+np.minimum(x,0)+np.maximum(x-(len(tau)-1),0)


def source_frame_map(rec,candidate,lag=None):
    lag=rec.lag if lag is None else float(lag)
    source_frame=source_time_at(rec,candidate,np.arange(len(candidate['state']))+lag)-lag
    return np.clip(np.floor(source_frame+.5).astype(np.int64),0,rec.T-1)


def resample_commands(action,tau,robot):
    out=interp_rows(action,tau)
    idx=np.clip(np.floor(np.asarray(tau)+1e-9).astype(int),0,len(action)-1)
    for arm in robot.arms:
        if robot.grip[arm]:
            col=robot.grip[arm]['state_index'];out[:,col]=action[idx,col]
    return out


def replay_reference(rec,tau,lead):
    state=interp_rows(rec.state,tau)
    action=resample_commands(rec.action,tau,rec.robot)
    future=np.minimum(np.arange(len(tau))+lead,len(tau)-1)
    source_future=interp_rows(rec.state,np.minimum(tau+lead,rec.T-1))
    for arm in rec.arms:
        cols=rec.robot.sidx[arm]
        action[:,cols]+=state[future][:,cols]-source_future[:,cols]
    return state,action


def pose_at(poses,time):
    x=float(np.clip(time,0,len(poses)-1));i=int(np.floor(x));j=min(i+1,len(poses)-1);f=x-i
    out=poses[i].copy()
    if f:
        out[:3,3]=(1-f)*poses[i][:3,3]+f*poses[j][:3,3]
        out[:3,:3]=poses[i][:3,:3]@pin.exp3(f*pin.log3(poses[i][:3,:3].T@poses[j][:3,:3]))
    return out


def action_reference_error(rec,c):
    """Independent construction of expected labels; never use the candidate action as its own reference."""
    tau=time_coordinates(rec,c);refS,refA=replay_reference(rec,tau,c['lead'])
    expected=refA.copy();future=np.minimum(np.arange(len(tau))+c['lead'],len(tau)-1)
    for arm in c['moved_arms']:
        cols=rec.robot.sidx[arm];expected[:,cols]+=(c['state']-refS)[future][:,cols]
    values=[c['state'],c['action'],c['ref_state'],c['ref_action']]
    if not all(np.isfinite(x).all() for x in values):return float('inf')
    return max(float(np.abs(refS-c['ref_state']).max()),float(np.abs(refA-c['ref_action']).max()),float(np.abs(c['action']-expected).max()))
