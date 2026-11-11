import sys; sys.path[:0]=['/home/stanley/Stanley_ws/sfa_hf_export/battery/bsfa','/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation']
import numpy as np, bdata
from sfa.robot import Robot
rb=Robot(); df=bdata.table()
for e in range(10):
    d=df[df.episode_index==e]; S=np.stack(d["observation.state"].to_numpy()).astype(float); A=np.stack(d["action"].to_numpy()).astype(float)
    ev=bdata.grip_events(A); sp=Robot.split(S)
    TL=rb.tcp_traj({"R":sp["R"],"L":sp["L"]},"L"); TR=rb.tcp_traj({"R":sp["R"],"L":sp["L"]},"R")
    pL=TL[:,:3,3]; v=np.r_[0,np.linalg.norm(np.diff(pL,axis=0),axis=1)]*30
    tg=ev["L"]["closes"][0]; t0=int(np.flatnonzero(v[:tg]>0.02)[0])
    # first time after tg+20 where speed stays < 1.5 cm/s for 15 frames
    settle=next(t for t in range(tg+20,len(S)-15) if (v[t:t+15]<0.015).all())
    dRL=np.linalg.norm(TR[:,:3,3]-pL,axis=1)
    near=next((t for t in range(tg,len(S)) if dRL[t]<0.20), None)
    rot=[np.degrees(np.arccos(np.clip((np.trace(TL[tg][:3,:3].T@TL[t][:3,:3])-1)/2,-1,1))) for t in (tg+30,settle)]
    print(e,"L onset",t0,"grasp",tg,"settle",settle,"R grasp",ev["R"]["closes"][0],"R-L<20cm at",near,"zL grasp/settle %.1f/%.1f"%(pL[tg,2]*100,pL[settle,2]*100),"rot",np.round(rot,0), "Lpos settle",np.round(pL[settle]*100,1))
