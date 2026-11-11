import sys; sys.path.insert(0,'.')
import numpy as np, srecon, sgen
from sfa.robot import Robot
rec=srecon.StationRecon(0, device="cuda:0"); print(rec.st_info); mk=sgen.schedule(rec); print(mk)
T0=rec.st_table; sp=Robot.split(rec.state)
for t in (0, 30, 60, 90, mk["t_grasp"]-15):
    rec.robot.fk(sp["R"][t], sp["L"][t], sp["gR"][t], sp["gL"][t]); hits={}
    for n,s,Q in rec.robot.link_points("hull"):
        if s not in "RL": continue
        ql=(Q-T0[:3,3])@T0[:3,:3]
        h=((np.abs(ql[:,0])<rec.st_ext[0]/2+0.01)&(np.abs(ql[:,1])<rec.st_ext[1]/2+0.01)&(ql[:,2]<rec.st_ext[2]+0.01)).sum()
        if h: hits[s+n]=(int(h), round(float(ql[:,2].min()),3))
    print(t, hits)
