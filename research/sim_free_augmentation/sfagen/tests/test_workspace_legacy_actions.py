"""A nominal action above a limit must reach the checker, not be silently clipped away from its state."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pinocchio as pin
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfagen.plan import _retarget
from sfagen.checks import nominal_action_error
class LegacyActionTests(unittest.TestCase):
 def test_retarget_does_not_hide_invalid_labels_by_clipping(self):
  m=pin.Model();jid=m.addJoint(0,pin.JointModelPX(),pin.SE3.Identity(),'R');fid=m.addFrame(pin.Frame('tcp',jid,0,pin.SE3.Identity(),pin.FrameType.OP_FRAME));data=m.createData()
  r=SimpleNamespace(m=m,d=data,tcp={'R':fid},vidx={'R':[0]},arms=['R'],sidx={'R':[0]},lo={'R':np.array([0.])},hi={'R':np.array([.2])})
  r.q_full=lambda q:np.asarray(q['R']);r.split=lambda S:{'R':S.copy()};r.fk=lambda q:pin.framesForwardKinematics(m,data,q['R']);r.frame=lambda f:data.oMf[f].homogeneous.copy()
  def trajectory(q,arm):
   out=[]
   for row in q[arm]:r.fk({arm:row});out.append(r.frame(fid))
   return np.stack(out)
  r.tcp_traj=trajectory;S=np.linspace(0,.1,10)[:,None];A=S+.1;rec=SimpleNamespace(robot=r,state=S,action=A)
  Sn,An,q,err,lead=_retarget(rec,np.ones(10),np.array([.1,0.,0.]),['R'])
  self.assertGreater(An[-1,0],.29)
  c=dict(state=Sn,action=An,lead=lead,moved_arms=['R']);self.assertLess(nominal_action_error(rec,c),1e-12)
  c['action']=np.minimum(An,.2);self.assertGreater(nominal_action_error(rec,c),.09)
if __name__=='__main__':unittest.main()
