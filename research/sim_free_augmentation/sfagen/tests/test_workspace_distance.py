"""Compare exact pair pruning with all-pairs distance, including intersecting geometry."""
import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pinocchio as pin,coal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfagen.checks import Checker
class DistanceTests(unittest.TestCase):
 def test_pruned_minimum_equals_all_pairs_with_penetration(self):
  m=pin.Model();joints=[m.addJoint(0,pin.JointModelPX(),pin.SE3(np.eye(3),np.array([x,0.,0.])),arm) for x,arm in [(0.,'R'),(1.,'L')]]
  geometry=pin.GeometryModel()
  for arm,jid,x in [('R',joints[0],0.),('L',joints[1],0.),('B',0,2.)]:geometry.addGeometryObject(pin.GeometryObject(arm,jid,pin.SE3(np.eye(3),np.array([x,0.,0.])),coal.Box(.2,.2,.2)))
  r=SimpleNamespace(m=m,d=m.createData(),col=geometry,col_arm=['R','L','B'],arms=['R','L'],sidx={'R':[0],'L':[1]})
  r.split=lambda S:{'R':S[:,[0]],'L':S[:,[1]]};r.q_full=lambda q,g:np.r_[q['R'],q['L']]
  ctx=SimpleNamespace(robot=r,cfg={},src=SimpleNamespace(fps=30));ck=Checker(ctx)
  S=np.array([[0,0],[.5,-.5],[.2,-.5],[-.4,.2],[.7,-.8]],float)
  old=ck.min_dist(S,1,reference=True);new=ck.min_dist(S,1)
  np.testing.assert_allclose(new,old,atol=1e-8);self.assertTrue(np.any(old<0))
if __name__=='__main__':unittest.main()
