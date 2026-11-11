import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfagen.workspace_plan import time_map
from sfagen.timing import source_frame_map,source_time_at,resample_commands,replay_reference,pose_at
import pinocchio as pin
class TimingTests(unittest.TestCase):
 def setUp(self):
  self.robot=SimpleNamespace(arms=['R'],grip={'R':{'state_index':1}},sidx={'R':[0]})
  self.S=np.c_[np.linspace(0,1,101),np.zeros(101)];self.A=self.S.copy();self.A[40:,1]=1
  self.rec=SimpleNamespace(T=101,lag=-1.75,state=self.S,action=self.A,robot=self.robot,arms=['R'])
  self.m={'t_motion':5,'t_approach':35,'t_grasp':45,'t_rejoin':85,'protected_windows':[[37,57],[65,76]]}
 def test_identity_camera_endpoints(self):
  c={'state':self.S};np.testing.assert_array_equal(source_frame_map(self.rec,c),np.arange(101))
 def test_monotone_and_event_protection(self):
  for scale in [1,1.25,2,3,6]:
   tau,kn=time_map(self.m,101,scale);self.assertEqual(tau[0],0);self.assertEqual(tau[-1],100);self.assertTrue(np.all(np.diff(tau)>0));self.assertTrue(np.max(np.diff(tau))<=1+1e-8)
   for a,b in self.m['protected_windows']:
    part=tau[(tau>=a)&(tau<=b)];np.testing.assert_allclose(part,np.round(part),atol=1e-8)
   c={'state':np.zeros((len(tau),2)),'tau':tau};jm=source_frame_map(self.rec,c);self.assertEqual(jm[0],0);self.assertEqual(jm[-1],100)
 def test_window_clock_keeps_late_interaction_pace(self):
  from sfagen.workspace_plan import window_time_map
  m={**self.m,'next_event':100,'t_rejoin':65}
  for scale in [1,1.25,2,4,6]:
   tau,_=window_time_map(m,101,scale,20)
   self.assertTrue(np.all(np.diff(tau)>0));self.assertEqual(tau[0],0);self.assertEqual(tau[-1],100)
   late=tau[tau>85.01];np.testing.assert_allclose(np.diff(late),1,atol=1e-8)
 def test_invalid_source_clock_rejected(self):
  from sfagen.timing import time_coordinates
  for tau in [np.array([0.,2.,1.]),np.array([0.,np.nan,100.]),np.array([1.,50.,100.])]:
   with self.assertRaises(ValueError):time_coordinates(self.rec,{'state':np.zeros((3,2)),'tau':tau})
 def test_pregrasp_guard_covers_rigid_approach_after_slowing(self):
  from sfagen.checks import pregrasp_guard_end
  rec=SimpleNamespace(objects=[{'grasp_frame':97}])
  tau=np.linspace(0,272,1089)
  c={'tau':tau,'marks':{'t_grasp':388,'t_approach_out':228}}
  self.assertEqual(pregrasp_guard_end(rec,c,0),328)
  self.assertGreater(pregrasp_guard_end(rec,c,0),c['marks']['t_approach_out'])
  self.assertEqual(pregrasp_guard_end(rec,{'marks':{'t_grasp':97}},0),82)
 def test_overhead_hand_is_far_from_small_object(self):
  from sfagen.workspace_plan import approach_distances
  o={'table':np.eye(4),'points':np.array([[-.01,-.01,0],[.01,.01,.02]])}
  P=np.array([[0,0,.4],[0,0,.025],[.2,0,.02]])
  distances=approach_distances(o,P)
  np.testing.assert_allclose(distances,[.38,.005,.19],atol=1e-12)
  self.assertGreater(distances[0],.1)
 def test_gripper_commands_zoh(self):
  tau=np.array([39.,39.5,39.99,40.,40.1]);out=resample_commands(self.A,tau,self.robot);np.testing.assert_array_equal(out[:,1],[0,0,0,1,1])
 def test_independent_reference_actions(self):
  tau,_=time_map(self.m,101,2);S,A=replay_reference(self.rec,tau,2)
  self.assertTrue(np.isfinite(A).all());self.assertFalse(np.shares_memory(A,self.A));self.assertFalse(np.shares_memory(S,self.S))
 def test_rotation_interpolation(self):
  P=np.repeat(np.eye(4)[None],2,axis=0);P[1,:3,:3]=pin.exp3(np.array([0.,0.,np.pi/2]));P[1,0,3]=2
  T=pose_at(P,.5);np.testing.assert_allclose(T[:3,:3],pin.exp3(np.array([0.,0.,np.pi/4])),atol=1e-10);self.assertEqual(T[0,3],1)
if __name__=='__main__':unittest.main()
