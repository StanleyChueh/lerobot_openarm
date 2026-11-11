import unittest,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sfagen.render import candidate_frames_tw
class ExposureTests(unittest.TestCase):
 def test_attachment_uses_fk_at_fractional_exposure(self):
  class Robot:
   tcp={'R':0}
   def split(self,S):return {'R':S}
   def fk(self,q,g):self.angle=q['R'][0]
   def frame(self,fid):
    x=self.angle;T=np.eye(4);T[:2,:2]=[[np.cos(x),-np.sin(x)],[np.sin(x),np.cos(x)]];return T
  X=np.eye(4);X[0,3]=1.;S=np.array([[0.],[np.pi/2]]);r=Robot()
  rec=SimpleNamespace(T=2,lag=.5,arms=['R'],robot=r,holder_at=lambda k,t:'R',objects=[{'holder':[('R',0,2)],'X':[X]}])
  poses=[]
  for angle in S[:,0]:r.fk({'R':np.array([angle])},{});poses.append(r.frame(0)@X)
  c={'state':S,'q':{'R':S},'objects':[np.array(poses)]}
  frame=next(candidate_frames_tw(rec,c));actual=frame[-1][0][:2,3]
  np.testing.assert_allclose(actual,[np.sqrt(.5),np.sqrt(.5)],atol=1e-12)
  self.assertGreater(np.linalg.norm(actual-np.array([.5,.5])),.2)
if __name__=='__main__':unittest.main()
