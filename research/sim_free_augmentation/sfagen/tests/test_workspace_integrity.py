import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import av,numpy as np,pandas as pd
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from sfagen.timing import replay_reference,action_reference_error
from sfagen.verify import video_span_report
class IntegrityTests(unittest.TestCase):
 def test_joint_delay_source_action_and_gripper_corruptions(self):
  robot=SimpleNamespace(arms=['R'],sidx={'R':[0]},grip={'R':{'state_index':1}})
  x=np.arange(101)/100;S=np.c_[np.sin(x),x*0];A=np.c_[np.sin(np.minimum(x+.02,1)),(x>=.4).astype(float)]
  rec=SimpleNamespace(T=101,state=S,action=A,robot=robot,arms=['R']);tau=np.linspace(0,100,201);refS,refA=replay_reference(rec,tau,2)
  state=refS.copy();state[:,0]+=.1*np.sin(np.pi*np.arange(201)/200)**2
  action=refA.copy();future=np.minimum(np.arange(201)+2,200);action[:,0]+=(state-refS)[future,0]
  c=dict(tau=tau,state=state,action=action,ref_state=refS,ref_action=refA,lead=2,moved_arms=['R'])
  self.assertLess(action_reference_error(rec,c),1e-12)
  for corruption in ['delay_one_joint','source_actions','wrong_gripper','self_reference','nonfinite']:
   d=copy.deepcopy(c)
   if corruption=='delay_one_joint':d['action'][:,0]=np.roll(d['action'][:,0],5)
   if corruption=='source_actions':d['action']=refA.copy()
   if corruption=='wrong_gripper':d['action'][90:110,1]=1-d['action'][90:110,1]
   if corruption=='self_reference':d['ref_action']=action.copy()
   if corruption=='nonfinite':d['action'][10,0]=float('nan')
   self.assertGreater(action_reference_error(rec,d),2e-6,corruption)
 def test_duplicate_variations_fail_closed(self):
  from sfagen.generate import variation_list
  cfg={'variations':{'object_position':{'objects':['can'],'shifts_cm':[[1,0],[1,0]]}}}
  with self.assertRaises(ValueError):variation_list(cfg,SimpleNamespace(arms=['R']))
 def test_per_episode_video_deficit_and_boundary_overlap_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);cam='observation.images.body_cam';v=root/f'videos/{cam}/chunk-000';v.mkdir(parents=True);(root/'meta/episodes/chunk-000').mkdir(parents=True)
   with av.open(str(v/'file-000.mp4'),'w') as out:
    stream=out.add_stream('libx264',rate=30);stream.width=32;stream.height=32;stream.pix_fmt='yuv420p'
    for i in range(12):
     frame=av.VideoFrame.from_ndarray(np.full((32,32,3),30+i,np.uint8),format='rgb24')
     for packet in stream.encode(frame):out.mux(packet)
    for packet in stream.encode():out.mux(packet)
   def meta(a=0,b=6/30,c=6/30,d=12/30):
    rows=[{'episode_index':i,'length':6,f'videos/{cam}/chunk_index':0,f'videos/{cam}/file_index':0,f'videos/{cam}/from_timestamp':x,f'videos/{cam}/to_timestamp':y} for i,(x,y) in enumerate([(a,b),(c,d)])]
    pd.DataFrame(rows).to_parquet(root/'meta/episodes/chunk-000/file-000.parquet')
   meta();self.assertTrue(video_span_report(root,[cam],30)['pass'])
   meta(b=5/30);self.assertFalse(video_span_report(root,[cam],30)['pass'])
   meta(c=5/30);self.assertFalse(video_span_report(root,[cam],30)['pass'])
if __name__=='__main__':unittest.main()
