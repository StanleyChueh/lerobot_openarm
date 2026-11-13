import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfagen.target_plan import target_offsets
from sfagen.handover_timing import handover_clock


class V02Tests(unittest.TestCase):
    def test_mixture_reports_heterogeneous_codecs_without_rewriting_sources(self):
        from sfagen.build import mixed_video_features
        a={'camera':dict(dtype='video',shape=[480,640,3],info={'video.codec':'av1','video.pix_fmt':'yuv420p','video.width':640})}
        b=copy.deepcopy(a);b['camera']['info'].update({'video.codec':'h264','video.pix_fmt':'yuv444p'})
        got=mixed_video_features([a,b])
        for f in got:
            self.assertIsNone(f['camera']['info']['video.codec']);self.assertIsNone(f['camera']['info']['video.pix_fmt'])
            self.assertEqual(f['camera']['info']['video.width'],640)
        self.assertEqual(a['camera']['info']['video.codec'],'av1');self.assertEqual(b['camera']['info']['video.codec'],'h264')
        self.assertEqual(mixed_video_features([a,a]),[a,a])

    def setUp(self):
        T = np.eye(4); T[:2, 3] = [.2, 0]
        self.rec = SimpleNamespace(e=0, objects=[dict(name="can", table=T)])
        self.cfg = dict(object="can", positions_xy_m=[[.24, 0], [.30, .09]],
                        source_positions_xy_m={0: [.2, 0], 1: [.29, 0], 2: [.22, 0]}, nearest_sources=2, max_shift_m=.08)

    def test_nearest_sources_and_cap_are_respected(self):
        rows = target_offsets(self.rec, self.cfg)
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0][3], [.04, 0, 0])
        cfg = copy.deepcopy(self.cfg); cfg["nearest_sources"] = 1
        self.assertEqual(target_offsets(self.rec, cfg), [])

    def test_stale_or_invalid_targets_fail_closed(self):
        for key, value in [("positions_xy_m", [[np.nan, 0]]), ("positions_xy_m", [[.24, 0], [.24, 0]]),
                           ("source_positions_xy_m", {0: [.201, 0]})]:
            cfg = copy.deepcopy(self.cfg); cfg[key] = value
            with self.assertRaises(ValueError):
                target_offsets(self.rec, cfg)

    def test_timing_keeps_duration_endpoints_and_grasp(self):
        for offset in [-10, 10]:
            tau, out, source = handover_clock(273, 97, 180, 210, offset)
            self.assertEqual(len(tau), 273); self.assertEqual(tau[0], 0); self.assertEqual(tau[-1], 272)
            self.assertTrue((np.diff(tau) > 0).all())
            np.testing.assert_array_equal(tau[:110], np.arange(110))
            np.testing.assert_allclose(tau[180+offset:211+offset], np.arange(180, 211))
            self.assertEqual(np.interp(180, tau, np.arange(273)), 180+offset)
        self.assertIsNone(handover_clock(220, 97, 180, 210, 10))

    def test_timing_plan_labels_camera_clock_and_corruption(self):
        from sfagen.handover_timing import plan
        from sfagen.timing import action_reference_error, source_frame_map
        T=273; t=np.arange(T); S=np.c_[np.sin(t/80), np.cos(t/90), np.zeros(T), np.zeros(T)]
        A=S[np.minimum(t+2,T-1)].copy(); A[180:,3]=1; A[200:,2]=1
        robot=SimpleNamespace(arms=['R','L'],sidx={'R':[0],'L':[1]},grip={'R':dict(state_index=2),'L':dict(state_index=3)})
        robot.split=lambda state: {'R':state[:,[0]],'L':state[:,[1]]}
        def pose(k,x):
            out=np.eye(4);out[0,3]=x/1000;return out
        rec=SimpleNamespace(T=T,state=S,action=A,robot=robot,arms=robot.arms,lag=-1.75,
             objects=[dict(arm='R',grasp_frame=97)],ev={'R':dict(opens=[200]),'L':dict(closes=[180])},
             sp={'gR':np.where(t>=210,1.,0.)},obj_pose=pose)
        for offset in [-10,10]:
            c=plan(rec,offset); tau=c['tau'];self.assertEqual(len(c['state']),T)
            self.assertLess(action_reference_error(rec,c),1e-12)
            np.testing.assert_allclose(c['objects'][0][:,0,3],tau/1000)
            np.testing.assert_array_equal(c['action'][:,3],A[np.floor(tau+1e-9).astype(int),3])
            self.assertEqual(np.flatnonzero(c['action'][:,3])[0],180+offset)
            self.assertEqual(source_frame_map(rec,c)[-1],T-1)
            for field in ['action','ref_action','ref_state']:
                bad=copy.deepcopy(c);bad[field][160,0]+=.01
                self.assertGreater(action_reference_error(rec,bad),2e-6)

    def test_warped_reference_cannot_enlarge_original_limits(self):
        from sfagen.checks import Checker
        from unittest.mock import patch
        ck=Checker.__new__(Checker)
        ck.robot=SimpleNamespace(arms=['R'],sidx={'R':[0]},lo={'R':np.array([-1.])},hi={'R':np.array([1.])})
        ck.cfg=dict(speed_ratio=1.25,accel_ratio=1.5,ik_pos_mm=1,ik_rot_deg=.5,ik_blend_pos_mm=5,ik_blend_rot_deg=2)
        ck.env=dict(speed=[1.],accel=[1.]);ck.kin=lambda A:(np.array([0.]),np.array([0.]))
        rec=SimpleNamespace(state=np.zeros((3,1)),action=np.zeros((3,1)))
        c=dict(state=rec.state,action=np.full((3,1),1.1),ref_state=rec.state,ref_action=np.full((3,1),1.1),w=np.zeros(3),ik_err=np.zeros((3,2)))
        with patch('sfagen.checks.nominal_action_error',return_value=0):
            failure=ck.quick_failure(rec,c)
        self.assertIn('limits',failure['fast_rejection_reasons'])


if __name__ == "__main__":
    unittest.main()
