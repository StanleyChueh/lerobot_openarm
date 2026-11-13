import sys, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfagen.distortion import frame_scores, episode_score, MIN_PX
from sfagen.target_plan import target_offsets


class V03Tests(unittest.TestCase):
    def test_distortion_score(self):
        f = lambda t, m, s: dict(tmask_px=t, miss_px=m, spill_px=s)  # noqa: E731
        fr = [dict(copied=True), f(10000, 1000, 500), f(MIN_PX - 1, 900, 900), f(10000, 0, 0)]
        np.testing.assert_allclose(frame_scores(fr), [0, .15, 0, 0])
        self.assertAlmostEqual(episode_score([f(10000, 2000, 0)] * 19 + [f(10000, 0, 0)]), .2)
        self.assertEqual(episode_score([dict(copied=True)] * 5), 0.0)      # zero-shift / untouched episode
        self.assertEqual(episode_score([]), 0.0)
        with self.assertRaises(ValueError):                                  # pre-instrumentation renders fail closed
            frame_scores([dict(warped_px=5000, inpainted_px=10)])

    def test_f1w_planner_family_and_all_sources(self):
        T = np.eye(4); T[:2, 3] = [.2, 0]; rec = SimpleNamespace(e=0, objects=[dict(name="can", table=T)])
        cfg = dict(object="can", positions_xy_m=[[.24, 0], [.30, .09]], source_positions_xy_m={0: [.2, 0], 1: [.29, 0], 2: [.22, 0]},
                   nearest_sources=10, max_shift_m=.14, planner="f1w")
        rows = target_offsets(rec, cfg)                                       # target 001 is 13.5 cm from source 0
        self.assertEqual(len(target_offsets(rec, {**cfg, "max_shift_m": .12})), 1)
        self.assertEqual([r[0] for r in rows], ["can_target_000", "can_target_001"])   # far target kept when all sources tried
        self.assertTrue(all(r[1] == "object_workspace" for r in rows))
        self.assertEqual(target_offsets(rec, {**cfg, "planner": "f1"})[0][1], "object_position")
        with self.assertRaises(ValueError):
            target_offsets(rec, {**cfg, "planner": "nope"})


    def test_action_bound_margin_keeps_recorded_pose_feasible(self):
        from sfagen.plan import state_bounds
        T = 60; t = np.linspace(0, 1, T)[:, None]; S = 0.5 * t; A = np.vstack([S[2:], S[-1:], S[-1:]]) + 0.1  # gap 0.1, lead 2
        robot = SimpleNamespace(lo={'R': np.array([-1.0])}, hi={'R': np.array([0.55])}, sidx={'R': [0]})
        rec = SimpleNamespace(robot=robot, state=S, action=A)
        lo0, hi0 = state_bounds(rec, 'R', S.copy(), True, 0.0); lo1, hi1 = state_bounds(rec, 'R', S.copy(), True, 0.01)
        self.assertTrue((hi1 <= hi0 + 1e-12).all() and (hi1 >= S - 1e-12).all() and (lo1 <= S + 1e-12).all())
        self.assertTrue((hi0 - hi1 <= 0.01 + 1e-12).all() and (hi1 < hi0 - 1e-9).any())   # margin acts where the label binds
        np.testing.assert_array_equal(state_bounds(rec, 'R', S.copy(), False, 0.01)[1], np.maximum(0.55, S))


if __name__ == "__main__":
    unittest.main()
