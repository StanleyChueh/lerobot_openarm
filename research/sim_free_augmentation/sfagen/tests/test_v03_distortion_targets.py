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


def _fake_rec(T=280, tg=80, nxt=200):
    """Right TCP rests at (.3,0,.1) for 20 frames, reaches the can at the origin by frame 60, holds, lifts after grasp."""
    P = np.zeros((T, 3)); P[:20] = [.3, 0, .1]; P[20:60] = np.linspace([.3, 0, .1], [0, 0, .02], 40); P[60:tg + 1] = [0, 0, .02]
    P[tg + 1:] = np.linspace([0, 0, .02], [0, 0, .25], T - tg - 1)

    def tcp(S, arm):
        out = np.tile(np.eye(4), (len(S), 1, 1)); out[:, :3, 3] = S[:, :3]; return out
    robot = SimpleNamespace(tcp_from_state=tcp)
    obj = dict(arm="R", grasp_frame=tg, table=np.eye(4), points=np.array([[-.03, -.03, 0], [.03, .03, .1]]))
    ev = {"R": dict(closes=[tg], opens=[230]), "L": dict(closes=[nxt], opens=[])}
    return SimpleNamespace(objects=[obj], robot=robot, state=P, T=T, ev=ev, ctx=SimpleNamespace(src=SimpleNamespace(fps=30)))


class V03Rev2Tests(unittest.TestCase):
    def test_wide_schedule_absorbs_shift_over_longest_event_free_span(self):
        from sfagen.workspace_plan import f1w_schedule
        rec = _fake_rec(); m0 = f1w_schedule(rec, 0); m1 = f1w_schedule(rec, 0, wide=True)
        self.assertEqual((m0["t_approach"], m0["t_grasp"]), (m1["t_approach"], m1["t_grasp"]))
        self.assertEqual(m1["t_motion"], 0); self.assertLessEqual(m1["t_motion"], m0["t_motion"])
        self.assertEqual(m1["t_rejoin"], 200 - 10); self.assertGreaterEqual(m1["t_rejoin"], m0["t_rejoin"])
        rec.ev["L"]["opens"] = [5]                                             # earlier gripper event: ramp starts after it
        self.assertEqual(f1w_schedule(rec, 0, wide=True)["t_motion"], min(5 + 12, m0["t_motion"]))

    def test_duration_and_speed_similarity_gates(self):
        from sfagen.checks import Checker
        rec = _fake_rec(); ck = object.__new__(Checker); ck.robot = rec.robot; ck.ctx = rec.ctx
        ck.cfg = dict(fixed_duration=True, speed_similarity=[0.8, 1.25])
        ed = np.zeros(rec.T, bool); ed[10:190] = True
        same = dict(state=rec.state.copy(), edited=ed, moved_arms=["R"])
        r = ck.timing_similarity(rec, same); self.assertTrue(r["timing_ok"]); self.assertEqual(r["tcp_peak_speed_ratio"], [1.0, 1.0])
        fast = dict(same, state=rec.state * np.r_[1.4, 1.4, 1.4]); self.assertFalse(ck.timing_similarity(rec, fast)["timing_ok"])
        longer = dict(same, state=np.vstack([rec.state, rec.state[-1:]]), edited=np.r_[ed, False])
        self.assertFalse(ck.timing_similarity(rec, longer)["timing_ok"])                    # one extra frame fails
        ck.cfg = {}; self.assertTrue(ck.timing_similarity(rec, longer)["timing_ok"])        # gates are opt-in (rev 1 unchanged)


if __name__ == "__main__":
    unittest.main()
