"""Offline tests for gate0_diagnostics.py (synthetic data only; no dataset, network, GPU or robot).

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python -m unittest \
        research/contact_aware_augmentation/scripts/test_gate0_diagnostics.py -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO)]

import gate0_diagnostics as g  # noqa: E402
from tools.dataset_diagnostics.loader import load_dataset  # noqa: E402
from tools.dataset_diagnostics.tests.fixtures import write_v30  # noqa: E402


class Projection(unittest.TestCase):
    def test_quaternion_matches_pinocchio(self):
        import pinocchio as pin
        q = g.BODY_CAM["quat_wxyz"]
        R = g.quat_wxyz_to_R(q)
        self.assertTrue(np.allclose(R, pin.Quaternion(*q).normalized().toRotationMatrix(), atol=1e-12))
        self.assertTrue(np.allclose(R.T @ R, np.eye(3), atol=1e-12))

    def test_point_on_optical_axis_hits_principal_point(self):
        R, t = g.quat_wxyz_to_R(g.BODY_CAM["quat_wxyz"]), np.array(g.BODY_CAM["pos"])
        p = t + R @ np.array([0.0, 0.0, 0.7])  # 0.7 m along camera +z
        uv, z = g.project(p[None])
        self.assertAlmostEqual(z[0], 0.7, places=9)
        self.assertTrue(np.allclose(uv[0], [g.BODY_CAM["cx"], g.BODY_CAM["cy"]], atol=1e-9))

    def test_offset_point_scales_with_focal_length(self):
        R, t = g.quat_wxyz_to_R(g.BODY_CAM["quat_wxyz"]), np.array(g.BODY_CAM["pos"])
        p = t + R @ np.array([0.1, -0.05, 0.5])
        uv, _ = g.project(p[None])
        self.assertAlmostEqual(uv[0, 0], g.BODY_CAM["fx"] * 0.2 + g.BODY_CAM["cx"], places=6)
        self.assertAlmostEqual(uv[0, 1], g.BODY_CAM["fy"] * -0.1 + g.BODY_CAM["cy"], places=6)


class Events(unittest.TestCase):
    def test_segments(self):
        self.assertEqual(g.segments(np.array([0, 1, 1, 0, 1], bool)), [(1, 3), (4, 5)])
        self.assertEqual(g.segments(np.zeros(4, bool)), [])

    def test_gripper_events_blocked_vs_empty(self):
        a = np.ones(100); a[30:70] = 0.0
        s_blocked = np.where(a < 0.5, 0.42, 0.96)
        s_empty = np.where(a < 0.5, -0.03, 0.96)
        eb, ee = g.gripper_events(a, s_blocked), g.gripper_events(a, s_empty)
        self.assertEqual((eb["first_close"], eb["first_open_after_close"]), (30, 70))
        self.assertTrue(eb["closes"][0]["blocked"])
        self.assertFalse(ee["closes"][0]["blocked"])

    def test_close_until_end_has_no_open(self):
        a = np.ones(50); a[20:] = 0.0
        ev = g.gripper_events(a, a)
        self.assertIsNone(ev["first_open_after_close"])


class Lag(unittest.TestCase):
    def test_best_lag_recovers_shift(self):
        rng = np.random.default_rng(0)
        acts, sts = {}, {}
        for e in range(3):
            a = np.cumsum(rng.normal(0, 0.01, (200, 16)), 0)
            s = np.vstack([np.repeat(a[:1], 3, 0), a[:-3]])  # state follows the command 3 frames later
            acts[e], sts[e] = a, s
        self.assertEqual(g.best_lag(acts, sts, list(range(14)))["k_best"], 3)


class OnSyntheticDataset(unittest.TestCase):
    """The handover fixture: right closes at 30 %, left at 55 %, right opens at 65 %; state lags 1 frame."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = write_v30(Path(cls.tmp.name) / "ds", lengths=(100, 120, 140))
        cls.view = load_dataset(root=root, label="synthetic")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_events(self):
        rows = g.episode_events(self.view)
        for r in rows:
            n = r["length"]
            self.assertEqual(r["right_close"], int(0.30 * n))
            self.assertEqual(r["left_close"], int(0.55 * n))
            self.assertEqual(r["right_open"], int(0.65 * n))
            self.assertTrue(r["right_blocked"] and r["left_blocked"])

    def test_signals_and_semantics(self):
        sig = g.signals(self.view)
        self.assertTrue(sig["all_vector_names_are_pos"])
        self.assertEqual(sig["keyword_hits_in_columns"], [])
        sem = g.action_semantics(self.view)
        self.assertEqual(sem["lag"]["k_best"], 1)

    def test_timing_checks(self):
        t = g.timing(self.view, decode_videos=False)
        self.assertTrue(t["frame_index_contiguous_from_0"])
        self.assertTrue(t["metadata_length_equals_rows"])
        self.assertLess(t["timestamp_minus_frame_index_over_fps_max_abs_s"], 1e-5)

    def test_fk_geometry_and_aligned_residual_run(self):
        ev = g.episode_events(self.view)
        rows = g.fk_geometry(self.view, ev, "pringles")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["tcp_distance_m"] > 0 for r in rows))
        al = g.event_aligned_residual(self.view, {r["episode"]: r["left_close"] for r in ev}, lag=1, half=20)
        self.assertEqual(al["n"], 3)
        self.assertEqual(len(al["right"]["res_norm_median"]), 41)


if __name__ == "__main__":
    unittest.main()
