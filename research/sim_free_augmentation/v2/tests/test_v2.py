"""CPU tests for v2 (no GPU, no network, no simulator). Needs the episode-0 SAM2 cache (run_episode.py once).

    cd research/sim_free_augmentation/v2
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ../../../.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa2 import metrics2 as M2, planner as pl, recon, render2, se3, sync, verify as vf  # noqa: E402
from sfa2.recon import rb  # noqa: E402

CFG = json.loads((HERE / "configs/frozen_v2.json").read_text())
VARS = {v["id"]: v for v in CFG["variations"]}
REC = recon.build(0, HERE / "outputs/cache/ep0", with_plate=False, log=lambda *a: None)
ROBOT = rb.Robot()
V = vf.Verifier(REC, ROBOT)
SIG = float(CFG["sigma_G_frames"])
_P = {}


def plan(vid, cond, scale=1.0):
    k = (vid, cond, scale)
    if k not in _P:
        _P[k] = pl.plan(REC, ROBOT, pl.make_spec(REC, ROBOT, VARS[vid], scale), cond, SIG)
    return _P[k]


class SE3(unittest.TestCase):
    def test_power_and_spin(self):
        D = se3.about(se3.rot([0, 0, 1], 40), [0.3, -0.1, 0.2]) @ se3.trans([0.01, 0.02, 0.0])
        np.testing.assert_allclose(se3.power(D, 0.5) @ se3.power(D, 0.5), D, atol=1e-9)
        A = se3.rot([1, 0, 0], 10)
        self.assertAlmostEqual(abs(se3.spin_deg(A[:3, :3], (A @ se3.rot([0, 0, 1], 25))[:3, :3])), 25.0, places=6)
        self.assertAlmostEqual(se3.axis_tilt_deg([0, 0, 1], [0, 1, 1]), 45.0, places=6)


class Reconstruction(unittest.TestCase):
    def test_joint_fit_reduces_closure(self):
        self.assertLess(REC.info["joint_closure"]["translation_mm_median"], 0.5 * REC.info["independent_closure"]["translation_mm_median"])
        self.assertLess(REC.info["joint_closure"]["axis_tilt_deg_median"], 2.0)


class MatchedConditions(unittest.TestCase):
    def test_identity_reproduces_source_in_every_condition(self):
        for c in pl.CONDITIONS:
            p = plan("identity", c)
            np.testing.assert_array_equal(p["state"], REC.state)
            ok = p["valid"]
            np.testing.assert_allclose(p["action"][ok], REC.action[ok], atol=1e-12)
            self.assertTrue(V(p)["ok"])

    def test_all_conditions_reach_the_same_targets(self):
        for c in pl.CONDITIONS:
            for s in "RL":
                self.assertLess(plan("pos_right_up", c)["targets"][s]["target_err_mm"], 4.0)

    def test_regrasp_separates_P_from_C(self):
        vP, vC = V(plan("grasp_spin30", "P")), V(plan("grasp_spin30", "C"))
        self.assertIn("attach6d_spin", vP["failed"])
        self.assertTrue(vC["ok"], vC["failed"])
        vP, vC = V(plan("grasp_higher", "P")), V(plan("grasp_higher", "C"))
        self.assertIn("attach6d_translation", vP["failed"])
        self.assertTrue(vC["ok"], vC["failed"])


class Corruptions(unittest.TestCase):
    def test_verifier_flags_rotation_and_gripper_corruptions(self):
        c = plan("pos_right_up", "C")
        self.assertTrue(V(c)["ok"])
        stL = next(t for t in range(REC.T) if rb.Robot.split(c["state"])["gL"][t] < recon.CLOSED)
        self.assertIn("attach6d_tilt", V(sync.rotate_object(c, 90.0, stL))["failed"])
        self.assertFalse(V(sync.shift_gripper(c, "L", 12))["ok"])      # receiver closes after the giver opens
        self.assertFalse(V(sync.shift_gripper(c, "R", -15))["ok"])     # giver closes before reaching the can

    def test_action_lead_detector(self):
        c = plan("pos_right_up", "C")
        self.assertEqual(sync.action_lead(c["state"], c["action"])["lead"], REC.lead)
        self.assertEqual(sync.action_lead(c["state"], sync.delay_actions(c, 6)["action"])["lead"], REC.lead - 6)


class Synchronisation(unittest.TestCase):
    def test_export_range_trims_unsupported_rows(self):
        a, b = sync.export_range(REC)
        self.assertEqual(a, int(np.ceil(-REC.lag)))
        self.assertEqual(b, REC.T - REC.lead)
        c = plan("timing_early", "C")
        self.assertTrue(np.isfinite(c["action"][a:b]).all())
        self.assertFalse(np.isfinite(c["action"][b:]).any())

    def test_time_warp_moves_gripper_events(self):
        c = plan("timing_early", "C")
        g = rb.Robot.split(c["action"][c["valid"]])
        lc = int(np.argmax(g["gL"] < 0.5))
        self.assertEqual(lc, REC.ev["left_close"] + VARS["timing_early"]["dt_frames"])

    def test_source_frame_map(self):
        """identity: video frame i is rendered from source frame i; a timing warp re-indexes source frames."""
        np.testing.assert_array_equal(render2.source_frame_map(REC, plan("identity", "C")["tau"]), np.arange(REC.T))
        j = render2.source_frame_map(REC, plan("timing_early", "C")["tau"])
        k = VARS["timing_early"]["dt_frames"]
        self.assertLessEqual(abs(int(j[REC.ev["left_close"] + k]) - REC.ev["left_close"]), 1)
        self.assertTrue((np.diff(j) >= 0).all())


class Metrics(unittest.TestCase):
    def test_warp_error_respects_region(self):
        """regression for the v1 bug: the region must restrict the flow-warp error itself."""
        rng = np.random.default_rng(0)
        F = np.repeat(rng.integers(0, 255, (1, 40, 50, 3), np.uint8), 3, axis=0)
        F[1:, :, 25:] = rng.integers(0, 255, (2, 40, 25, 3), np.uint8)     # right half changes, left half static
        z = np.zeros((2, 40, 50, 2), np.float32)
        left = np.zeros((3, 40, 50), bool)
        left[:, :, :20] = True
        r = M2.warp_error_regions(F, {"fwd": z, "bwd": z}, {"left": left, "right": ~left})
        self.assertEqual(r["left"]["mean"], 0.0)
        self.assertGreater(r["right"]["mean"], 20.0)


if __name__ == "__main__":
    unittest.main()
