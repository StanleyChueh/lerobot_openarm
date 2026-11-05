"""CPU unit tests for the simulation-free augmentation prototype (no GPU, no network, no simulator).

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover \
        -s research/sim_free_augmentation/tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from sfa import data, metrics as M, plan as pl, recover as rc, render as rd, robot as rb  # noqa: E402

EP = data.load(0, cams=())
R, CAM = rb.Robot(), rb.Camera()
G = rb.Robot.split(EP.state)
EV = EP.events


def _X(p, axis=(1.0, 0, 0)):
    import run_pipeline as rp
    return rp.frame_X({"p": np.asarray(p, float), "axis": np.asarray(axis, float)})


class Formats(unittest.TestCase):
    def test_dataset_conventions(self):
        self.assertEqual(EP.state.shape[1], 16)
        x = rb.Robot.merge(G["R"], G["L"], EP.state[:, rb.NAMES.index("RJ8.pos")], EP.state[:, rb.NAMES.index("LJ8.pos")])
        np.testing.assert_allclose(x, EP.state)
        self.assertEqual(EV, {"right_close": 97, "left_close": 176, "right_open": 181})
        self.assertEqual(data.action_lead(EP.state, EP.action)["k_best"], 2)

    def test_link_sides(self):
        """regression: 'openarm_left_right_finger' belongs to the LEFT arm (substring test used to say right)."""
        sides = {n: s for n, s, H, S in R.links}
        self.assertEqual(sides["left_right_finger"], "L")
        self.assertEqual(sides["right_left_finger"], "R")


class Kinematics(unittest.TestCase):
    def test_ik_round_trip(self):
        R.fk(G["R"][150], G["L"][150])
        T = R.frame(R.tcp["R"])
        q, pe, re = R.ik("R", T, G["R"][100])
        self.assertLess(pe, 1e-3)
        self.assertLess(re, 0.5)

    def test_fit_point_recovers_a_synthetic_attachment(self):
        p_true = np.array([0.12, 0.01, -0.05])
        fr = list(range(110, 170))
        uv = np.full((EP.T, 2), np.nan)
        for t in fr:
            T = rc.tcp_pose(R, G, "R", t - 1.5)
            uv[t] = CAM.project((T[:3, :3] @ p_true + T[:3, 3])[None])[0][0]
        f = rc.fit_point(R, CAM, G, "R", fr, uv, -1.5)
        self.assertLess(np.linalg.norm(f["p"] - p_true), 1e-3)
        self.assertLess(f["rms_px"], 0.1)


class ContactModel(unittest.TestCase):
    X = {"R": _X([0.157, 0.02, -0.063]), "L": _X([0.093, 0.014, -0.044])}
    table = rc.tcp_pose(R, G, "R", EV["right_close"] + 8) @ X["R"]

    def plan(self, kind, delta=(0.0, -0.045, 0.04)):
        return pl.make_plan(R, G, EV, self.X, self.table, 30.0, kind, delta, seed=1)

    def test_contact_plan_satisfies_the_model(self):
        p = self.plan("contact")
        c = p.checks
        self.assertTrue(c["all_ok"], c)
        self.assertLess(c["receiver_capture_dev_m"], 1e-3)
        self.assertEqual(c["can_moved_before_grasp_m"], 0.0)
        # both hands carry the SAME offset during dual contact (transfer coordination)
        for t in range(EV["left_close"], EV["right_open"]):
            np.testing.assert_allclose(p.offsets["R"][t], p.offsets["L"][t], atol=1e-9)

    def test_geometry_only_violates_attachment(self):
        c = self.plan("geometry").checks
        self.assertFalse(c["attachment_ok"])
        self.assertFalse(c["can_continuous"])
        self.assertGreater(c["receiver_capture_dev_m"], 0.05)

    def test_zero_offset_reproduces_the_recording(self):
        p = self.plan("contact", (0.0, 0.0, 0.0))
        for s in "RL":
            np.testing.assert_allclose(p.q[s], G[s])
        a = pl.actions(G, rb.Robot.split(EP.action), EP.action, p, 2)
        np.testing.assert_allclose(a, EP.action)

    def test_candidate_actions_keep_the_command_state_relation(self):
        p = self.plan("contact")
        a = pl.actions(G, rb.Robot.split(EP.action), EP.action, p, 2)
        c = pl.action_checks(R, a, p.q, 2)
        c0 = pl.action_checks(R, EP.action, {"R": G["R"], "L": G["L"]}, 2)
        self.assertTrue(c["contracts_ok"])
        self.assertAlmostEqual(c["max_cmd_vs_state_at_lead_rad"], c0["max_cmd_vs_state_at_lead_rad"], places=9)
        np.testing.assert_array_equal(a[:, rb.NAMES.index("RJ8.pos")], EP.action[:, rb.NAMES.index("RJ8.pos")])


class Rendering(unittest.TestCase):
    def test_identity_edit_returns_the_frame(self):
        rng = np.random.default_rng(0)
        fr = rng.integers(0, 255, (480, 640, 3), np.uint8)
        masks = {k: np.zeros((480, 640), np.uint8) for k in ("R", "L", "can")}
        masks["R"][200:300, 400:500] = 1
        q = {s: G[s][150] for s in "RL"}
        out, nm, info = rd.render_frame(150, fr, fr * 0, R, CAM, q, q, {"R": 0.5, "L": 1.0}, masks, None, None, [0, 0, 1])
        np.testing.assert_array_equal(out, fr)
        self.assertEqual(info["n_layers"], 0)

    def test_homography_of_a_translation(self):
        src = np.random.default_rng(1).uniform(0, 600, (30, 2))
        H = rd.homography(src, src + [12.0, -7.0])
        np.testing.assert_allclose(H @ np.r_[src[0], 1.0], np.r_[src[0] + [12.0, -7.0], 1.0], atol=1e-6)

    def test_follow_error_and_iou(self):
        a = np.zeros((5, 3, 2))
        b = a + [3.0, 4.0]
        e = M.follow_error(b, np.ones((5, 3), bool), a, np.ones((5, 3), bool))
        self.assertAlmostEqual(e["median_px"], 5.0)
        m = np.zeros((10, 10), bool)
        m[:5] = True
        self.assertAlmostEqual(M.mask_iou(m, m), 1.0)


if __name__ == "__main__":
    unittest.main()
