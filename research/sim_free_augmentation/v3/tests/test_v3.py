"""v3 regression tests: the Stage-1 corrections, including the v2 review's three label false negatives.

    cd research/sim_free_augmentation/v3
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= ../../../.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa3 import labels as lb, planner3 as pl, recon3, verify3 as v3  # noqa: E402
from sfa3.recon3 import rb  # noqa: E402
from sfa2 import se3  # noqa: E402

CFG = json.loads((HERE / "configs/v3.json").read_text())
VARS = {v["id"]: v for f in CFG["families"].values() for v in f} | {"identity": {"id": "identity"}}
REC = recon3.build(0, with_plate=False, log=lambda *a: None)
ROBOT = rb.Robot()
V = v3.Verifier3(REC, ROBOT)
_P = {}


def plan(vid, cond, s=1.0):
    if (vid, cond, s) not in _P:
        _P[(vid, cond, s)] = pl.plan(REC, ROBOT, pl.make_spec(REC, ROBOT, VARS[vid], s), cond)
    return _P[(vid, cond, s)]


def accepted(p):
    return p["ik"]["ok"] and pl.targets_ok(p, CFG["common_target_tolerance"]) and V(p)["ok"] and lb.validate(REC, p)["ok"]


def delayed(c, name, k):
    d = copy.deepcopy(c)
    i = rb.NAMES.index(name)
    d["action"][k:, i] = c["action"][:-k, i]
    d["action"][:k, i] = c["action"][0, i]
    d["valid"] = np.isfinite(d["action"]).all(1) & c["valid"]
    return d


class LabelFalseNegatives(unittest.TestCase):
    """the three cases that escaped v2's checks (CODEX_SIM_FREE_V2_REVIEW.md) must now be flagged."""

    def test_source_actions_substituted(self):
        c = plan("pos_right_up", "C")
        d = copy.deepcopy(c)
        d["action"] = np.where(np.isfinite(c["action"]), REC.action, np.nan)
        self.assertTrue(lb.validate(REC, c)["ok"])
        self.assertFalse(lb.construction(REC, d)["ok"])

    def test_single_joint_delay(self):
        c = plan("pos_right_up", "C")
        d = delayed(c, "RJ1.pos", 6)
        self.assertFalse(lb.construction(REC, d)["ok"])
        self.assertIn("RJ1.pos", lb.per_joint_check(REC, d)["mismatched_joints"])

    def test_gripper_command_only_delay(self):
        c = plan("pos_right_up", "C")
        d = delayed(c, "LJ8.pos", 3)
        g = lb.gripper_check(REC, d)
        self.assertFalse(g["ok"])
        self.assertFalse(lb.construction(REC, d)["ok"])


class Contracts(unittest.TestCase):
    def test_identity_is_the_source(self):
        for c in pl.CONDITIONS:
            p = plan("identity", c)
            np.testing.assert_array_equal(p["state"], REC.state)
            np.testing.assert_allclose(p["action"][p["valid"]], REC.action[p["valid"]], atol=1e-12)
            self.assertTrue(accepted(p))

    def test_true_zero_order_hold(self):
        p = plan("timing_early", "C")
        zi = np.floor(p["tau"] + 1e-9).astype(int)
        i = rb.NAMES.index("LJ8.pos")
        np.testing.assert_array_equal(p["action"][p["valid"], i], REC.action[zi[p["valid"]], i])

    def test_warp_keeps_gripper_latency(self):
        for vid in ("timing_early", "rot_handover_late6"):
            g = lb.gripper_check(REC, plan(vid, "C"))
            self.assertTrue(g["ok"], (vid, g["problems"]))

    def test_cylinder_support_height(self):
        can = se3.trans([0, 0, 0.5]) @ se3.rot([1, 0, 0], 180 - 20)       # lid up, axis tilted 20 deg from -z
        a = can[:3, 2]
        expect = min(0.5, 0.5 + 0.23 * a[2]) - 0.0375 * np.sqrt(1 - a[2] ** 2)
        self.assertAlmostEqual(v3.cylinder_min_z(can), expect, places=12)
        self.assertLess(v3.cylinder_min_z(can), 0.5 + 0.23 * a[2])        # below v2's bottom-centre value

    def test_hash_changes_with_one_value(self):
        c = plan("pos_right_up", "C")
        d = copy.deepcopy(c)
        d["action"][100, 0] += 1e-9
        self.assertNotEqual(lb.candidate_hash(c), lb.candidate_hash(d))


class Baseline(unittest.TestCase):
    def test_common_tolerance_on_approach(self):
        """v2's G missed the capture target by 8.5-10.5 mm on approach variations; G+ must reach it within 2 mm."""
        for c in pl.CONDITIONS:
            self.assertTrue(pl.targets_ok(plan("approach_above", c), CFG["common_target_tolerance"]))

    def test_regrasp_aware_attachment(self):
        """v2's G failed grasp_higher only through its stale receiver attachment; G+ must pass the 6D checks."""
        v = V(plan("grasp_higher", "G+"))
        self.assertTrue(v["checks"]["attach6d_translation"]["ok"] and v["checks"]["attach6d_spin"]["ok"], v["failed"])

    def test_new_initial_position_moves_the_standing_can_and_the_grasp(self):
        p = plan("init_x+3", "C")
        np.testing.assert_allclose(p["can"][0][:3, 3] - REC.table[:3, 3], [0.03, 0, 0], atol=1e-9)
        m = p["marks"]
        ROBOT.fk(p["q"]["R"][m["grasp"]], p["q"]["L"][m["grasp"]])
        T_new = ROBOT.frame(ROBOT.tcp["R"])
        ROBOT.fk(REC.g["R"][m["grasp"]], REC.g["L"][m["grasp"]])
        T_src = ROBOT.frame(ROBOT.tcp["R"])
        np.testing.assert_allclose(T_new[:3, 3] - T_src[:3, 3], [0.03, 0, 0], atol=2e-3)
        np.testing.assert_array_equal(p["state"][0], REC.state[0])                 # starts from the real configuration
        self.assertTrue(accepted(p))


class FrozenV2(unittest.TestCase):
    def test_v2_tree_unchanged(self):
        r = subprocess.run(["git", "status", "--porcelain", "--", str(HERE.parent / "v2" / "sfa2"), str(HERE.parent / "v2" / "configs"),
                            str(HERE.parent / "v2" / "outputs" / "ep0" / "results.json")], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
