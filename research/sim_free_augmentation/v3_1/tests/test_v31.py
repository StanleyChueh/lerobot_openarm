"""v3.1 regression tests (Task B). CPU only.

    cd research/sim_free_augmentation/v3_1
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
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE))
from sfa3 import labels as lb, planner3 as pl, recon3  # noqa: E402
from sfa3.recon3 import rb  # noqa: E402
from sfa31 import protocol  # noqa: E402

V3_WAYPOINT = pl.waypoint_delta
protocol.install()
CFG = json.loads((HERE / "configs/v3_1.json").read_text())
VARS = {v["id"]: v for f in CFG["families"].values() for v in f}
REC = recon3.build(0, with_plate=False, log=lambda *a: None)
ROBOT = rb.Robot()


def plan(vid, cond, s=1.0):
    return pl.plan(REC, ROBOT, pl.make_spec(REC, ROBOT, VARS[vid], s), cond)


class BoundaryStates(unittest.TestCase):
    def test_gplus_and_c_start_from_the_recorded_state(self):
        for vid in ("init_x-6", "init_y-3", "init_x+5_pos_right_up", "pos_right_up", "approach_above"):
            g, c = plan(vid, "G+"), plan(vid, "C")
            np.testing.assert_array_equal(g["state"][0], REC.state[0])
            np.testing.assert_array_equal(c["state"][0], REC.state[0])
            self.assertLess(np.abs(g["state"][2] - c["state"][2]).max(), 0.01, vid)     # first exported row

    def test_v3_gplus_did_move_the_first_state(self):
        """documents the v3 confound the fix removes (Gaussian tails at t = 0)."""
        pl.waypoint_delta = V3_WAYPOINT
        try:
            g = plan("init_x-6", "G+")
        finally:
            protocol.install()
        self.assertGreater(np.abs(g["state"][0] - REC.state[0]).max(), 1e-3)

    def test_targets_unchanged(self):
        g, c = plan("init_x-6", "G+"), plan("init_x-6", "C")
        self.assertTrue(pl.targets_ok(g, CFG["common_target_tolerance"]) and pl.targets_ok(c, CFG["common_target_tolerance"]))
        np.testing.assert_allclose(g["table_new"], c["table_new"])


class LabelProtocol(unittest.TestCase):
    def test_source_passes_hard_and_is_residual_consistent(self):
        src = {"state": REC.state, "action": REC.action, "can": None, "tau": np.arange(REC.T, dtype=float),
               "valid": np.arange(REC.T) <= REC.T - 1 - REC.lead}
        v = protocol.validate_v31(REC, src)
        self.assertTrue(v["ok"])
        self.assertFalse(v["residual_uncertain"])

    def test_hard_gate_still_catches_the_v2_false_negatives(self):
        c = plan("pos_right_up", "C")
        d = copy.deepcopy(c)
        d["action"] = np.where(np.isfinite(c["action"]), REC.action, np.nan)
        self.assertFalse(protocol.validate_v31(REC, d)["ok"])
        for name, k in (("RJ1.pos", 6), ("LJ8.pos", 3)):
            d = copy.deepcopy(c)
            i = rb.NAMES.index(name)
            d["action"][k:, i] = c["action"][:-k, i]
            self.assertFalse(protocol.validate_v31(REC, d)["ok"], name)

    def test_apparent_lead_is_diagnostic_not_a_gate(self):
        """orient_yaw30/C at full scale was rejected in v3 by the apparent-lead rule only."""
        c = plan("orient_yaw30", "C")
        v = protocol.validate_v31(REC, c)
        self.assertTrue(v["lead_diagnostic_flag"])
        self.assertTrue(v["construction"]["ok"] and v["gripper"]["ok"])
        self.assertTrue(v["ok"])


class FrozenV3(unittest.TestCase):
    def test_v3_tree_unchanged(self):
        r = subprocess.run(["git", "status", "--porcelain", "--", str(HERE.parent / "v3" / "sfa3"), str(HERE.parent / "v3" / "configs"),
                            str(HERE.parent / "v3" / "outputs" / "ep0" / "results.json")], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
