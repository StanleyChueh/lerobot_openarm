"""Replay-check gate on a real OpenArm source demo (CPU): the unchanged demo passes; a joint over its limit and an arm
3 cm below the real table envelope fail. Needs the Pringles source dataset in the local Hugging Face cache."""
import sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from sfagen.context import Context
from sfagen.checks import Checker


class ReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx = Context(HERE / "configs/openarm_pringles_handover.yaml", work_dir=tempfile.mkdtemp())
        cls.r = ctx.robot; cls.S0, cls.A0 = ctx.src.arrays(0); cls.S0, cls.A0 = cls.S0[::2], cls.A0[::2]  # every 2nd frame: faster
        ctx.src.arrays = lambda e: (cls.S0, cls.A0); cls.ck = Checker(ctx); cls.ck.envelope([0])
        T = len(cls.S0); cls.rec = SimpleNamespace(robot=cls.r, state=cls.S0, action=cls.A0, objects=[], T=T, obj_pose=lambda k, t: None)

    def cand(self, S, A, moved=()):
        T = len(S)
        return {"state": S, "action": A, "edited": np.ones(T, bool), "w": np.zeros(T), "ik_err": np.zeros((T, 2)), "moved_arms": list(moved),
                "t_rejoin": T - 1, "objects": [], "family": "interaction_location", "lead": 0}

    def test_unchanged_source_passes(self):
        out = self.ck.check(self.rec, self.cand(self.S0.copy(), self.A0.copy()))
        self.assertTrue(out["pass"], out)

    def test_joint_over_limit_fails(self):
        arm = self.r.arms[0]; j = self.r.sidx[arm][0]; S, A = self.S0.copy(), self.A0.copy(); t = len(S) // 2
        top = max(self.r.hi[arm][0], S[:, j].max(), A[:, j].max()) + 0.1
        S[t, j] = A[t, j] = top
        out = self.ck.check(self.rec, self.cand(S, A, moved=[arm]))
        self.assertFalse(out["pass"]); self.assertGreater(out["limit_violations"], 0)

    def test_arm_below_table_envelope_fails(self):
        env = dict(self.ck.env); self.ck.env["min_z_m"] = env["min_z_m"] + 0.03   # the demo now lies 3 cm below the envelope
        try:
            out = self.ck.check(self.rec, self.cand(self.S0.copy(), self.A0.copy()))
        finally:
            self.ck.env = env
        self.assertFalse(out["pass"]); self.assertLess(out["lowest_arm_z_m"], out["real_lowest_arm_z_m"] - 0.01)


if __name__ == "__main__":
    unittest.main()
