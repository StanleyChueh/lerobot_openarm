"""pipeline v1 blend_space='joint': exact IK only on full-offset frames, joint-space blend elsewhere (plan.py docstring)."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pinocchio as pin
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfagen.plan import _retarget, joint_blend_ik, smoothstep


def toy_rec(T=40):
    """3 prismatic joints (x, y, z) -> TCP = q; enough for exact position IK (orientation is constant)."""
    m = pin.Model(); j = 0
    for name, jm in (("x", pin.JointModelPX()), ("y", pin.JointModelPY()), ("z", pin.JointModelPZ())):
        j = m.addJoint(j, jm, pin.SE3.Identity(), name)
    fid = m.addFrame(pin.Frame("tcp", j, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME)); data = m.createData()
    r = SimpleNamespace(m=m, d=data, tcp={"R": fid}, vidx={"R": [0, 1, 2]}, arms=["R"], sidx={"R": [0, 1, 2]},
                        lo={"R": -np.ones(3)}, hi={"R": np.ones(3)})
    r.q_full = lambda q: np.asarray(q["R"]); r.split = lambda S: {"R": S.copy()}
    r.fk = lambda q: pin.framesForwardKinematics(m, data, np.asarray(q["R"])); r.frame = lambda f: data.oMf[f].homogeneous.copy()
    r.tcp_traj = lambda q, arm: np.stack([(r.fk({arm: row}), r.frame(fid))[1] for row in q[arm]])
    S = np.c_[np.linspace(0, .3, T), np.zeros(T), np.linspace(.2, 0, T)]
    return SimpleNamespace(robot=r, state=S, action=S.copy())


class BlendSpaceTests(unittest.TestCase):
    def setUp(self):
        self.rec = toy_rec(); t = np.arange(40.)
        self.W = np.minimum(smoothstep((t - 2) / 12), 1 - smoothstep((t - 26) / 10)); self.d = np.array([.05, -.03, 0.])

    def test_joint_blend(self):
        S, W, d = self.rec.state, self.W, self.d
        Sn, An, q, err, _ = _retarget(self.rec, W, d, ["R"], blend_space="joint")
        self.assertEqual(Sn.shape, S.shape)                                    # length unchanged (no retiming)
        self.assertTrue(np.array_equal(Sn[W <= 1e-9], S[W <= 1e-9]))           # untouched frames bit-identical (start)
        hold = W >= .999
        self.assertLess(np.abs(Sn[hold] - (S[hold] + d)).max(), 1e-4)         # exact (ik() tolerance 0.1 mm) at full offset
        self.assertLess(err[hold, 0].max(), 1e-4); self.assertEqual(err[~hold].max(), 0.0)
        dq = Sn - S; v = np.diff(dq, axis=0)                                   # Hermite edges: velocity continuous, no kink
        self.assertLess(np.abs(np.diff(v, axis=0)).max(), .01)
        self.assertLess(np.abs(v[0]).max(), 1e-3)                              # leaves the source with ~zero offset velocity
        Sc, *_ = _retarget(self.rec, W, d, ["R"])                              # default stays the Cartesian blend
        Sc2, *_ = _retarget(self.rec, W, d, ["R"], blend_space="cartesian")
        self.assertTrue(np.array_equal(Sc, Sc2))

    def test_exact_from(self):
        r, S, W, d = self.rec.robot, self.rec.state, self.W, self.d
        TT = r.tcp_traj({"R": S}, "R"); lo, hi = -np.ones((40, 3)), np.ones((40, 3))
        q, exact = joint_blend_ik(r, "R", TT, W, np.tile(d, (40, 1)) * W[:, None], S, {"R": S}, lo, hi, exact_from=20)
        self.assertEqual(int(np.flatnonzero(exact)[0]), 20)                    # exact segment starts at max(hold, 20)
        self.assertLess(np.abs(q[exact] - (S[exact] + d)).max(), 1e-4)
        self.assertTrue(np.array_equal(q[W <= 1e-9], S[W <= 1e-9]))

    def test_no_full_offset_frame(self):
        r = self.rec.robot; W = .5 * self.W
        self.assertIsNone(joint_blend_ik(r, "R", r.tcp_traj({"R": self.rec.state}, "R"), W, W[:, None] * self.d,
                                         self.rec.state, {"R": self.rec.state}, -np.ones((40, 3)), np.ones((40, 3))))
        Sj, *_ = _retarget(self.rec, W, self.d, ["R"], blend_space="joint")    # falls back to the Cartesian blend
        Sc, *_ = _retarget(self.rec, W, self.d, ["R"])
        self.assertTrue(np.array_equal(Sj, Sc))


if __name__ == "__main__":
    unittest.main()
