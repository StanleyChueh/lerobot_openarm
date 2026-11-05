"""Reproducible CPU tests for the generation-feasibility dry run (no simulation, no GPU, no network).

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python -m unittest discover -s research/generation_feasibility/tests -v
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

from gf import checks, config as C, descriptors as D, kin, plans, sim_metrics as M, sources  # noqa: E402

K = kin.Kin()
SIM10 = [plans.segment_source(e, K) for e in sources.load_sim10()]
GT = sources.load_gt_hdf5("V9") + sources.load_gt_hdf5("stanley")
ZERO = {**{k: 0.0 for k in C.B_RANGES}, "approach_speed": 1.0, "bump_angle": 0.0, "receiver_dwell": 0, "release_delay": 0}


class Kinematics(unittest.TestCase):
    def test_fk_matches_simulator_grasp_frames(self):
        """URDF FK + 2.4 mm reproduces the recorded sim grasp frames (pre-step obs[t+1] vs post-step state[t])."""
        for e in GT[:3]:
            for s in "RL":
                P = K.grasp_traj(s, e.q[s][:-1])
                err = np.linalg.norm(P[:, :3, 3] - e.eef[s][1:, :3, 3], axis=1)
                self.assertLess(err.max(), 1e-3)
                R_err = [np.linalg.norm(a[:3, :3] - b[:3, :3]) for a, b in zip(P[::25], e.eef[s][1::25])]
                self.assertLess(max(R_err), 1e-3)

    def test_ik_round_trip_and_limits(self):
        e = SIM10[0].ep
        for t in (0, 120, 200):
            T = K.grasp_pose("L", e.q["L"][t])
            q, pe, re = K.ik("L", T, e.q["L"][0])
            self.assertLess(pe, 1e-3)
            self.assertLess(re, 1.0)
            self.assertTrue(np.all(q >= K.lo["L"]) and np.all(q <= K.hi["L"]))


class BaselineA(unittest.TestCase):
    def test_A_translates_only_the_right_approach(self):
        s = SIM10[4]
        can = s.ep.can0 + np.array([0.03, -0.02, 0.0])
        p = plans.plan_A(s, can)
        m = p.marks
        # right approach (segment 0) moved by the can offset; after the grasp seam: untransformed source poses
        d_close = p.target["R"][m["R_close"]][:3, 3] - plans.smooth_positions(s.TR, C.SMOOTH_WINDOW)[s.t_rc][:3, 3]
        self.assertLess(np.linalg.norm(d_close[:2] - [0.03, -0.02]), 0.006)
        tail = p.target["R"][m["R_release_cmd"]][:3, 3] - s.TR[s.t_rr][:3, 3]
        self.assertLess(np.linalg.norm(tail), 0.01)
        # the whole left arm is the source, delayed by LEFT_INTERP
        L = plans.smooth_positions(s.TL, C.SMOOTH_WINDOW)
        dl = np.linalg.norm(p.target["L"][C.LEFT_INTERP:C.LEFT_INTERP + s.ep.T, :3, 3] - L[:, :3, 3], axis=1)
        self.assertLess(dl.max(), 1e-9)

    def test_A_replays_the_handover(self):
        """A's receiver path is the source's: path novelty << 1 for every pilot pose."""
        SD = [checks.source_descriptor(s) for s in SIM10]
        sc = D.source_scale(SD)
        for pose in json.loads(C.EVAL_POSES.read_text())["poses"][:C.PILOT_N]:
            can = np.array([pose["x"], pose["y"], C.CAN_REST_Z])
            p = plans.plan_A(SIM10[plans.nearest_source(SIM10, can)], can)
            nv = D.novelty(checks.plan_descriptor(p), SD, sc)
            self.assertLess(nv["path"]["ratio"], 0.5)
            self.assertFalse(D.outside_envelope(checks.plan_descriptor(p), D.envelope(SD))["novel"])

    def test_zero_parameter_B_equals_A(self):
        for s in SIM10[:4]:
            a = plans.plan_A(s, s.ep.can0)
            b = plans.plan_phase(s, s.ep.can0, ZERO, "B")
            for k in "RL":
                np.testing.assert_allclose(a.target[k], b.target[k], atol=1e-12)
                np.testing.assert_array_equal(a.grip[k], b.grip[k])
            self.assertEqual(a.marks["L_close"], b.marks["L_close"])


class ConditionC(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_C_constraints_hold_on_accepted_plans(self):
        n_acc = 0
        for i in range(8):
            s = SIM10[i % 10]
            prm = plans.sample_params(np.random.default_rng(100 + i))
            p, r, tried = checks.make_C(s, s.ep.can0, prm, K, i)
            if p.status == "rejected":
                continue
            n_acc += 1
            self.assertTrue(r["all_ok_C"])
            self.assertEqual(r["speed_violations"], 0)
            self.assertTrue(r["receiver_jaw_capture"])
            self.assertGreaterEqual(p.marks["L_close"], p.marks["giver_arrive"])
            # the giver holds until the receiver has closed + overlap (synchronisation 2)
            self.assertGreaterEqual(p.marks["R_release_cmd"], p.marks["L_close"] + C.RELEASE_OVERLAP)
            self.assertIsNotNone(p.runtime["receiver_reanchor"])
            self.assertTrue(p.runtime["receiver_close_gate"])
        self.assertGreater(n_acc, 4)

    def test_reanchor_model_tolerates_attachment_error_open_loop_does_not(self):
        s = SIM10[0]
        prm = plans.sample_params(np.random.default_rng(5))
        pC, _, _ = checks.make_C(s, s.ep.can0, prm, K)
        pB = plans.plan_phase(s, s.ep.can0, prm, "B")
        self.assertGreater(checks.attachment_sensitivity(pC, s, self.rng), 0.9)
        self.assertLess(checks.attachment_sensitivity(pB, s, self.rng), checks.attachment_sensitivity(pC, s, self.rng))

    def test_retime_keeps_arms_synchronised(self):
        s = SIM10[4]           # this source itself exceeds the 2.5 rad/s cap
        p = plans.plan_phase(s, s.ep.can0, ZERO, "C")
        before = p.marks["L_close"] - p.marks["giver_arrive"]
        n = checks.retime(p, s, K)
        self.assertGreater(n, 0)
        J = checks.solve_joints(p, s, K)
        self.assertLessEqual(max(np.abs(np.diff(J[k]["q"], axis=0)).max() for k in "RL"), C.STEP_CAP * 1.001)
        self.assertLessEqual(abs((p.marks["L_close"] - p.marks["giver_arrive"]) - before), n)


class SimMetrics(unittest.TestCase):
    def test_replica_matches_recorded_success_and_sources_are_stable(self):
        for e in GT:
            m = M.rollout_metrics(e)
            self.assertEqual(m["success_final"], e.meta["success_attr"], e.name)
            self.assertTrue(m["stable_success"], e.name)

    def _ep(self):
        return copy.deepcopy(GT[0])

    def test_dropped_can_is_not_stable(self):
        e = self._ep()
        m0 = M.rollout_metrics(e)
        t = m0["t_giver_release"] + 5
        e.can[t:, 2] = C.CAN_REST_Z          # can back on the pad, hands still where they were
        e.can[t:, 0] += 0.2                  # and away from the receiver
        self.assertFalse(M.rollout_metrics(e)["stable_success"])

    def test_receiver_shut_on_air_is_not_success(self):
        e = self._ep()
        m0 = M.rollout_metrics(e)
        e.grip_state["L"][m0["t_pass"]:] = 0.0     # jaws fully shut: aperture below the receiver range
        m = M.rollout_metrics(e)
        self.assertFalse(m["success_final"])
        self.assertFalse(m["stable_success"])

    def test_giver_never_releasing_is_not_success(self):
        e = self._ep()
        m0 = M.rollout_metrics(e)
        t = m0["t_pass"]
        e.grip_state["R"][t:] = e.grip_state["R"][t]          # right jaws stay shut on the can
        e.eef["R"][t + 1:] = e.eef["R"][t + 1]                # right hand stays at the pass ...
        e.can[t:] = e.can[t]                                  # ... and the can stays in it; the left leaves without it
        self.assertFalse(M.rollout_metrics(e)["stable_success"])

    def test_executed_descriptor_on_a_recorded_episode(self):
        e = GT[0]
        d = M.executed_descriptor(e, M.rollout_metrics(e))
        self.assertIsNotNone(d)
        self.assertGreater(d["axial_L"], d["axial_R"])       # receiver above the giver on the can (GT)
        self.assertLess(d["radial_L"], C.EEF_TO_CAN_RADIAL)


class Protocol(unittest.TestCase):
    def test_manifest_pairs_conditions(self):
        mf = HERE / "outputs/dry_run/pilot_manifest.json"
        if not mf.exists():
            self.skipTest("run scripts/run_dry_run.py first")
        m = json.loads(mf.read_text())
        pilot = [c for c in m["candidates"] if c["stage"] == "pilot"]
        by = {}
        for c in pilot:
            by.setdefault(c["i"], {})[c["cond"]] = c
        self.assertEqual(len(by), C.PILOT_N)
        for i, d in by.items():
            self.assertEqual({d[k]["trial"] for k in "ABC"}.__len__(), 1)
            self.assertEqual({d[k]["source"] for k in "ABC"}.__len__(), 1)
            for key in ("pass_dx", "pass_dy", "pass_dz", "grasp_azimuth_deg", "grasp_axial", "approach_bump"):
                self.assertAlmostEqual(d["B"]["params"][key] * d["C"]["params"]["scale"], d["C"]["params"][key])
            for key in ("receiver_dwell", "release_delay", "approach_speed", "bump_angle"):
                self.assertEqual(d["B"]["params"][key], d["C"]["params"][key])
        smoke = {c["trial"] for c in m["candidates"] if c["stage"] == "smoke"}
        self.assertFalse(smoke & {c["trial"] for c in pilot})

    def test_preflight_refuses_without_approval(self):
        r = subprocess.run([sys.executable, str(HERE / "scripts/pilot_preflight.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("REFUSED", r.stderr)

    def test_gate1r_verifier_frozen(self):
        """The failed Gate 1R line is not modified by this experiment."""
        diff = subprocess.run(["git", "diff", "--stat", "277906945c7c3f2cf234bfd98ece80051e3b19c6", "--",
                               "research/contact_aware_augmentation"], cwd=C.REPO, capture_output=True, text=True)
        self.assertEqual(diff.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
