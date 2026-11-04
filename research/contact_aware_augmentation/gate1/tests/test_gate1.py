"""Offline tests for the Gate 1 pilot code (synthetic data; no dataset, network, GPU or robot).

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python -m unittest discover -s research/contact_aware_augmentation/gate1/tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "annotation")]

import annotate_gate1 as ag  # noqa: E402
import evaluate_annotations as ea  # noqa: E402
import gate1_core as gc  # noqa: E402
import lid_tracker as lt  # noqa: E402
from gate1_core import g0  # noqa: E402
from reconcile_eval import cmh  # noqa: E402


def rot(axis, ang):
    from scipy.spatial.transform import Rotation as R
    return R.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * ang).as_matrix()


def synthetic_hands(T=200, seed=0):
    """Two smooth TCP trajectories in front of the body camera (world frame), with rotation and depth change."""
    t = np.linspace(0, 1, T)
    out = {}
    for h, (x0, y0) in (("right", (0.25, -0.05)), ("left", (0.22, 0.12))):
        P = np.zeros((T, 4, 4))
        for i, s in enumerate(t):
            # rotation and depth change comparable to the real carry (the real giver turns the can toward the receiver)
            P[i, :3, :3] = rot([0.3, 1, 0.2], 2.5 * s) @ rot([1, 0, 0], np.pi * 0.6)
            P[i, :3, 3] = [x0 + 0.12 * np.sin(2 * np.pi * s), y0 + 0.08 * np.sin(np.pi * s), 0.30 + 0.15 * s]
            P[i, 3, 3] = 1
        out[h] = P
    return out


def synthetic_episode(T=200, events=(40, 120, 126), noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Th = synthetic_hands(T)
    rc, lc, ro = events
    pR, pL = np.array([0.12, 0.02, -0.05]), np.array([0.08, 0.015, -0.04])
    uv = np.zeros((T, 2))
    # object rigid with the right hand from rc, then with the left hand from ro; make the two consistent at ro by
    # choosing pL so that both predict the same world point at ro
    Xro = Th["right"][ro, :3, :3] @ pR + Th["right"][ro, :3, 3]
    pL = Th["left"][ro, :3, :3].T @ (Xro - Th["left"][ro, :3, 3])
    X0 = Th["right"][rc, :3, :3] @ pR + Th["right"][rc, :3, 3]
    for t in range(T):
        if t < rc:
            X = X0
        elif t < ro:
            X = Th["right"][t, :3, :3] @ pR + Th["right"][t, :3, 3]
        else:
            X = Th["left"][t, :3, :3] @ pL + Th["left"][t, :3, 3]
        uv[t] = g0.project(X[None])[0][0] + rng.normal(0, noise, 2)
    E = gc.Episode(0, np.zeros((T, 16)), np.zeros((T, 16)), [], Th, {"right_close": rc, "left_close": lc, "right_open": ro},
                   uv, np.ones(T, bool))
    return E, pR, pL


class Tracker(unittest.TestCase):
    def test_tracks_moving_lid_and_flags_border(self):
        T, H, W = 40, 480, 640
        frames = np.zeros((T, H, W, 3), np.uint8)
        frames[...] = (30, 30, 40)
        import cv2
        centres = [(200 + 6 * t, 200 + 2 * t) for t in range(T - 5)] + [(636, 300)] * 5  # last 5 at the border
        for t, c in enumerate(centres):
            col = cv2.cvtColor(np.uint8([[[22, 90, 230]]]), cv2.COLOR_HSV2RGB)[0, 0].tolist()
            cv2.circle(frames[t], c, 40, col, -1)
        tr = lt.track(frames, centres[0])
        err = np.linalg.norm(tr.uv[: T - 5] - np.array(centres[: T - 5]), axis=1)
        self.assertLess(err.max(), 1.0)
        self.assertTrue(all(s == "ok" for s in tr.status[: T - 5]))
        self.assertTrue(all(s != "ok" for s in tr.status[T - 5:]))

    def test_lost_when_nothing_matches(self):
        frames = np.zeros((3, 100, 100, 3), np.uint8)
        tr = lt.track(frames, (50, 50))
        self.assertEqual(tr.status, ["lost"] * 3)
        self.assertTrue(np.isnan(tr.uv).all())


class RigidModel(unittest.TestCase):
    def test_fit_recovers_point_and_rigid_rotation_is_not_flagged(self):
        Th = synthetic_hands()
        p = np.array([0.1, 0.02, -0.04])
        uv = gc.project_points(Th["right"], p, g0.BODY_CAM)
        self.assertTrue(((uv > 0) & (uv < [640, 480])).all(), "synthetic points must be in the image")
        f = gc.fit_point(Th["right"], uv, g0.BODY_CAM)
        self.assertLess(f["rms"], 1e-3)
        self.assertLess(np.linalg.norm(f["p"] - p), 1e-4)
        # the Gate-0 M2 drift of this perfectly rigid point is large (Codex counterexample 2)
        tcp = g0.project(Th["right"][:, :3, 3])[0]
        m2 = np.max(np.linalg.norm((uv - tcp) - (uv[0] - tcp[0]), axis=1))
        self.assertGreater(m2, 20.0)

    def test_constant_image_offset_is_not_rigid(self):
        Th = synthetic_hands()
        p = np.array([0.1, 0.02, -0.04])
        uv = gc.project_points(Th["right"], p, g0.BODY_CAM) + np.array([20.0, 0])
        f = gc.fit_point(Th["right"], uv, g0.BODY_CAM)
        self.assertGreater(np.linalg.norm(f["p"] - p), 0.005)  # absorbed partly into a different point ...

    def test_temporal_shift_raises_residual(self):
        E, _, _ = synthetic_episode()
        s = gc.sample_from_episode(E)
        base = gc.score_C(s, g0.BODY_CAM, 180)
        sh = gc.sample_from_episode(E)
        sh.uv = E.uv[np.clip(np.arange(len(E.uv)) - 6, 0, None)]
        shifted = gc.score_C(sh, g0.BODY_CAM, 180)
        self.assertLess(base["res_giver"], 1.5)
        self.assertGreater(shifted["res_giver"], 3 * base["res_giver"])


class Phases(unittest.TestCase):
    def test_giver_is_first_closing_hand(self):
        self.assertEqual(gc.phase_structure({"right_close": 10, "left_close": 50, "right_open": 55})["giver"], "right")
        self.assertEqual(gc.phase_structure({"right_close": 60, "left_close": 50, "right_open": 70, "left_open": 65})["giver"], "left")

    def test_phase_logic(self):
        E, _, _ = synthetic_episode()
        s = gc.sample_from_episode(E)
        self.assertTrue(gc.score_E(s)["phase_logic_ok"])
        s.events = dict(s.events, right_open=s.events["left_close"] - 10)
        self.assertFalse(gc.score_E(s)["phase_logic_ok"])


class Injection(unittest.TestCase):
    def setUp(self):
        self.E, self.pR, self.pL = synthetic_episode()
        self.samples = gc.inject(self.E, {"giver": self.pR, "recv": self.pL}, g0.BODY_CAM)

    def test_variants_and_labels(self):
        names = {s.name for s in self.samples}
        for k in (3, 6, 10):
            self.assertIn(f"time_lag_{k}", names)
            self.assertIn(f"time_lead_{k}", names)
        for cm in (1, 2, 4):
            self.assertIn(f"disp_world_x_{cm}cm", names)
            self.assertIn(f"disp_hand_jaw_y_{cm}cm", names)
        self.assertIn("transfer_failure_stays_with_giver", names)
        self.assertTrue(all(not s.valid for s in self.samples))

    def test_displacements_keep_ok_mask_and_have_no_step(self):
        ro = self.E.events["right_open"]
        for s in self.samples:
            if s.family in ("displacement_world", "displacement_hand", "slip", "displacement_px"):
                self.assertTrue(np.array_equal(s.ok, self.E.ok), s.name)
                d = s.uv - self.E.uv
                step = np.linalg.norm(np.diff(d, axis=0), axis=1)
                # the injected delta changes smoothly: no frame-to-frame jump larger than the ramp slope allows
                self.assertLess(step.max(), 12.0, s.name)
                if s.family == "displacement_hand":
                    self.assertLess(np.linalg.norm(d[ro] - d[ro - 1]), 3.0, s.name)

    def test_injected_displacement_is_detected_by_relative_model_only_when_not_rigid(self):
        tail = len(self.E.uv) - 11
        world = next(s for s in self.samples if s.name == "disp_world_x_4cm")
        hand = next(s for s in self.samples if s.name == "disp_hand_jaw_y_4cm")
        base = gc.score_C(gc.sample_from_episode(self.E), g0.BODY_CAM, tail)
        cw, ch = gc.score_C(world, g0.BODY_CAM, tail), gc.score_C(hand, g0.BODY_CAM, tail)
        self.assertGreater(cw["res_giver"], 2 * base["res_giver"])  # world-constant offset is not hand-rigid
        self.assertLess(ch["res_giver"], 2 * base["res_giver"] + 0.5)  # hand-rigid offset is (by construction)


class Thresholds(unittest.TestCase):
    def test_freeze_and_score(self):
        dev = [{"C.res_giver": 2.0, "C.res_recv": 1.0, "C.takeover_px": 5.0, "C.abs_giver_mm": 4.0, "C.abs_recv_mm": 4.0,
                "C.release_px": 40.0, "C.phase_logic_ok": True},
               {"C.res_giver": 2.5, "C.res_recv": 1.5, "C.takeover_px": 6.0, "C.abs_giver_mm": 5.0, "C.abs_recv_mm": 5.0,
                "C.release_px": 50.0, "C.phase_logic_ok": True}]
        th = gc.freeze_thresholds(dev)
        self.assertAlmostEqual(th["C.res_giver"]["tau"], max(1.5 * 2.5, gc.FLOORS["res_giver"]))
        self.assertAlmostEqual(th["C.release_px"]["tau"], 40.0 / 1.5)
        ok = dict(dev[0])
        self.assertFalse(gc.method_score(ok, th, "C")[1])
        bad = dict(dev[0], **{"C.res_giver": 20.0})
        s, flag, why = gc.method_score(bad, th, "C")
        self.assertTrue(flag)
        self.assertIn("C.res_giver", why)
        logic = dict(dev[0], **{"C.phase_logic_ok": False})
        self.assertTrue(gc.method_score(logic, th, "C")[1])

    def test_auc(self):
        self.assertEqual(gc.auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(gc.auc([1], [1]), 0.5)
        self.assertEqual(gc.auc([0], [1, 2]), 0.0)


class Annotation(unittest.TestCase):
    def test_manifest_deterministic_and_excludes_sealed(self):
        ev = {e: {"right_close": 90, "left_close": 170, "right_open": 175} for e in ag.PILOT}
        ln = {e: 250 for e in ag.PILOT}
        a, b = ag.manifest_rows(ev, ln), ag.manifest_rows(ev, ln)
        self.assertEqual(a, b)
        self.assertFalse({r["episode"] for r in a} & ag.SEALED)
        self.assertTrue(any(r["repeat"] == 1 for r in a))

    def test_session_state_machine(self):
        man = [{"episode": 0, "frame": 5, "purpose": "key", "repeat": 0}, {"episode": 0, "frame": 6, "purpose": "key", "repeat": 0}]
        S = ag.Session("T", man)
        self.assertFalse(S.next())  # incomplete frame cannot be skipped
        S.click(10, 20)
        S.mark_uncertain_next(); S.click(11, 21)
        S.occluded()
        S.undo(); S.occluded()
        S.click(1, 2); S.click(3, 4)
        self.assertTrue(S.complete())
        self.assertTrue(S.next())
        rows = S.rows()
        self.assertEqual([r["visibility"] for r in rows], ["visible", "uncertain", "occluded", "visible", "visible"])
        S2 = ag.Session("T", man, rows)  # resume
        self.assertEqual(S2.i, 1)

    def test_ground_truth_rule(self):
        def r(a, k, vis, u, v, rep=0):
            return {"annotator": a, "episode": "0", "frame": "5", "repeat": str(rep), "keypoint": k, "visibility": vis, "u": u, "v": v}
        rows = [r("A", "lid_center", "visible", 100, 100), r("B", "lid_center", "visible", 103, 100),   # agree -> GT
                r("A", "right_jaw_mid", "visible", 100, 100), r("B", "right_jaw_mid", "visible", 120, 100),  # disagree
                r("A", "left_jaw_mid", "uncertain", 100, 100), r("B", "left_jaw_mid", "visible", 100, 100),  # uncertain
                r("A", "lid_center", "visible", 101, 100, rep=1)]
        gt, st = ea.ground_truth(rows)
        self.assertEqual(set(gt), {(0, 5, "lid_center")})
        self.assertTrue(np.allclose(gt[(0, 5, "lid_center")], [101.5, 100]))
        self.assertEqual(st["excluded_disagreement"], 1)
        self.assertEqual(st["excluded_uncertain_or_occluded"], 1)
        rep = ea.repeatability(rows)
        self.assertEqual(rep["intra_px"]["n"], 1)

    def test_evaluator_pending_without_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                ea.main(["--gate1", d, "--out", str(Path(d) / "o.json")])
            self.assertEqual(cm.exception.code, 3)
            self.assertIn("PENDING", (Path(d) / "o.json").read_text())


class Reconcile(unittest.TestCase):
    def test_cmh_no_difference(self):
        r = cmh([((10, 10), (10, 10)), ((5, 15), (5, 15))])
        self.assertGreater(r["p"], 0.5)
        self.assertAlmostEqual(r["common_odds_ratio_hist_vs_matched"], 1.0)


if __name__ == "__main__":
    unittest.main()
