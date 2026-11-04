"""Gate 1R tests: Codex's counterexamples against the Gate 1 verifier (must reproduce the blind spots) and against the
Gate 1R verifier (must not pass silently), component semantics, and the annotation workflow.

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python -m unittest discover -s research/contact_aware_augmentation/gate1r/tests -t research/contact_aware_augmentation/gate1r/tests -v

Synthetic tests need no dataset. RealPilot tests use the committed fixture fixtures/pilot_inputs.npz (recorded joint
streams + Gate 1 lid tracks of episodes 0/4/8) and the frozen Gate 1 folds; no dataset access.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "annotation"), str(HERE.parents[1] / "gate1")]

import annotate_gate1r as an  # noqa: E402
import corruptions_r as cr  # noqa: E402
import evaluate_gate1r_annotations as ea  # noqa: E402
import fixtures_r as fx  # noqa: E402
import gate1_core as gc  # noqa: E402
import spec as S  # noqa: E402
import verifier_r as vr  # noqa: E402

FOLD = fx.fold_from()
S0, P0, GEO = fx.episode(0)


def run(s, P=P0):
    return vr.verify(s, FOLD, P)


class Gate1BlindSpots(unittest.TestCase):
    """The three Codex failures are properties of the Gate 1 code, reproduced here on synthetic data."""

    def setUp(self):
        self.g = fx.gate1_view(S0, P0)
        tail = len(S0.uv) - 11
        self.tail = tail
        base = gc.score_C(self.g, gc.g0.BODY_CAM, tail)
        row = {f"C.{k}": v for k, v in base.items()}
        self.th = gc.freeze_thresholds([row, row])
        self.base = gc.method_score(row, self.th, "C_rel")

    def test_F1_gate1_sample_has_no_action_field(self):
        self.assertNotIn("a", {f.name for f in __import__("dataclasses").fields(gc.Sample)})

    def test_F2_gap_excursion_leaves_C_rel_unchanged(self):
        w = gc.windows(self.g, self.tail)
        s = copy.deepcopy(self.g)
        start, stop = int(w["giver"][-1]) + 1, int(w["recv"][0])
        s.uv[start:stop, 0] += 60 * np.sin(np.linspace(0, np.pi, stop - start + 2))[1:-1] ** 2
        r = {f"C.{k}": v for k, v in gc.score_C(s, gc.g0.BODY_CAM, self.tail).items()}
        self.assertAlmostEqual(gc.method_score(r, self.th, "C_rel")[0], self.base[0], places=9)

    def test_F3_missing_receiver_is_a_finite_pass(self):
        w = gc.windows(self.g, self.tail)
        s = copy.deepcopy(self.g)
        s.ok[w["recv"]] = False
        r = {f"C.{k}": v for k, v in gc.score_C(s, gc.g0.BODY_CAM, self.tail).items()}
        sc, flag, why = gc.method_score(r, self.th, "C_rel")
        self.assertTrue(np.isfinite(sc))
        self.assertFalse(flag)
        self.assertIn("unevaluable_component", why)


class Gate1RVerifier(unittest.TestCase):
    def test_original_and_hard_negatives_pass(self):
        self.assertEqual(run(S0)["status"], "PASS", json.dumps(run(S0), default=str)[:2000])
        ok = S0.ok.copy(); ok[np.random.default_rng(0).choice(len(ok), len(ok) // 10, replace=False)] = False
        self.assertEqual(run(cr._clone(S0, "d", "h", "valid", ok=ok))["status"], "PASS")
        self.assertEqual(run(cr._clone(S0, "n", "h", "valid", uv=S0.uv + np.random.default_rng(1).normal(0, 1.5, S0.uv.shape)))["status"], "PASS")

    def test_F1_action_offsets_fail_component_A(self):
        arm = [i for i, n in enumerate(S0.names) if not n.endswith("8.pos")]
        for c, reason in ((1.0, "tracking_error_contract"), (0.1, "command_response")):
            a = S0.a.copy(); a[:, arm] += c
            r = run(cr._clone(S0, "x", "x", "invalid", a=a))
            self.assertEqual(r["A"]["status"], "FAIL")
            self.assertIn(reason, r["A"]["reasons"])
            self.assertEqual(r["status"], "FAIL")

    def test_F2_transition_excursion_fails_component_C(self):
        ev = vr.command_events(S0)
        seg = vr.segments(ev, len(S0.uv))["transition"]
        uv = S0.uv.copy(); uv[seg, 0] += 60 * np.sin(np.linspace(0, np.pi, len(seg) + 2))[1:-1] ** 2
        r = run(cr._clone(S0, "x", "x", "invalid", uv=uv))
        self.assertEqual(r["C"]["status"], "FAIL")
        self.assertIn("transition_track_inconsistent", r["C"]["reasons"])

    def test_F3_missing_receiver_never_passes(self):
        ev = vr.command_events(S0)
        seg = vr.segments(ev, len(S0.uv))
        for frac in (0.25, 0.5, 1.0):
            rs = np.concatenate([seg["recv_calib"], seg["recv_pred"]])
            ok = S0.ok.copy(); ok[rs[: int(round(frac * len(rs)))]] = False
            r = run(cr._clone(S0, "m", "m", "invalid", ok=ok))
            self.assertNotEqual(r["status"], "PASS", frac)
            self.assertEqual(r["D"]["status"], "UNKNOWN")

    def test_temporal_offset_fails(self):
        T = len(S0.uv)
        src = np.clip(np.arange(T) - 10, 0, T - 1)
        r = run(cr._clone(S0, "t", "t", "invalid", uv=S0.uv[src].copy()))
        self.assertEqual(r["status"], "FAIL")

    def test_command_only_phase_edit_fails(self):
        a = S0.a.copy()
        j = S0.names.index("RJ8.pos")
        a[fx.EV["lc"] - 10:, j] = vr.GRIP["right"][1]  # giver commanded open 10 frames before the receiver closes
        r = run(cr._clone(S0, "p", "p", "invalid", a=a))
        self.assertEqual(r["status"], "FAIL")
        self.assertTrue(any("jaw_ignores_command" in x for x in r["A"]["reasons"]) or r["C"]["status"] == "FAIL")

    def test_prediction_is_not_refit(self):
        """A slip after the calibration segment is caught although a refit on the whole hold could absorb part of it."""
        ev = vr.command_events(S0)
        seg = vr.segments(ev, len(S0.uv))
        uv = S0.uv.copy()
        pred = seg["giver_pred"]
        uv[pred] += np.linspace(0, 40, len(pred))[:, None] * np.array([1.0, 0.0])
        r = run(cr._clone(S0, "s", "s", "invalid", uv=uv))
        self.assertEqual(r["B"]["giver"]["status"], "FAIL")

    def test_missing_threshold_is_unknown_and_absolute_contact_unknown(self):
        fold = dict(FOLD, tau=dict(FOLD["tau"], recv_pred=None))
        r = vr.verify(S0, fold, P0)
        self.assertEqual(r["B"]["receiver"]["status"], "UNKNOWN")
        self.assertNotEqual(r["status"], "PASS")
        self.assertEqual(run(S0)["absolute_contact"]["status"], "UNKNOWN")

    def test_combine(self):
        self.assertEqual(vr.combine(["PASS", "UNKNOWN", "FAIL"]), "FAIL")
        self.assertEqual(vr.combine(["PASS", "UNKNOWN"]), "UNKNOWN")
        self.assertEqual(vr.combine(["PASS", "PASS"]), "PASS")

    def test_spec_is_versioned_and_sealed_disjoint(self):
        self.assertTrue(S.SPEC_VERSION.startswith("gate1r-"))
        self.assertFalse(set(S.PILOT) & set(S.SEALED))


class Annotation(unittest.TestCase):
    prov = {"session": "t", "tool_sha256": "x", "repo_commit": "y", "dataset_rev": "z"}

    def test_keypoint_session(self):
        items = [{"episode": 0, "frame": 3, "purpose": "k", "repeat": 0}]
        K = an.KeypointSession("A", items, self.prov)
        self.assertFalse(K.next())
        K.set_confidence(1); K.click(1, 2)
        K.occluded(); K.click(3, 4); K.click(5, 6); K.out_of_frame(); K.click(7, 8); K.click(9, 9); K.occluded()
        self.assertTrue(K.next())
        rows = K.rows()
        self.assertEqual(len(rows), len(an.KEYPOINTS))
        self.assertEqual(rows[0]["confidence"], 1)
        self.assertEqual(rows[1]["confidence"], "")
        self.assertEqual({r["dataset_rev"] for r in rows}, {"z"})

    def test_event_session(self):
        E = an.EventSession("A", self.prov, {e: 100 for e in S.PILOT})
        E.step(30); E.mark("v")
        E.not_observable("v")
        rows = E.rows()
        self.assertEqual(rows[0]["frame"], 30)
        self.assertEqual(rows[1]["observable"], 0)

    def test_manifest_pilot_only(self):
        ev = {e: {"right_close": 90, "left_close": 170, "right_open": 175} for e in S.PILOT}
        rows = an.manifest_rows(ev, {e: 260 for e in S.PILOT})
        self.assertEqual({r["episode"] for r in rows}, set(S.PILOT))
        self.assertEqual(rows, an.manifest_rows(ev, {e: 260 for e in S.PILOT}))

    def test_gt_rules(self):
        def kp(a, k, vis, conf, u, rep=0):
            return {"annotator": a, "episode": "0", "frame": "5", "repeat": str(rep), "keypoint": k, "visibility": vis,
                    "confidence": str(conf), "u": u, "v": 50}
        rows = [kp("A", "obj_lid_center", "visible", 3, 100), kp("B", "obj_lid_center", "visible", 2, 104),
                kp("A", "R_pad_tip_1", "visible", 1, 100), kp("B", "R_pad_tip_1", "visible", 3, 100),
                kp("A", "R_pad_tip_2", "visible", 3, 100), kp("B", "R_pad_tip_2", "visible", 3, 120)]
        gt, st = ea.keypoint_gt(rows)
        self.assertEqual(set(gt), {(0, 5, "obj_lid_center")})
        self.assertEqual(st["excluded_not_visible_or_low_conf"], 1)
        self.assertEqual(st["excluded_disagreement"], 1)

        def ev(a, obs, f, c):
            return {"annotator": a, "episode": "0", "event": "lift_off", "observable": str(obs), "frame": str(f), "confidence": str(c)}
        egt, no = ea.event_gt([ev("A", 1, 100, 3), ev("B", 1, 101, 2)])
        self.assertAlmostEqual(egt[(0, "lift_off")], 100.5)
        egt, no = ea.event_gt([ev("A", 1, 100, 3), ev("B", 1, 110, 3)])
        self.assertEqual(egt, {})

    def test_evaluator_pending(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                ea.main(["--gate1", d, "--out", str(Path(d) / "o.json")])
            self.assertEqual(cm.exception.code, 3)
            self.assertFalse(json.loads((Path(d) / "o.json").read_text())["sealed_ready"])


FIX = HERE / "fixtures" / "pilot_inputs.npz"
G1 = HERE.parents[1] / "outputs" / "gate1" / "metrics_full.json"


@unittest.skipUnless(FIX.exists() and G1.exists(), "real-data fixture or Gate 1 metrics missing")
class RealPilotCounterexamples(unittest.TestCase):
    """Codex's three failures on the recorded pilot, against the frozen Gate 1 folds (inputs from the fixture)."""

    @classmethod
    def setUpClass(cls):
        d = np.load(FIX, allow_pickle=True)
        cls.names = list(d["names"])
        cls.folds = {f["test"]: f for f in json.loads(G1.read_text())["folds"]}
        sys.path.insert(0, str(HERE.parents[1] / "scripts"))
        import gate0_diagnostics as g0

        class _V:
            def names(self, _):
                return cls.names
        cls.fk = g0.FK(_V())
        cls.eps = {e: vr.Sample("original", "valid", "valid", e, e, d[f"q{e}"], d[f"a{e}"], cls.names, d[f"uv{e}"], d[f"ok{e}"])
                   for e in S.PILOT}

    def test_gate1_passes_all_three_on_every_episode(self):
        for e, s in self.eps.items():
            f = self.folds[e]
            P = gc.fk_hands(self.fk, s.q, f["lag"])
            g = fx.gate1_view(s, P)
            tail = int(np.where(s.ok)[0].max()) - 10
            refC = {k: np.array(v) / 1e3 for k, v in f["refC_mm"].items()}
            score = lambda x: gc.method_score({f"C.{k}": v for k, v in gc.score_C(x, gc.g0.BODY_CAM, tail, refC).items()}, f["thresholds"], "C_rel")  # noqa: E731
            base = score(g)
            w = gc.windows(g, tail)
            ex = copy.deepcopy(g)
            a0, a1 = int(w["giver"][-1]) + 1, int(w["recv"][0])
            ex.uv[a0:a1, 0] += 60 * np.sin(np.linspace(0, np.pi, a1 - a0 + 2))[1:-1] ** 2
            mi = copy.deepcopy(g)
            mi.ok[w["recv"]] = False
            self.assertFalse(base[1], e)
            self.assertAlmostEqual(score(ex)[0], base[0], places=9)
            self.assertFalse(score(mi)[1])


if __name__ == "__main__":
    unittest.main()
