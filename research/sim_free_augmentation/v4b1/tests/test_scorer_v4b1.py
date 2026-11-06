"""Amendment A1 scorer contract tests. All numbers here are synthetic FIXTURES, not experimental results.

    CUDA_VISIBLE_DEVICES= env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import score_v4b1 as sc  # noqa: E402

E = sc.StudyContractError
GOOD_STEMS = [f"{a}_s{s}" for a in sc.NEW for s in sc.SEEDS]


def c(pt, lo, hi, seeds=(0, 1, 2), seed_val=None, metric=("far/grasp", "all40/handover")):
    sv = pt if seed_val is None else seed_val
    out = {m: [pt, [lo, hi]] for m in metric}
    out["per_seed"] = {f"s{s}": {m: (sv[i] if isinstance(sv, (list, tuple)) else sv) for m in metric} | {"all40/pb": 0.0}
                       for i, s in enumerate(seeds)}
    return out


def fixture(**kw):
    """a clean positive study; keyword overrides replace single contrasts."""
    C = {"RpBoth__vs__RpBoth_NoSynthRGB": c(.06, .02, .10),
         "RpBoth_rerun__vs__RpBoth": c(.005, -.02, .03),
         "RpBoth_NoSynthRGB__vs__R": c(.04, .00, .08),
         "RpBoth_rerun__vs__RpBoth_NoSynthRGB": c(.06, .02, .10),
         "RpPos_rerun__vs__RpPos": c(.004, -.01, .02),
         "RpPos_PhaseMask__vs__RpPos": c(.03, .01, .05),
         "RpPos_MatchedRandMask__vs__RpPos": c(.005, -.01, .02),
         "RpPos_PhaseMask__vs__RpPos_MatchedRandMask": c(.025, .005, .045),
         "RpPos_PhaseMask__vs__RpPos_rerun": c(.03, .01, .05)}
    C.update({k.replace("__", "__vs__", 1) if "__vs__" not in k else k: v for k, v in kw.items()})
    return C


def screen(deg=.01, grasp=.12, grasp_lo=.06):
    return {"RpPos_PhaseMask": {"far_grasp_rel": grasp, "far_grasp_ci": [grasp_lo, grasp + .06], "handover_degradation": deg,
                                "handover_degradation_ci": [deg - .01, deg + .01], "PASS": deg <= .02 and grasp >= .05}}


class FakeZ:
    def __init__(self, d):
        self.d, self.files = d, list(d)

    def __getitem__(self, k):
        return self.d[k]


class TestCompleteness(unittest.TestCase):
    def test_full_registered_set_accepted(self):
        self.assertEqual(set(sc.check_names(GOOD_STEMS)), set(sc.NEW))

    def test_one_and_two_seed_inputs_rejected(self):
        for seeds in ([0], [0, 1], [1, 2]):
            with self.assertRaises(E):
                sc.check_names([f"{a}_s{s}" for a in sc.NEW for s in seeds])
            C = fixture()
            for k in C:
                C[k]["per_seed"] = {f"s{s}": C[k]["per_seed"][f"s{s}"] for s in seeds}
            with self.assertRaises(E):
                sc.decisions(C, screen(), None)

    def test_missing_registered_arm_rejected(self):
        for arm in sc.NEW:
            with self.assertRaises(E):
                sc.check_names([s for s in GOOD_STEMS if not s.startswith(arm + "_s")])

    def test_missing_rerun_or_control_contrast_rejected(self):
        for k in ("RpBoth_rerun__vs__RpBoth", "RpPos_rerun__vs__RpPos", "RpPos_PhaseMask__vs__RpPos_MatchedRandMask"):
            C = fixture()
            del C[k]
            with self.assertRaises(E):
                sc.decisions(C, screen(), None)

    def test_unregistered_or_superseded_or_extra_seed_rejected(self):
        for bad in ("RpPos_RandMask_s0", "Foo_s1", "RpPos_PhaseMask_s3", "noseed"):
            with self.assertRaises(E):
                sc.check_names(GOOD_STEMS + [bad])

    def test_smoke_allows_incomplete_but_protects_names(self):
        sc.check_names(["RpPos_PhaseMask_s0"], smoke=True)
        with self.assertRaises(E):
            sc.check_names(["R_s0"], smoke=True)


class TestFrozenReferenceProtection(unittest.TestCase):
    def test_new_predictions_cannot_shadow_frozen(self):
        for ref in ("R_s0", "RpPos_s1", "RpBoth_s2", "v3_R_s0", "RpHO_s0"):
            with self.assertRaises(E):
                sc.check_names(GOOD_STEMS + [ref])
            with tempfile.TemporaryDirectory() as td:
                for st in GOOD_STEMS + [ref]:
                    (Path(td) / f"{st}.npz").write_bytes(b"")
                with self.assertRaises(E):
                    sc.load_study(td)
                with self.assertRaises(E):
                    sc.load_study(td, smoke=True)

    def test_frozen_directory_refused(self):
        with self.assertRaises(E):
            sc.load_study(sc.VPRED)

    def test_frozen_reference_hashes_currently_valid(self):
        prov = sc.check_frozen_refs()
        self.assertEqual(len(prov), 9)
        self.assertTrue(all(v["step"] == 1000 for v in prov.values()))


class TestArchiveContract(unittest.TestCase):
    def setUp(self):
        self.items = np.array([(10, t) for t in range(sc.N_ITEMS)])
        self.ok = {"items": self.items} | {k: np.zeros((sc.N_ITEMS, 50, 16), np.float16) for k in sc.SC["conditions"]}

    def test_ok(self):
        sc.check_archive("x", FakeZ(self.ok), self.items, sc.SC["conditions"])

    def test_violations(self):
        bad = []
        d = dict(self.ok); d["items"] = self.items[::-1]; bad.append(d)
        d = dict(self.ok); del d["gripper_only"]; bad.append(d)
        d = dict(self.ok); d["blank"] = np.zeros((sc.N_ITEMS, 50, 14), np.float16); bad.append(d)
        d = dict(self.ok); d["original"] = d["original"].copy(); d["original"][3, 4, 5] = np.nan; bad.append(d)
        d = dict(self.ok); d["object_only"] = np.full((sc.N_ITEMS, 50, 16), np.inf, np.float16); bad.append(d)
        for d in bad:
            with self.assertRaises(E):
                sc.check_archive("x", FakeZ(d), self.items, sc.SC["conditions"])


class TestDecisionContract(unittest.TestCase):
    def test_clean_positive_fixture(self):
        d = sc.decisions(fixture(), screen(), None)
        self.assertEqual(d["H1"]["verdict"], "GENERATED_RGB_USEFUL")
        self.assertEqual(d["H2"]["Q2_answer"], "YES")
        self.assertEqual(d["H2"]["mechanism_label"], "PHASE_PLACED_SUPERVISION_REDUCES_HANDOVER_ERROR_BEYOND_MATCHED_RANDOM")

    def test_effects_smaller_than_rerun_variability(self):
        d = sc.decisions(fixture(RpBoth_rerun__vs__RpBoth=c(.08, .02, .14)), screen(), None)["H1"]
        self.assertEqual(d["verdict"], "INCREMENTAL_VALUE_NOT_ESTABLISHED")
        self.assertIn("effect within replication noise |RpBoth_rerun vs RpBoth|", d["not_established_reasons"])
        d = sc.decisions(fixture(RpBoth__vs__RpBoth_NoSynthRGB=c(-.05, -.08, -.02), RpBoth_rerun__vs__RpBoth=c(-.10, -.15, -.05)), screen(), None)["H1"]
        self.assertEqual(d["verdict"], "INCREMENTAL_VALUE_NOT_ESTABLISHED")          # harmful also guarded
        h = sc.decisions(fixture(RpPos_rerun__vs__RpPos=c(.05, .04, .06)), screen(), None)["H2"]
        self.assertEqual(h["A_handover_improvement_vs_RpPos"]["status"], "not_resolved")
        self.assertEqual(h["D_phase_specific_increment_vs_matched_random"]["status"], "not_resolved")
        self.assertEqual(h["mechanism_label"], "HANDOVER_REDUCTION_NOT_ESTABLISHED")
        self.assertEqual(h["Q2_answer"], "NOT_SUPPORTED_BY_THIS_SCREEN")
        self.assertFalse(h["E_statistical_resolution"]["A_exceeds_noise"])

    def test_harmful_beyond_noise(self):
        d = sc.decisions(fixture(RpBoth__vs__RpBoth_NoSynthRGB=c(-.05, -.08, -.02)), screen(), None)["H1"]
        self.assertEqual(d["verdict"], "GENERATED_RGB_HARMFUL")

    def test_confidence_intervals_overlapping_zero(self):
        d = sc.decisions(fixture(RpBoth__vs__RpBoth_NoSynthRGB=c(.05, -.01, .11)), screen(), None)["H1"]
        self.assertEqual(d["verdict"], "INCREMENTAL_VALUE_NOT_ESTABLISHED")
        self.assertIn("95% CI includes 0", d["not_established_reasons"])
        d = sc.decisions(fixture(RpBoth__vs__RpBoth_NoSynthRGB=c(.05, .01, .09, seed_val=[.08, .09, -.01])), screen(), None)["H1"]
        self.assertEqual(d["verdict"], "INCREMENTAL_VALUE_NOT_ESTABLISHED")          # seed disagreement
        h = sc.decisions(fixture(RpPos_PhaseMask__vs__RpPos_MatchedRandMask=c(0., -.05, .05)), screen(), None)["H2"]
        self.assertEqual(h["D_phase_specific_increment_vs_matched_random"]["status"], "not_resolved")
        self.assertNotIn("BUDGET", h["mechanism_label"])
        self.assertNotIn("EQUIVAL", h["mechanism_label"])
        self.assertEqual(h["Q2_answer"], "NOT_SUPPORTED_BY_THIS_SCREEN")

    def test_both_masking_methods_improve(self):
        h = sc.decisions(fixture(RpPos_MatchedRandMask__vs__RpPos=c(.028, .01, .046),
                                 RpPos_PhaseMask__vs__RpPos_MatchedRandMask=c(.002, -.02, .024)), screen(), None)["H2"]
        self.assertEqual(h["mechanism_label"], "BOTH_MASKS_REDUCE_HANDOVER_ERROR_PHASE_SPECIFICITY_NOT_ESTABLISHED")
        self.assertFalse(h["original_P3_beats_control"])
        self.assertEqual(h["Q2_answer"], "NOT_SUPPORTED_BY_THIS_SCREEN")

    def test_handover_improves_but_guardrail_violated(self):
        h = sc.decisions(fixture(), screen(deg=.025), None)["H2"]
        self.assertEqual(h["A_handover_improvement_vs_RpPos"]["status"], "improved")     # reduction is still reported
        self.assertFalse(h["B_residual_handover_degradation_vs_R"]["within_guardrail_point_estimate"])
        self.assertFalse(h["original_P1_reduction_and_guardrail"])
        self.assertNotIn("DOES_NOT_REDUCE", h["mechanism_label"])
        self.assertEqual(h["Q2_answer"], "NOT_SUPPORTED_BY_THIS_SCREEN")

    def test_grasp_deteriorates_while_handover_improves(self):
        C = fixture(RpPos_PhaseMask__vs__RpPos=c(.03, .01, .05))
        C["RpPos_PhaseMask__vs__RpPos"]["far/grasp"] = [-.08, [-.12, -.04]]
        h = sc.decisions(C, screen(grasp=.03, grasp_lo=-.01), None)["H2"]
        self.assertEqual(h["A_handover_improvement_vs_RpPos"]["status"], "improved")
        self.assertFalse(h["C_far_grasp_preservation"]["P2_preserved_vs_R"])
        self.assertTrue(h["C_far_grasp_preservation"]["far_grasp_reduced_vs_RpPos"])
        self.assertEqual(h["Q2_answer"], "NOT_SUPPORTED_BY_THIS_SCREEN")

    def test_phase_mask_worsens(self):
        h = sc.decisions(fixture(RpPos_PhaseMask__vs__RpPos=c(-.03, -.05, -.01)), screen(), None)["H2"]
        self.assertEqual(h["mechanism_label"], "PHASE_MASK_INCREASES_HANDOVER_ERROR")


if __name__ == "__main__":
    unittest.main()
