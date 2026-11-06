"""Amendment A1 tests: per-row matched random control (H2). CPU; parquet-only except one dataset-view spot check.

    cd research/sim_free_augmentation/v4b1
    CUDA_VISIBLE_DEVICES= env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sfa4b1 import interventions as iv  # noqa: E402
from sfa4b1 import schedule_masks as sm  # noqa: E402

PH, MR, BASE = "RpPos_PhaseMask", "RpPos_MatchedRandMask", "RpPos_rerun"


def k_closed_form(f, L, ge):
    """independent count of originally valid targets the phase mask retains: steps f..f+49 with t < min(L, grasp_end)."""
    return int(np.clip(min(L, ge) - f, 0, iv.CHUNK))


class TestMatchedControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = sm.Meta()
        cls.E = {s: {a: sm.expected(cls.meta, a, s) for a in (PH, MR, BASE)} for s in iv.CFG["seeds"]}

    def test_full_schedules_exactly_matched(self):
        """all registered seeds x 1,000 sampled batches x 16 rows."""
        m = self.meta
        checked = boundary_rows = zero_rows = 0
        for seed, E in self.E.items():
            p, r, b = E[PH], E[MR], E[BASE]
            self.assertEqual(p["index"].shape, (1000, 16))
            # identical original batches, sources, synthetic flags, original padding
            for k in ("index", "source", "is_syn", "valid_orig"):
                self.assertTrue(np.array_equal(p[k], r[k]) and np.array_equal(p[k], b[k]), (seed, k))
            self.assertTrue(np.array_equal(b["valid"], b["valid_orig"]))
            syn = p["is_syn"]
            self.assertEqual(int(syn.sum(1).min()), 8)
            self.assertEqual(int(syn.sum(1).max()), 8)
            vp, vr = p["valid"].sum(-1), r["valid"].sum(-1)
            # exact retained count per sampled row, against an independent closed form
            for s_, j in zip(*np.nonzero(syn)):
                i = int(p["index"][s_, j])
                K = k_closed_form(int(m.frame[i]), int(m.length[i]), m.grasp_end[int(m.ep[i])])
                self.assertEqual(int(vp[s_, j]), K)
                self.assertEqual(int(vr[s_, j]), K)
                checked += 1
                zero_rows += K == 0
                boundary_rows += 0 < K < int(p["valid_orig"][s_, j].sum())
            # retained subset of originally valid targets; padding never un-masked
            self.assertFalse((r["valid"] & ~r["valid_orig"]).any())
            self.assertFalse((p["valid"] & ~p["valid_orig"]).any())
            # identical active-row sets, per-batch valid counts and loss denominators (x16 channels)
            self.assertTrue(np.array_equal(vp > 0, vr > 0))
            self.assertTrue(np.array_equal(vp.sum(1), vr.sum(1)))
            self.assertTrue(np.array_equal(16 * vp.sum(1), 16 * vr.sum(1)))
            # identical source-level direct-target exposure, overall and per batch
            for src in range(10):
                self.assertEqual(int((vp * (p["source"] == src) * syn).sum()), int((vr * (r["source"] == src) * syn).sum()))
                self.assertTrue(np.array_equal((vp * (p["source"] == src)).sum(1), (vr * (r["source"] == src)).sum(1)))
            # identical real-target loss weights: real rows fully supervised as originally, same denominator
            self.assertTrue(np.array_equal(p["valid"][~syn], p["valid_orig"][~syn]))
            self.assertTrue(np.array_equal(r["valid"][~syn], r["valid_orig"][~syn]))
            self.assertTrue(np.allclose(1 / vp.sum(1), 1 / vr.sum(1)))
            # the control is not trivially the phase mask: positions differ in boundary-crossing rows
            cross = syn & (vp > 0) & (vp < p["valid_orig"].sum(-1))
            self.assertGreater(int((~(p["valid"] == r["valid"]).all(-1) & cross).sum()), 0.9 * cross.sum())
        self.assertEqual(checked, 3 * 8000)
        self.assertGreater(zero_rows, 0)
        self.assertGreater(boundary_rows, 0)

    def test_uniform_positions_within_valid(self):
        """retained positions are uniform among a crossing row's valid steps (normalised rank mean 0.5)."""
        ranks = []
        for E in self.E.values():
            p, r = E[PH], E[MR]
            vp = p["valid"].sum(-1)
            cross = p["is_syn"] & (vp > 0) & (vp < p["valid_orig"].sum(-1))
            for s_, j in zip(*np.nonzero(cross)):
                ids = np.flatnonzero(r["valid_orig"][s_, j])
                keep = np.flatnonzero(r["valid"][s_, j])
                ranks += list((np.searchsorted(ids, keep) + 0.5) / len(ids))
        self.assertLess(abs(np.mean(ranks) - 0.5), 0.02)

    def test_deterministic(self):
        e2 = sm.expected(self.meta, MR, 1)
        self.assertTrue(np.array_equal(e2["valid"], self.E[1][MR]["valid"]))

    def test_zero_retention_and_episode_boundaries(self):
        L, ge, seed, gi = 100, 60, 0, 123456
        for f, K in ((70, 0), (60, 0), (59, 1), (11, 49), (10, 50), (0, 50)):
            ex = iv.synthetic_exclusion("matched_random", seed, gi, f, L, ge)
            self.assertEqual(int((~ex).sum()), K, f)
            self.assertEqual(int((~iv.synthetic_exclusion("phase", seed, gi, f, L, ge) & iv.valid_targets(f, L)).sum()), K, f)
        # grasp-end beyond the episode end: K = number of valid steps, padding stays excluded
        for f in (80, 99, 51):
            valid = iv.valid_targets(f, L)
            ex = iv.synthetic_exclusion("matched_random", seed, gi, f, L, 200)
            self.assertTrue(np.array_equal(~ex, valid), f)
        # partially padded crossing chunk: retained inside valid, exactly K
        valid = iv.valid_targets(80, 110)
        ex = iv.synthetic_exclusion("matched_random", seed, gi, 80, 110, 95)
        self.assertEqual(int((~ex).sum()), 15)
        self.assertFalse((~ex & ~valid).any())
        with self.assertRaises(AssertionError):
            iv.matched_random_retained(0, 1, np.zeros(50, bool), 1)

    def test_phase_boundaries_exhaustive(self):
        """the Codex audit's exhaustive check (322,800 entries) plus the matched control at every frame of the 25 episodes."""
        m, n = self.meta, 0
        for E, eps in m.sel["arms"]["RpPos"].items():
            for e in eps:
                L, ge = m.sr[e][1] - m.sr[e][0], m.grasp_end[e]
                for f in range(L):
                    self.assertTrue(np.array_equal(iv.phase_mask(f, ge), np.arange(f, f + 50) >= ge))
                    n += 50
                    keep = ~iv.synthetic_exclusion("matched_random", 0, m.nR + m.sr[e][0] + f, f, L, ge)
                    self.assertEqual(int(keep.sum()), k_closed_form(f, L, ge))
                    self.assertFalse((keep & ~iv.valid_targets(f, L)).any())
        self.assertEqual(n, 322800)


class TestMatchedDatasetViewAndApplier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        V31 = HERE.parent / "v3_1"
        ld = lambda n: LeRobotDataset(f"local/{n}", root=V31 / "outputs/lerobot" / n, delta_timestamps={"action": [k / 30 for k in range(50)]})  # noqa: E731
        cls.dr, cls.ds = ld("sfa_v31_R"), ld("sfa_v31_Gp_matched_evidence_complete")

    def test_view_items(self):
        d1, d0 = iv.TwoPartV41(self.dr, self.ds, MR, 2), iv.mx.TwoPart(self.dr, self.ds)
        nR = d1.nR
        for i in (5, nR + 7, nR + 150, nR + 23600):
            a, b = d1[i], d0[i]
            for k in ("action", "observation.state", "action_is_pad", "observation.images.body_cam"):
                self.assertTrue(torch.equal(a[k], b[k]), (i, k))
            self.assertEqual(int(a["sfa_index"]), i)
            if i < nR:
                self.assertFalse(a["sfa_target_mask"].any())
            else:
                f, e = int(a["frame_index"]), int(a["episode_index"])
                exp = iv.synthetic_exclusion("matched_random", 2, i, f, d1.syn_len[e], d1.grasp_end[e])
                self.assertTrue(np.array_equal(a["sfa_target_mask"].numpy(), exp))
                self.assertTrue((a["sfa_target_mask"] | ~a["action_is_pad"]).all())   # padding subset of exclusion

    def test_applier_records_applied_masks(self):
        def fake_pre(batch, camera_keys, rename_map, preprocessor):
            out = dict(batch)
            out[iv.CAM] = out.pop("observation.images.body_cam")
            return out
        B = 4
        is_syn = torch.tensor([False, True, False, True])
        tmask = torch.zeros(B, 50, dtype=torch.bool)
        tmask[1, ::2] = True
        tmask[0, :] = True                      # must never reach a real row
        pad = torch.zeros(B, 50, dtype=torch.bool)
        pad[3, 40:] = True
        ap = iv.Applier(MR, fake_pre, "/dev/null")
        out = ap({"observation.images.body_cam": torch.rand(B, 3, 8, 8), "observation.state": torch.zeros(B, 16), "action": torch.zeros(B, 50, 16),
                  "action_is_pad": pad.clone(), "sfa_is_synth": is_syn, "sfa_target_mask": tmask, "sfa_index": torch.arange(B) + 10}, [], {}, None)
        self.assertFalse(out["action_is_pad"][0].any())
        self.assertEqual(int(out["action_is_pad"][1].sum()), 25)
        self.assertTrue(torch.equal(out["action_is_pad"][3], pad[3]))
        self.assertEqual(ap.rows[-1]["loss_denominator_elements"], 16 * int((~out["action_is_pad"]).sum()))
        rec = np.unpackbits(ap.rec_valid[-1], axis=-1)[:, :50].astype(bool)
        self.assertTrue(np.array_equal(rec, ~out["action_is_pad"].numpy()))
        self.assertTrue(np.array_equal(ap.rec_index[-1], np.arange(B) + 10))
        ap.rows.clear()


if __name__ == "__main__":
    unittest.main()
