"""v4B regression tests (CPU only).

    cd research/sim_free_augmentation/v4
    CUDA_VISIBLE_DEVICES= env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa4b import mixture as mx  # noqa: E402

V31 = HERE.parent / "v3_1"


def ranges(name):
    import pandas as pd
    ep = pd.read_parquet(sorted((V31 / "outputs/lerobot" / name / "meta/episodes").rglob("*.parquet"))[0])
    return {int(e): (int(a), int(b)) for e, a, b in zip(ep["episode_index"], ep["dataset_from_index"], ep["dataset_to_index"])}


class TestV4B(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sel = mx.selection()
        cls.rr, cls.sr = ranges("sfa_v31_R"), ranges("sfa_v31_Gp_matched_evidence_complete")
        cls.nR = max(b for a, b in cls.rr.values())
        cls.audit = {r["ec_dataset_episode_index"]: r for r in json.loads(mx.AUDIT.read_text())["candidates"]
                     if r["in_frozen_Gp_matched_evidence_complete_dataset"]}

    def sampler(self, arm, seed=0, steps=1000):
        real, syn = mx.frame_pools(arm, self.sel, self.rr, self.sr, self.nR)
        return mx.StratifiedBatches(arm, self.sel, real, syn, steps, 16, seed, mx.CFG["training"]["sampler_seed"], 0.5)

    def test_selection_matched_and_classes(self):
        k = self.sel["k"]
        for arm in ("RpPos", "RpHO", "RpBoth"):
            self.assertEqual({E: len(v) for E, v in self.sel["arms"][arm].items()}, k, arm)
        for i in sum(self.sel["arms"]["RpPos"].values(), []) + sum(self.sel["arms"]["RpPosAll"].values(), []):
            self.assertEqual(self.audit[i]["class"], "position_only")
        for i in sum(self.sel["arms"]["RpHO"].values(), []):
            self.assertEqual(self.audit[i]["class"], "handover_only")
        for E, eps in self.sel["arms"]["RpBoth"].items():
            self.assertTrue(set(eps) <= set(self.sel["arms"]["RpPos"][E]) | set(self.sel["arms"]["RpHO"][E]))
        for i in sum(self.sel["arms"]["RpPosAll"].values(), []) + sum(self.sel["arms"]["RpHO"].values(), []):
            self.assertTrue(self.audit[i]["evidence_complete"])
            self.assertEqual(self.audit[i]["E"], next(E for E, v in self.sel["arms"]["RpPosAll"].items() if i in v) if self.audit[i]["class"] == "position_only"
                             else next(E for E, v in self.sel["arms"]["RpHO"].items() if i in v))

    def test_real_fraction_and_source_exposure(self):
        src_of = {}
        for e, (a, b) in self.rr.items():
            src_of.update({i: e for i in range(a, b)})
        for E, eps in self.sel["arms"]["RpPosAll"].items():
            for i in eps:
                a, b = self.sr[i]
                src_of.update({self.nR + j: E for j in range(a, b)})
        for E, eps in self.sel["arms"]["RpHO"].items():
            for i in eps:
                a, b = self.sr[i]
                src_of.update({self.nR + j: E for j in range(a, b)})
        for arm in ("R", "RpPos", "RpHO", "RpBoth", "RpPosAll"):
            s = self.sampler(arm)
            cnt = np.zeros(10)
            allowed = {self.nR + j for E, eps in self.sel["arms"].get(arm, {}).items() for i in eps for j in range(*self.sr[i])}
            for b in s:
                self.assertEqual(len(b), 16)
                n_real = sum(i < self.nR for i in b)
                self.assertEqual(n_real, 16 if arm == "R" else 8, arm)
                for i in b:
                    if i >= self.nR:
                        self.assertIn(i, allowed, arm)
                    cnt[src_of[i]] += 1
            np.testing.assert_allclose(s.w @ np.eye(10) * (0.5 if arm != "R" else 1.0) + (0 if arm == "R" else 0.5 * np.array(
                [1 / 8 if E in self.sel["sources"] else 0 for E in range(10)])), np.full(10, 0.1), atol=1e-12)
            frac = cnt / cnt.sum()
            self.assertLess(np.abs(frac - 0.1).max(), 0.012, (arm, frac))

    def test_common_random_numbers(self):
        a, b, c = self.sampler("RpPos", 1, 30), self.sampler("RpHO", 1, 30), self.sampler("RpBoth", 1, 30)
        for t in range(30):
            self.assertEqual(a.batch(t)[:8], b.batch(t)[:8])
            self.assertEqual(a.batch(t)[:8], c.batch(t)[:8])
        self.assertNotEqual(mx.schedule_digest(self.sampler("RpPos", 0, 30)), mx.schedule_digest(self.sampler("RpPos", 1, 30)))
        self.assertEqual(mx.schedule_digest(self.sampler("RpPos", 2, 30)), mx.schedule_digest(self.sampler("RpPos", 2, 30)))

    def test_frozen_inputs_unchanged(self):
        ref = json.loads((HERE / "configs/frozen_inputs_sha256.json").read_text())
        for rel, h in ref.items():
            self.assertEqual(hashlib.sha256((HERE.parent / rel).read_bytes()).hexdigest(), h, rel)

    def test_old_experiments_untouched(self):
        repo = HERE.parents[2]
        r = subprocess.run(["git", "status", "--porcelain", "--", "research/sim_free_augmentation/v3", "research/sim_free_augmentation/v3_1",
                            "research/sim_free_augmentation/v3_2", "research/sim_free_augmentation/v2"], cwd=repo, capture_output=True, text=True)
        self.assertEqual([l for l in r.stdout.splitlines() if l.strip()], [])


if __name__ == "__main__":
    unittest.main()
