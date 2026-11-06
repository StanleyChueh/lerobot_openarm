"""v4B.1 implementation tests (CPU).

    cd research/sim_free_augmentation/v4b1
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
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sfa4b1 import interventions as iv  # noqa: E402
from sfa.data import command_events  # noqa: E402

mx = iv.mx
V4, V31 = HERE.parent / "v4", HERE.parent / "v3_1"
PHASES_V31 = ("approach", "grasp", "carry", "handover", "release")


def load_ds(name):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(f"local/{name}", root=V31 / "outputs/lerobot" / name, delta_timestamps={"action": [k / 30 for k in range(50)]})


def phase_of(T, ev):
    rc, lc, ro = ev["right_close"], ev["left_close"], ev["right_open"]
    return np.array(["approach" if t < rc - 5 else "grasp" if t < rc + 15 else "carry" if t < lc - 5 else
                     "handover" if t < ro + 5 else "release" for t in range(T)])


class TestV4B1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dr, cls.ds = load_ds("sfa_v31_R"), load_ds("sfa_v31_Gp_matched_evidence_complete")
        cls.sel = mx.selection()
        cls.rr, cls.sr = mx.episode_ranges(cls.dr.meta), mx.episode_ranges(cls.ds.meta)
        cls.nR = len(cls.dr)
        cls.A = np.stack(cls.ds.hf_dataset["action"])

    def test_frozen_inputs_and_v4_untouched(self):
        ref = json.loads((V4 / "configs/frozen_inputs_sha256.json").read_text())
        for rel, h in ref.items():
            self.assertEqual(hashlib.sha256((HERE.parent / rel).read_bytes()).hexdigest(), h, rel)
        r = subprocess.run(["git", "status", "--porcelain", "--", "research/sim_free_augmentation/v4", "research/sim_free_augmentation/v3_1",
                            "research/sim_free_augmentation/v3"], cwd=HERE.parents[2], capture_output=True, text=True)
        self.assertEqual([l for l in r.stdout.splitlines() if l.strip()], [])

    def test_schedules_equal_frozen_v4b(self):
        """sample-pair equality: every v4B.1 arm draws exactly the frozen v4B batches of its base arm."""
        for arm, spec in iv.CFG["arms"].items():
            for seed in iv.CFG["seeds"]:
                real, syn = mx.frame_pools(spec["base"], self.sel, self.rr, self.sr, self.nR)
                s = mx.StratifiedBatches(spec["base"], self.sel, real, syn, 1000, 16, seed, mx.CFG["training"]["sampler_seed"], 0.5)
                frozen = json.loads((V4 / f"outputs/v4b/train/{spec['base']}_s{seed}_mixture.json").read_text())["sampler"]["schedule_sha256"]
                self.assertEqual(mx.schedule_digest(s), frozen, (arm, seed))

    def test_action_targets_identical_and_aligned(self):
        for arm in iv.CFG["arms"]:
            d1, d0 = iv.TwoPartV41(self.dr, self.ds, arm, 0), mx.TwoPart(self.dr, self.ds)
            for i in (3, 260, self.nR + 7, self.nR + 150, self.nR + 23600):
                a, b = d1[i], d0[i]
                for k in ("action", "observation.state", "action_is_pad", "observation.images.body_cam"):
                    self.assertTrue(torch.equal(a[k], b[k]), (arm, i, k))
                if i >= self.nR:
                    j, f = i - self.nR, int(a["frame_index"])
                    e = int(a["episode_index"])
                    lo, hi = self.sr[e]
                    n = min(50, hi - lo - f)
                    np.testing.assert_array_equal(a["action"][:n].numpy(), self.A[j:j + n])
                    self.assertTrue(a["action_is_pad"][n:].all() and not a["action_is_pad"][:n].any())

    def test_phase_mask_boundaries_and_grippers(self):
        d = iv.TwoPartV41(self.dr, self.ds, "RpPos_PhaseMask", 0)
        for E, eps in self.sel["arms"]["RpPos"].items():
            for e in eps:
                lo, hi = self.sr[e]
                ev = command_events(self.A[lo:hi])
                ph = phase_of(hi - lo, ev)
                ge = ev["right_close"] + 15
                self.assertEqual(d.grasp_end[e], ge)
                for f in (0, max(0, ge - 51), ge - 50, ge - 10, ge - 1, ge, hi - lo - 1):
                    if not 0 <= f < hi - lo:
                        continue
                    m = d[self.nR + lo + f]["sfa_target_mask"].numpy()
                    for k in range(50):
                        t = f + k
                        if t < hi - lo:
                            self.assertEqual(bool(m[k]), ph[t] not in ("approach", "grasp"), (e, f, k, ph[t]))
                # gripper events: right close supervised, left close / right open excluded
                self.assertLess(ev["right_close"], ge)
                self.assertGreaterEqual(ev["left_close"], ge)
                self.assertGreaterEqual(ev["right_open"], ge)
        for i in (0, 100, 2000):
            self.assertFalse(d[i]["sfa_target_mask"].any())

    def test_random_mask_rate_and_real_untouched(self):
        d = iv.TwoPartV41(self.dr, self.ds, "RpPos_RandMask", 1)
        ms = np.stack([d[self.nR + j]["sfa_target_mask"].numpy() for j in range(0, 23000, 97)])
        self.assertLess(abs(ms.mean() - iv.CFG["p_drop"]), 0.01)
        self.assertTrue(np.array_equal(d[self.nR + 5]["sfa_target_mask"].numpy(), d[self.nR + 5]["sfa_target_mask"].numpy()))
        for i in (0, 1234):
            self.assertFalse(d[i]["sfa_target_mask"].any())

    def test_applier_image_and_mask_effects(self):
        def fake_pre(batch, camera_keys, rename_map, preprocessor):
            out = {k: v for k, v in batch.items()}
            out[iv.CAM] = out.pop("observation.images.body_cam")
            return out
        B = 4
        is_syn = torch.tensor([False, True, False, True])
        tmask = torch.zeros(B, 50, dtype=torch.bool)
        tmask[1, 20:] = True
        tmask[0, 10:] = True                     # would be a bug if applied to a real row
        base = {"observation.images.body_cam": torch.rand(B, 3, 480, 640), "observation.state": torch.zeros(B, 16),
                "action_is_pad": torch.zeros(B, 50, dtype=torch.bool)}
        for arm, spec in iv.CFG["arms"].items():
            ap = iv.Applier(arm, fake_pre, "/dev/null")
            b = {k: v.clone() for k, v in base.items()} | {"sfa_is_synth": is_syn, "sfa_target_mask": tmask}
            out = ap(b, [], {}, None)
            self.assertEqual(tuple(out[iv.CAM].shape), (B, 3, 480, 640))
            self.assertTrue(torch.equal(out[iv.CAM][~is_syn], base["observation.images.body_cam"][~is_syn]))
            self.assertFalse(out["action_is_pad"][0].any())                         # real row never masked
            if spec["synth_rgb"] == "missing":
                self.assertTrue((out[iv.CAM][is_syn] == 0).all())
                self.assertTrue(torch.equal(out[f"{iv.CAM}_padding_mask"], ~is_syn))
            else:
                self.assertTrue(torch.equal(out[iv.CAM], base["observation.images.body_cam"]))
                self.assertNotIn(f"{iv.CAM}_padding_mask", out)
            if spec["target_mask"] != "none":
                self.assertEqual(int(out["action_is_pad"][1].sum()), 30)
            else:
                self.assertFalse(out["action_is_pad"].any())
            ap.rows.clear()


if __name__ == "__main__":
    unittest.main()
