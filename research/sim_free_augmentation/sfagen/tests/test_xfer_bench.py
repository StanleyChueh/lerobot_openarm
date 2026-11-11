"""R1 shift-limit benchmark: phase alignment, binning and the frozen r* decision rule (CPU, synthetic numbers)."""
import itertools, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import xfer_bench as xb


def pairs_with(value_of_bin, copy=10.0):
    """90 ordered pairs over 10 episodes; bin assigned cyclically so every bin has pairs from many episodes."""
    out = []
    for i, (a, b) in enumerate(itertools.permutations(range(10), 2)):
        k = i % (len(xb.EDGES) - 1)
        out.append({"a": a, "b": b, "bin": k, "render": value_of_bin(k) + 0.01 * ((a * 7 + b) % 5), "copy": copy})
    return out


class XferBench(unittest.TestCase):
    def test_phase_frames_ordered_and_clipped(self):
        f = xb.video_frames(t_grasp=97, t_left_close=176, lag=-0.5, T=273)
        self.assertEqual(list(f), list(xb.PHASES))
        mids = [f[p][1] for p in xb.PHASES]
        self.assertEqual(mids, sorted(mids))                                   # start < approach < grasp < lift
        self.assertEqual(f["grasp"][1], round(97 + 0.5))                       # video frame = state time - lag
        self.assertTrue(all(0 <= j < 273 for v in f.values() for j in v))
        g = xb.video_frames(t_grasp=10, t_left_close=12, lag=-3, T=12)         # clipped to the episode
        self.assertTrue(all(0 <= j <= 11 for v in g.values() for j in v))

    def test_bins_cover_edges(self):
        self.assertEqual(xb.bin_of(0.0), 0); self.assertEqual(xb.bin_of(2.99), 0); self.assertEqual(xb.bin_of(3.0), 1)
        self.assertEqual(xb.bin_of(14.9), 5); self.assertEqual(xb.bin_of(21.0), 6); self.assertIsNone(xb.bin_of(21.5))

    def test_r_star_stops_at_first_large_drop(self):
        rows, r = xb.bin_stats(pairs_with(lambda k: 20.0 if k < 4 else 18.0), n_boot=200)   # bins [0,9) flat, then -2 dB
        self.assertEqual(r, 9); self.assertTrue(all(x["passes"] for x in rows[1:4])); self.assertFalse(rows[4]["passes"])

    def test_small_drop_within_half_db_passes_everywhere(self):
        rows, r = xb.bin_stats(pairs_with(lambda k: 20.0 - 0.05 * k), n_boot=200)
        self.assertEqual(r, 21); self.assertTrue(xb.decision(r).startswith("A"))

    def test_render_not_better_than_copy_fails(self):
        rows, r = xb.bin_stats(pairs_with(lambda k: 20.0, copy=25.0), n_boot=200)
        self.assertEqual(r, 3); self.assertTrue(xb.decision(r).startswith("B"))

    def test_psnr_region(self):
        a = np.zeros((4, 4, 3), np.uint8); b = a.copy(); b[0, 0] = 10; m = np.zeros((4, 4), bool); m[0, 0] = True
        self.assertAlmostEqual(xb.psnr(a, b, m), 10 * np.log10(255 ** 2 / 100), places=6)
        self.assertIsNone(xb.psnr(a, b, np.zeros((4, 4), bool)))


if __name__ == "__main__":
    unittest.main()
