"""pipeline v1 auto targets (from source positions) and partial visibility."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfagen.target_plan import auto_targets


class AutoTargetTests(unittest.TestCase):
    def test_lattice_within_cap_of_sources(self):
        src = {0: np.array([.50, .10]), 1: np.array([.62, -.20])}
        g = auto_targets({"grid_step_m": .03, "max_shift_m": .12}, src)
        d = np.linalg.norm(g[:, None] - np.array(list(src.values()))[None], axis=2).min(1)
        self.assertTrue((d <= .12 + 1e-9).all())
        self.assertTrue(np.allclose(np.round(g / .03), g / .03))             # on the 3 cm lattice
        self.assertEqual(len(np.unique(g, axis=0)), len(g))
        disc = int(np.pi * (.12 / .03) ** 2)                                  # roughly two disjoint 12 cm discs
        self.assertTrue(1.5 * disc < len(g) < 2.5 * disc, len(g))
        g2 = auto_targets({"grid_step_m": .03, "max_shift_m": .12}, dict(reversed(list(src.items()))))
        self.assertTrue(np.array_equal(g, g2))                                # deterministic, order-independent


if __name__ == "__main__":
    unittest.main()
