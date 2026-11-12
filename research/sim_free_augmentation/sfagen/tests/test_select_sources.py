"""C1 selection rule (CPU)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from select_sources import kcenter


class KCenter(unittest.TestCase):
    def test_centre_first_then_farthest(self):
        P = [[0, 0], [10, 0], [-10, 0], [0, 1], [0, 9]]
        self.assertEqual(kcenter(P, 2), [3, 1])                 # centroid (0, 2) -> (0, 1); farthest: (10, 0) (tie with (-10, 0): lower index)
    def test_nested_and_unique(self):
        P = [[x, (x * 7) % 5] for x in range(10)]
        s = kcenter(P, 10); self.assertEqual(sorted(s), list(range(10))); self.assertEqual(kcenter(P, 4), s[:4])


if __name__ == "__main__":
    unittest.main()
