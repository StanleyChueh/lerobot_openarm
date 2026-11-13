"""Presentation checks: uncertainty at extreme counts and evidence immutability."""
import hashlib
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from visualize_eval_v00 import success_interval, RESULT, load


class VisualDiagnosisTest(unittest.TestCase):
    def test_zero_successes_still_have_uncertainty(self):
        value, lo, hi = success_interval(0, 4)
        self.assertEqual((value, lo), (0, 0))
        self.assertGreater(hi, .5)

    def test_perfect_small_sample_is_not_certain(self):
        value, lo, hi = success_interval(5, 5)
        self.assertEqual((value, hi), (1, 1))
        self.assertLess(lo, .5)

    def test_visual_build_did_not_change_saved_evidence(self):
        path = RESULT / "visual_build.json"
        if not path.exists():
            self.skipTest("Build visual report before running artifact checks")
        report = load(path)
        repo = RESULT.parents[4]
        for file, sha in report["experimental_input_hashes"].items():
            self.assertEqual(hashlib.sha256((repo / file).read_bytes()).hexdigest(), sha, file)
        self.assertFalse(report["experimental_results_changed"])
        self.assertEqual(len(list((RESULT / "charts").glob("*.svg"))), 8)
        self.assertEqual(len(list((RESULT / "charts").glob("*.png"))), 8)

    def test_report_local_links_and_chart_captions(self):
        page = RESULT / "index.html"
        if not page.exists():
            self.skipTest("Build visual report first")
        class Links(HTMLParser):
            def __init__(self):
                super().__init__(); self.targets = []; self.captions = 0
            def handle_starttag(self, tag, attrs):
                if tag == "figcaption":
                    self.captions += 1
                for key, value in attrs:
                    if key in ("href", "src", "data-zoom") and value and not value.startswith(("#", "http")):
                        self.targets.append(value)
        parser = Links(); parser.feed(page.read_text())
        for target in parser.targets:
            self.assertTrue((RESULT / target).exists(), target)
        self.assertEqual(parser.captions, 8)


if __name__ == "__main__":
    unittest.main()
