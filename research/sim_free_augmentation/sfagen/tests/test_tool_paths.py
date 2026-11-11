"""The tools run the tracked code: sfagen from this repo, the published baselines from legacy/ (CPU, imports only)."""
import importlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "tools"))
import _provenance


class ToolPaths(unittest.TestCase):
    def test_xbench_imports_only_tracked_code(self):
        for task, needed in (("pringles", {"mgw", "sfagen.render"}), ("battery", {"srecon", "srender", "sfagen.render"})):
            p = subprocess.run([sys.executable, str(HERE / "tools/xbench.py"), task, "--imports-only"], capture_output=True, text=True, cwd=HERE / "tools")
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            rows = {r["module"]: r for r in json.loads(p.stdout[p.stdout.index("["):])}
            self.assertTrue(needed <= set(rows), (task, needed - set(rows)))
            self.assertTrue(Path(rows["sfagen.render"]["file"]).is_relative_to(HERE))
            for m in needed - {"sfagen.render"}:
                self.assertTrue(rows[m]["in_repo"], rows[m])

    def test_untracked_module_is_refused(self):
        ext = Path(tempfile.mkdtemp()); (ext / "zz_untracked_mod.py").write_text("X = 1\n")
        old = _provenance.EXTERNAL; _provenance.EXTERNAL = ext; sys.path.insert(0, str(ext))
        try:
            importlib.import_module("zz_untracked_mod")
            with self.assertRaises(RuntimeError):
                _provenance.baseline_modules()
        finally:
            _provenance.EXTERNAL = old; sys.path.remove(str(ext)); sys.modules.pop("zz_untracked_mod", None)


if __name__ == "__main__":
    unittest.main()
