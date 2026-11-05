"""v3.2 regression tests (three-camera generation). Needs the episode-0 wrist flow cache (gen_wrist.py run once)."""
from __future__ import annotations
import json, sys, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
from sfa3 import recon3  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402
from sfa32 import render_wrist as rw, wristcam as wc  # noqa: E402
import gen_wrist as gw  # noqa: E402
import export32 as ex  # noqa: E402

EP = data.load(0)
REC = recon3.build(0, with_plate=False, log=lambda *a: None)
ROBOT = rb.Robot()
SRC, _ = gw.source_geometry(0, EP, ROBOT, "cuda:0")


class Wrist(unittest.TestCase):
    def test_identity_reproduces_the_source_wrist_frames(self):
        c = gw.load_cand(0, "identity__C__s1.0")
        frames = list(range(60, 70)) + list(range(175, 185))
        out = gw.render_candidate(REC, ROBOT, SRC, c, frames)
        for side in "RL":
            for n, i in enumerate(frames):
                j = int(round(i + SRC[side].lag - SRC[side].lag))
                np.testing.assert_array_equal(out[side][0][n], EP.frames[wc.CAM_OF[side]][j])

    def test_translated_camera_changes_the_image(self):
        S = SRC["R"]
        T_new = S.Tc[60].copy()
        T_new[:3, 3] += [0.03, 0.0, 0.0]
        im, lab = rw.render_frame(S, 60, T_new, REC.table, REC.table)
        self.assertGreater(np.abs(im.astype(int) - S.F[60].astype(int)).mean(), 1.0)
        self.assertGreater((lab == rw.LAB["reprojected"]).mean(), 0.3)

    def test_unchanged_wrist_frame_rule_flags_a_copied_source_video(self):
        key = "init_x-3__C__s1.0"
        if not (gw.CACHE / "ep0" / f"wristR_{key}.npy").exists():
            self.skipTest("full render not available")
        c = gw.load_cand(0, key)
        self.assertEqual(sum(ex.unchanged_violations(0, key, c, ROBOT, EP).values()), 0)
        bad = gw.CACHE / "ep0" / "wristR_test_copy.npy"
        np.save(bad, EP.frames[wc.CAM_OF["R"]])
        np.save(gw.CACHE / "ep0" / "wristL_test_copy.npy", EP.frames[wc.CAM_OF["L"]])
        lab = np.load(gw.CACHE / "ep0" / f"wristlab_{key}.npz")
        np.savez_compressed(gw.CACHE / "ep0" / "wristlab_test_copy.npz", L=np.zeros_like(lab["L"]), R=np.zeros_like(lab["R"]))
        try:
            self.assertGreater(ex.unchanged_violations(0, "test_copy", c, ROBOT, EP)["R"], 10)
        finally:
            for p in ("wristR_test_copy.npy", "wristL_test_copy.npy", "wristlab_test_copy.npz"):
                (gw.CACHE / "ep0" / p).unlink()

    def test_mount_rule_recorded(self):
        c = json.loads(wc.CALIB.read_text())
        for s in "RL":
            self.assertIn("rule", c[s]["mount"])
            self.assertEqual(np.array(c[s]["mount"]["R"]).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
