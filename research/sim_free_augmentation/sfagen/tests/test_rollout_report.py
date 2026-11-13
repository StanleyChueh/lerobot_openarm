"""Small independent checks for image detection and temporal event handling."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from rollout_report import lid, first_run, first_close


class RolloutReportTest(unittest.TestCase):
    def test_lid_ignores_small_orange_robot_parts_and_tracks_missing(self):
        im = np.zeros((480, 640, 3), np.uint8)
        cv2.ellipse(im, (330, 210), (45, 35), 0, 0, 360, (240, 180, 40), -1)
        cv2.rectangle(im, (20, 20), (50, 30), (240, 160, 20), -1)
        uv, area = lid(im)
        np.testing.assert_allclose(uv, [330, 210], atol=1)
        self.assertIsNone(lid(np.zeros_like(im), uv, area))
        self.assertIsNone(lid(im, [100, 100], area))

    def test_persistence_does_not_bridge_occlusion(self):
        self.assertEqual(first_run([False, True, True, True, False]), 1)
        self.assertIsNone(first_run([True, True, False, True, True]))
        self.assertIsNone(first_run([]))

    def test_close_missing_is_explicit(self):
        class NoClose:
            def gripper_events(self, values):
                return {"R": {"closes": []}}
        self.assertIsNone(first_close(NoClose(), np.zeros((3, 16))))

    def test_camera_and_object_translation_cancel(self):
        from eval_v00_wrist import relative_pixel
        C = np.eye(4); K = np.diag([600., 600., 1.])
        P = np.array([[0.01, 0.02, 0.2], [-0.01, 0.01, 0.3]])
        delta = np.array([0.06, -0.03, 0.0]); D = C.copy(); D[:3, 3] = delta
        uv, _ = relative_pixel(C, P, K)
        translated, _ = relative_pixel(D, P + delta, K)
        np.testing.assert_allclose(uv, translated, atol=1e-10)

    def test_wrist_phase_boundaries(self):
        from eval_v00_wrist import phase
        self.assertEqual(phase(60, 100, 160), "final_approach")
        self.assertEqual(phase(95, 100, 160), "grasp")
        self.assertEqual(phase(155, 100, 160), "handover")
        self.assertIsNone(phase(120, 100, 160))


if __name__ == "__main__":
    unittest.main()
