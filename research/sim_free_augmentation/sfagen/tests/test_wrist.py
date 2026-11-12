"""Wrist renderer geometry (CPU): element motions and two-view triangulation."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfagen.wrist import WristRenderer, triangulate

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def pose(t=(0, 0, 0)):
    T = np.eye(4); T[:3, 3] = t; return T


class Wrist(unittest.TestCase):
    def test_unchanged_scene_gives_identity_for_every_element(self):
        Ms = np.stack([pose((0.1 * i, 0, 0.5)) for i in range(3)]); C = pose((0, 0, 1))
        D = WristRenderer._deltas(None, {"Ms": Ms, "C": C}, Ms, C)
        self.assertEqual(D.shape, (4, 4, 4)); np.testing.assert_allclose(D, np.tile(np.eye(4), (4, 1, 1)), atol=1e-12)

    def test_part_rigid_with_camera_is_identity_static_world_is_not(self):
        Ms = np.stack([pose((0, 0, 0.3)), pose((1, 0, 0))]); C = pose((0, 0, 0.5)); d = np.array([0.05, -0.02, 0.0])
        Mt = Ms.copy(); Mt[0, :3, 3] += d; Ct = C.copy(); Ct[:3, 3] += d        # part 0 and camera translate together
        D = WristRenderer._deltas(None, {"Ms": Ms, "C": C}, Mt, Ct)
        np.testing.assert_allclose(D[0], np.eye(4), atol=1e-12)                # moves with the camera: unchanged in view
        np.testing.assert_allclose(D[1][:3, 3], -d, atol=1e-12)                 # static part: opposite camera motion
        np.testing.assert_allclose(D[-1][:3, 3], -d, atol=1e-12)                # static world

    def test_triangulation_recovers_plane_depth_under_translation(self):
        Z, tx = 0.6, 0.03; H, W = 48, 64; Kc = K.copy(); Kc[0, 2], Kc[1, 2] = W / 2, H / 2
        flow = np.zeros((H, W, 2)); flow[..., 0] = -Kc[0, 0] * tx / Z               # camera k = camera j moved +tx along x
        A = pose((tx, 0, 0)); d = triangulate(flow, A, Kc, far=2.5)
        np.testing.assert_allclose(np.median(d), Z, rtol=1e-3)

    def test_no_parallax_falls_back_to_far(self):
        d = triangulate(np.zeros((20, 30, 2)), pose(), K, far=2.5); self.assertTrue(np.allclose(d, 2.5))


if __name__ == "__main__":
    unittest.main()
