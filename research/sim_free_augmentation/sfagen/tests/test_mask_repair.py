"""R05: prompt cut by depth, arm-mask repair and fixed-focal calibration on synthetic robots (CPU, no dataset).
Three URDFs written here: one arm with a rotating jaw, one arm without a gripper block, two arms with prismatic fingers."""
import struct, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfagen import calibrate, mesh
from sfagen.camera import Camera
from sfagen.recon import Recon, box_points
from sfagen.robot import Robot
from sfagen.segment import arm_prompt

TMP = Path(tempfile.mkdtemp())


def box_stl(name, hx, hy, hz):
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (0, 2 * hz)], np.float32)
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    p = TMP / f"{name}.stl"
    p.write_bytes(b"\0" * 80 + struct.pack("<I", 12) + b"".join(struct.pack("<12fH", 0, 0, 0, *v[list(t)].ravel(), 0) for t in f))
    return p


def link(n, stl):
    return f'<link name="{n}"><visual><geometry><mesh filename="{stl}"/></geometry></visual></link>'


def joint(n, parent, child, xyz, kind="revolute", axis="0 1 0"):
    return (f'<joint name="{n}" type="{kind}"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/>'
            f'<axis xyz="{axis}"/><limit lower="-1.5" upper="1.5" effort="9" velocity="9"/></joint>')


def make(kind):
    """-> (robot config, arm names). kind: jaw | none | fingers2 (two arms)."""
    seg, fin = box_stl("seg", .02, .02, .075), box_stl("fin", .006, .02, .03)
    x = ['<robot name="toy">', link("base", seg)]; arms = {}
    for a, x0 in (("L", -.2), ("R", .2)) if kind == "fingers2" else (("A", 0.),):
        prev, names = "base", []
        for i in range(3):
            x += [link(f"{a}{i}", seg), joint(f"{a}j{i}", prev, f"{a}{i}", f"{x0 if i == 0 else 0} 0 .15")]
            prev = f"{a}{i}"; names.append(f"{a}j{i}")
        cfg = {"joints": names, "state_indices": [0, 1, 2], "tcp_frame": prev}
        if kind == "jaw":
            x += [link("jaw", fin), joint("jawj", prev, "jaw", ".03 0 .15")]
            cfg["gripper"] = {"state_index": 3, "open_raw": 1.0, "closed_raw": 0.0, "finger_joints": ["jawj"], "finger_open": 0.6}
        if kind == "fingers2":
            for f_, sx in (("f1", .02), ("f2", -.02)):
                x += [link(a + f_, fin), joint(a + f_ + "j", prev, a + f_, f"{sx} 0 .15", "prismatic", "1 0 0")]
            cfg["gripper"] = {"state_index": 3, "open_raw": 1.0, "closed_raw": 0.0, "finger_joints": [a + "f1j", a + "f2j"], "finger_open": 0.01}
        arms[a] = cfg
    p = TMP / f"{kind}.urdf"; p.write_text("\n".join(x + ["</robot>"]))
    return {"urdf": str(p), "arms": arms}


def cam(W=640):
    c = Camera.__new__(Camera); f = 600 * W / 640; c.W, c.H = W, W * 3 // 4
    c.K = np.array([[f, 0, c.W / 2], [0, f, c.H / 2], [0, 0, 1.]]); c.cfg = {}
    c.R = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]]); c.t = np.array([0., -1.0, .35])   # looks along +y at the arm
    return c


def toy(kind, W=640, objects=()):
    rob = Robot(make(kind)); r = Recon.__new__(Recon); c = cam(W)
    r.robot, r.cam, r.dev, r.arms, r.W, r.H, r.lag = rob, c, "cpu", rob.arms, c.W, c.H, 0.0
    r.G = mesh.Geometry(rob, device="cpu", spacing=0.003)
    r.gv = {s: np.array([[.2, -.3, .25]]) for s in rob.arms}; r.grip_v = {s: np.array([1.0]) for s in rob.arms if rob.grip[s]}
    r.objects = [{"points": box_points([.06, .06, .12])} for _ in objects]; r.obj_pose = lambda k, t: objects[k]
    r.masks = {s: np.zeros((1, c.H, c.W), np.uint8) for s in rob.arms}
    return r


def gripper_geoms(rob, s):
    last = rob.m.getJointId(rob.cfg["arms"][s]["joints"][-1]); out = []
    for i, g in enumerate(rob.vis.geometryObjects):
        j = g.parentJoint
        while j > 0 and j != last:
            j = rob.m.parents[j]
        if j == last:
            out.append(i)
    return out


class MaskRepair(unittest.TestCase):
    def test_gripper_links_belong_to_their_arm_for_any_gripper_type(self):
        for kind, n in (("jaw", 2), ("none", 1), ("fingers2", 3)):
            rob = Robot(make(kind))
            for s in rob.arms:
                g = gripper_geoms(rob, s)
                self.assertEqual(len(g), n, kind); self.assertTrue(all(rob.geom_arm[i] == s for i in g), kind)

    def test_repair_restores_a_deleted_gripper_and_drops_background(self):
        for kind, W in (("jaw", 640), ("none", 640), ("fingers2", 640), ("jaw", 256)):
            r = toy(kind, W); sd = r.robot_sil(0, sides=True); lab = r.scene_labels(0)
            for s in r.arms:
                g = np.isin(lab, gripper_geoms(r.robot, s)); self.assertGreater(g.sum(), 20, kind)
                m = (sd == s) & ~g; m[:6, :6] = True                      # gripper missed, plus a far background blob
                r.masks[s][0] = m
            before = {s: r.masks[s][0].copy() for s in r.arms}; added = r._repair(0, sd, 0.6)
            self.assertGreaterEqual(added, len(r.arms), kind)
            for s in r.arms:
                g = np.isin(lab, gripper_geoms(r.robot, s)); m = r.masks[s][0] > 0
                self.assertGreater((m & g).sum() / g.sum(), 0.55, (kind, W))  # 2 px erosion of a thin link
                self.assertFalse(m[:6, :6].any())                            # background dropped
                self.assertFalse((m & ~(sd == s)).any())                     # nothing outside the projected arm
                keep = (sd == s) & ~g
                self.assertTrue(np.array_equal(m & keep, (before[s] > 0) & keep))  # well covered links untouched
            if len(r.arms) == 2:
                self.assertFalse(((r.masks["L"][0] > 0) & (r.masks["R"][0] > 0)).any())

    def test_object_in_front_owns_its_pixels_and_prompt_keeps_the_gripper(self):
        r0 = toy("jaw"); r0.robot.fk({"A": r0.gv["A"][0]}, {"A": 1.0}); tip = r0.robot.frame(r0.robot.tcp["A"])[:3, 3]
        T = np.eye(4); T[:3, 3] = tip + [.07, -.10, .02]                       # box beside the gripper, nearer to the camera
        r = toy("jaw", objects=[T]); sd = r.robot_sil(0, sides=True); lab = r.scene_labels(0)
        obj = lab >= r.G.ng; self.assertGreater(obj.sum(), 200)
        r.masks["A"][0] = (sd == "A") | obj; r._repair(0, sd, 0.6)
        self.assertFalse(((r.masks["A"][0] > 0) & obj).any())                  # ownership: the proxy in front wins
        g = np.isin(lab, gripper_geoms(r.robot, "A"))
        r1 = toy("jaw"); r1.masks = None; full = (arm_prompt(r1, 0, "A")[g] > 0).mean()   # no object: the hull prompt itself
        self.assertEqual((arm_prompt(r, 0, "A", cut="depth")[g] > 0).mean(), full)       # depth cut keeps every visible gripper pixel
        self.assertLess((arm_prompt(r, 0, "A")[g] > 0).mean(), full - 0.05)              # the 25-px neighbourhood cut removes part of it

    def test_contour_icp_recovers_a_shifted_camera(self):
        import torch
        from scipy.ndimage import distance_transform_edt
        for kind in ("jaw", "fingers2"):
            rob = Robot(make(kind)); true = cam(320); G = mesh.Geometry(rob, device="cpu", spacing=0.003); frames = []
            for q in ([.2, -.3, .25], [-.4, .5, .1], [.6, .2, -.3], [-.1, -.6, .5]):
                qs = {s: np.array(q) for s in rob.arms}; gs = {s: 1.0 for s in rob.arms if rob.grip[s]}
                Wp = G.world(G.placements(qs, gs)); lab = mesh.splat(Wp, G.gid, true, foot=1, dev="cpu")[1].numpy()
                m = cv2.morphologyEx((lab >= 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
                edge = m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0); dist, (iy, ix) = distance_transform_edt(~edge, return_indices=True)
                frames.append((Wp, iy, ix, dist, np.zeros_like(m)))
            P = torch.cat([f[0] for f in frames]).numpy()[::50]          # error measured on the robot's own points (what masks use)
            px = lambda c: float(np.median(np.linalg.norm(c.project(P)[0] - true.project(P)[0], axis=1)))  # noqa: E731
            start = true.perturbed([0.03, -0.02, 0.02, 0.02, -0.015, 0.01, 0]); got = calibrate.contour_icp(G, start, frames, "cpu")
            self.assertLess(px(got), 0.25 * px(start), kind); self.assertLess(px(got), 2.5, kind)   # measured 1.8 px (320 px wide) from 12 px; threshold set after the first run

    def test_match_exposure_removes_a_global_gain_but_keeps_the_object(self):
        from sfagen.recon import match_exposure
        rng = np.random.default_rng(0); ref = rng.integers(20, 180, (120, 160, 3)).astype(np.uint8)   # no saturation after the gain
        img = np.clip(ref.astype(float) * [1.3, 1.2, 1.25] + [8, 5, 0], 0, 255).astype(np.uint8); img[40:70, 50:90] = 250   # brighter frame + one object
        d0 = np.abs(img.astype(int) - ref).max(2) > 30; d1 = np.abs(match_exposure(img, ref).astype(int) - ref).max(2) > 30
        out = np.ones((120, 160), bool); out[40:70, 50:90] = False
        self.assertGreater(d0[out].mean(), 0.3); self.assertLess(d1[out].mean(), 0.02)
        self.assertGreater(d1[40:70, 50:90].mean(), 0.5)

    def test_tracks_compete_per_pixel(self):
        import torch
        from sfagen.segment import owners
        lg = torch.tensor([[[2.0, 1.0, -1.0, -3.0]], [[1.0, 3.0, 0.5, -2.0]]])                    # 2 tracks, 4 pixels
        o = owners(lg); self.assertEqual(o.tolist(), [[[True, False, False, False]], [[False, True, True, False]]])
        self.assertFalse((o[0] & o[1]).any())

    def test_fixed_focal_calibration_keeps_the_focal_length(self):
        rob = Robot(make("jaw")); true = cam(160); G = mesh.Geometry(rob, device="cpu", spacing=0.01); frames = []
        import torch
        for q in ([.2, -.3, .25], [-.4, .5, .1]):
            lab = mesh.splat(G.world(G.placements({"A": np.array(q)}, {"A": 1.0})), G.gid, true, foot=1, dev="cpu")[1]
            arm = torch.tensor([s != "B" for s in G.side] + [False])[lab.clamp(min=0)] & (lab >= 0)
            frames += [{"e": 0, "fit": True, "q": {"A": np.array(q)}, "g": {"A": 1.0}, "fg": arm}, {"e": 1, "fit": False, "q": {"A": np.array(q)}, "g": {"A": 1.0}, "fg": arm}]
        orig = calibrate.load_frames; calibrate.load_frames = lambda ctx, **k: frames
        try:
            for fit_focal in (False, True):
                ctx = SimpleNamespace(robot=rob, cam0=true.perturbed([0, .02, 0, .03, 0, .02, 0]), work=TMP,
                                      cfg={"calibration": {"fit_focal": fit_focal, "maxfev": 16}})
                rep = calibrate.run(ctx, "cpu", log=lambda *a: None)
                self.assertGreaterEqual(rep["fit_iou"]["fitted"], rep["fit_iou"]["initial"])
                if not fit_focal:
                    self.assertEqual(rep["params7"][6], 0.0)
        finally:
            calibrate.load_frames = orig


if __name__ == "__main__":
    unittest.main()
