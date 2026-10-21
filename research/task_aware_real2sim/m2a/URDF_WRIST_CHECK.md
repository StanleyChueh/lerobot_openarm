# Can the URDF provide the wrist-camera transforms? (M2A follow-up, 2026-10-19)

Read-only analysis of files and of live RealSense metadata. No robot motion, no simulator change, no capture, no tuning.

## 1. What the robot description actually contains (VERIFIED)
| Source | Content |
|---|---|
| `v1_camera.urdf` | `openarm_{left,right}_camera_joint`: fixed, parent link7, xyz (0, 0, 0.06), rpy 0. **No optical frame exists anywhere in the URDF.** `ee_tcp` ≡ link7. Joint-7 axis is (0, −1, 0) on the left and (0, +1, 0) on the right (mirrored chains). |
| Camera visual mesh `cam.dae` | "相機架組合" (camera-bracket assembly), identity node transform, mm units (URDF scale 0.001; **left uses −0.001 in y = mirrored**). It contains a D435-sized body (front/back faces ≈ 89 × 27 mm) whose front-face normal in camera_link is **(−0.337, 0, 0.942): tilted 19.7° toward the gripper axis (−x), 0° sideways.** Front-face centroid in link7: **(0.061, 0.000, 0.116) m**. |
| Collision boxes | Axis-aligned (rpy 0): a coarse envelope that does not encode the tilt. |
| `v1_camera_isaac.usd` | Imported links only (camera_link as a plain Xform, rpy 0). No Camera prims. |
| Saved GUI stage `v1_camera_isaac_camera_fixed.usd` (Aug 11) | USD Camera prims (OpenGL convention). wrist_cam: pos (0.06, 0.001, 0.11125), quat (0.19188, 0.68057, 0.68057, 0.19188). **right_wrist_cam: pos (0.05759, 0, 0.11608), quat (0.03701, 0.70614, 0.70614, 0.03701): a different value.** |
| RealSense (live `rs-enumerate-devices -c`) | Distortion coefficients 0. **Depth→colour translation ≈ +15.0 mm along the long axis** on all three cameras, i.e. the RGB lens is off the depth origin. The depth origin's position inside the housing is not in the URDF or CAD. |

## 2. Every manual override between URDF and `CameraCfg`
1. **Rotation:** the URDF gives none. The cfg copies the left GUI value: optical axis tilted **31.5°**. The cfg comment cites "Orient (−180, 6.001, −90)". Under USD `rotateXYZ` that Orient is the **right** GUI value (6° tilt), so the comment does not describe the number in the cfg.
2. **Right camera:** the cfg reuses the **left** camera's pose. The right GUI value differs by 25.5° (it is 6° tilted). Neither GUI value has a measurement record.
3. **Position:** the cfg (0.06, 0.001, 0.11125) is not the URDF's (0, 0, 0.06) that the comment claims. It lies within 5 mm of the CAD front-face centroid, so it is effectively "on the camera face". The lens's lateral offset along the face is not modelled.
4. **Lens:** horizontal aperture set from the measured fx (MEASURED); principal point forced to the centre (renderer limit, ≈0.6–0.8°).

## 3. Can conventions, mirroring or overrides explain the observed error?
- **Optical-frame conversion: no.** The cfg's ad-hoc "(x, −w, −z, y)" rule is algebraically q_usd ⊗ R_x(180°), the correct OpenGL→ROS conversion. cfg ≡ the GUI-left value at 0.0°.
- **Mirrored mounts: not for rotation.** Every candidate's tilt lies in the link x–z plane, which a y-mirror leaves unchanged (mirror(cfg) = cfg at 0.0°). Sharing one rotation between the arms is consistent with mirror-symmetric brackets. Mirroring can only move the RGB lens sideways (image-x), by an unknown amount up to about ±3 cm (≈ ±60 px at 0.3 m).
- **Configuration override: plausibly, but unproven.**

  | Candidate | Tilt | Angle to CAD |
  |---|---|---|
  | CAD | 19.7° | — |
  | cfg / GUI-left | 31.5° | 11.8° (a pure pitch about image x ≈ 126 px vertical) |
  | GUI-right | 6° | 13.7° |
  | Earlier real-data AX=XB rotations (independent, rotation residual 1.1° / 1.8°) | 29° (left), 39° (right) | 10.4° / 20.3° |

  The AX=XB rotations sit **closer to the cfg than to the CAD**. They disagree with each other by 15° and come with implausible translations.

  The sim-vs-real rest view (sim shows the pad edge where real shows wall) would move the right way under the CAD pitch. However, the pad geometry is hand-set, so this is **not evidence**. The three sources (CAD, GUI tuning, AX=XB) disagree by 10–20°, and only a real measurement can decide.

## 4. Would "using the URDF directly" fix it?
**Not literally.** The URDF camera_link has rpy 0 and no optical frame. Treated as an optical frame it is 94° off (z-forward) or about 120° off (REP-103 x-forward). What the robot description *can* provide is a **CAD-derived transform**, computed algorithmically and with no tuning:
- **Rotation:** forward = the CAD front-face normal; image-x = link +y (the face's long axis; upright sign from the real images).
- **Position:** the CAD face centroid plus the lens offset.

```
candidate X_cad (parent openarm_{left,right}_ee_tcp, ROS):  pos (0.0609, -0.0001, 0.1156)   rot wxyz (0.69671, -0.12082, -0.12082, 0.69671)
known gaps: RGB lens lateral offset along the face (unknown, <= ~3 cm), lens depth behind the face (mm), CAD-vs-as-built tolerance
```
Whether X_cad is right depends on the bracket being built as drawn. That can only be shown with a real fiducial.

## 5. Smallest independent real-world validation (≈ 15 min, operator-run, static poses)
- **Target:** one ChArUco board, A4 or A3, `DICT_5X5_100`, square size **measured with calipers**. Taped flat on the pad near the hand-over area; it never moves.
- **Poses:** **5 static poses per arm**, reached by slow teleop near the demo workspace, gripper open, no contact. Pose 0 = the recorded home/rest pose. The other four differ by ≥ 30° in wrist orientation about two axes, with board distance 0.25–0.5 m. Hold ≥ 1 s per pose. Record with `lerobot-record`, so each pose has joint state plus all 3 cameras. The body camera should see the board too.
- **Analysis (OpenCV + `urdf_fk.py`, no Claude):**
  1. Per pose: ChArUco corners → `solvePnP` with the factory intrinsics → T_cam←board.
  2. For a candidate X: T_base←board(k) = FK(q_k) · X · T_cam←board(k).
  3. If X is right, this is constant over k. Score the spread and the leave-one-pose-out corner reprojection.
  4. Evaluate X_cad and X_cfg side by side.
  5. A free 5-pose `cv2.calibrateHandEye` solve (Tsai / Park / Daniilidis) is informational only.

## 6. Pre-registered PASS / BLOCK for skipping full hand-eye calibration (per camera)
| # | Criterion | Threshold |
|---|---|---|
| Q | Detection / intrinsics sanity | ≥ 20 corners per pose; PnP reprojection RMS ≤ 1.0 px |
| P1 | Board-in-base consistency under candidate X | translation std ≤ 5 mm, rotation std ≤ 1.0° over 5 poses |
| P2 | Leave-one-pose-out reprojection of the board corners | median RMS ≤ 6 px, worst pose ≤ 12 px (≈ 0.6° / 1.1°) |
| P3 | Agreement with the free 5-pose solve (only when the methods agree with each other within 1 cm / 1°) | ≤ 1 cm and ≤ 1.5° |
| F | FK sanity: board pose from the body cam vs the wrist chain | ≤ 2 cm (informational; above it, kinematic error is flagged) |

- **PASS** (all of Q, P1, P2, P3 for the candidate) → adopt that candidate with its residuals reported (outcome A/B). If X_cad passes, the transform is derived from robot geometry and only validated, never tuned.
- **BLOCK → full ChArUco calibration (outcome C)** if both candidates fail P1/P2, if the free solve is unstable, or if F > 2 cm (the FK error must then be modelled too).

## 7. Verdict
- **A (URDF directly): no.** The URDF has no optical frame.
- **B (configuration correction): the most plausible outcome, not yet established.** The CAD-derived transform replaces two undocumented GUI overrides with geometry from the robot description. It changes the pitch by 11.8° and the position by 0.5 cm, and leaves the lens's lateral offset unknown. It must pass the 5-pose test in §6 before it may enter the cfg.
- **C (full 20–25-pose ChArUco):** needed only if the 5-pose test blocks. Since the 5-pose capture uses the same board and recording path, its data also seeds the full calibration.
