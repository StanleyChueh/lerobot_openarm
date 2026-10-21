# Real2Sim Milestone 1: dataset and method-provenance audit

**Date:** 2026-10-19 · **Scope:** CPU-only, read-only. No GPU, no Isaac, no installs or downloads, no simulator edits, no robot.
**Status labels:** VERIFIED = checked in this audit against files or data. INFERRED = derived indirectly or taken from repo docs without re-running. UNKNOWN = no evidence. Parameter origin: MEASURED (physical or hardware query) / FITTED (script optimises against data) / HAND (chosen by a person or an assistant session) / ASSET (third-party file).

## 0. Provenance

| Item | Value |
|---|---|
| lerobot_openarm | `official-lerobot-quest` f6f7b50405ec, clean |
| IsaacLab | `real2sim2real-v1` f23a86145c93, clean. The real-look pipeline landed in 163b050b (2026-10-19); battery work in 8d9f4d07 / f23a8614 ("Claude generated") |
| lerobot | `main` 2f2b5679510a, clean (**not** `update_dev`, as the prompt assumed) |
| dora-openarm-data-collection | `dev` 33c0594236f8, clean |
| Real dataset | local `~/.cache/huggingface/lerobot/ethanCSL/openarm_pringles_lerobot_real_v00`. data md5 `7c4fa5e34f…`, info md5 `d4ae42b3b8…`. Hub revision UNKNOWN (no Hub call made) |
| Sim dataset | local `…/openarm_pringles_lerobot_sim_v01_reallook_physfit`. data md5 `bd540af02f…` |
| Isaac Sim version | UNKNOWN (not pinned in the repo) |
| Local models (HF cache, used by the pipeline) | grounding-dino-base rev 12bdfa3120, sam2.1-hiera-large 665f8e2ad6, Depth-Anything-V2-Metric-Indoor-Large-hf d2fc6a9360, dinov2-base f9e44c814b. **The code does not pin revisions** |
| Audit scripts and evidence (temporary, scratchpad) | `/tmp/claude-1000/-home-csl-Stanley-ws-lerobot-openarm/efa713b8-30cb-424f-8b73-130c999cd472/scratchpad/m1/` (`schema_diff.py/.json`, `matched_sheets*.py`, `sheet_*.jpg`, `same_t_*.jpg`, `can_centroid.py/.json`) |

## 1. Bottom line

The `-v1` environment is a **good engineering baseline** for teleop collection, but it does **not yet qualify as a reproducible algorithmic Real2Sim baseline**:
- Body camera and robot appearance were fitted by documented scripts.
- Wrist-camera mounts, table/pad geometry, can geometry, lighting, sensor noise and the can-start choice are hand-authored or selected by outcome.
- All intermediate inputs (`logs/`) and the can USD are gitignored.
- Model revisions and the Isaac version are not pinned.
- No driver script or run manifest exists.

Claude Code is **not** a runtime dependency: no script calls an LLM or API (VERIFIED by grep). However, scene decisions were made in interactive sessions, and those decisions are not reproducible from code.

## 2. Dataset schema and semantics (VERIFIED, `schema_diff.json`)

| Field | real_v00 | sim_v01_reallook_physfit |
|---|---|---|
| Episodes / frames | 50 / 12,115 | 8 / 2,031 |
| fps, measured dt | 30, 0.0333 s (max 0.0333) | 30, 0.0333 s |
| Episode length min/med/max | 188 / 248.5 / 282 | 229 / 251.5 / 280 |
| State/action keys | 16, interleaved `RJ1,LJ1,…,RJ8,LJ8`; action names == state names | identical |
| Cameras | body_cam, wrist_cam (left), right_wrist_cam: 480×640 RGB, AV1, **no depth** | identical |
| Grippers (J8) state range | R −1.277…−0.51, L 0.404…1.19 (motor rad) | R −1.298…−0.803, L 0.628…1.212 |
| Gripper action | R −1.298…0, L 0…1.212; 82 / 27 distinct values (near-binary, not strictly binary) | same range; 15 / 6 distinct |
| Start pose: max deviation from home | 0.032 rad | 0.023 rad |
| End pose: back at home | yes (median 0.03 rad) | yes (0.019 rad) |
| Action→state lag (best shift) | 2 frames; mean err 0.0137 → 0.0086 rad | 2 frames; 0.0099 → 0.0038 rad |
| \|action − state\| arm p95 / max | 0.034 / 0.268 rad | 0.037 / 0.163 rad |
| Arm state speed p99 | 0.95 rad/s | 0.86 rad/s |

**Key finding (VERIFIED):** `sim_v01_reallook_physfit` is **not teleop data**. Its action columns are bit-identical (max |Δ| = 0) to real episodes **0, 2, 3, 4, 5, 6, 9, 10**. It was produced by `record_replayed_episodes.py` (real actions replayed open-loop, with the can placed by `can_starts_use.json`, keeping only replays that met the hand-over condition). Those 8 episodes all lie inside the contact fit's **training** split (0–34), so their success is in-sample. `record_physfit.log` shows 10 saved replays (real eps 0–9) while the local copy has 8 from a different selection. The exact run that produced the local copy is UNKNOWN.

## 3. Camera calibration inventory

| Camera | Intrinsics in sim | Extrinsic in sim | Status |
|---|---|---|---|
| body_cam (D435i) | fx = fy = 607.22 (`rs-enumerate-devices`, MEASURED); principal point forced to (320, 240), real (326.4, 238.7) | FITTED by `calibrate_camera.py` (silhouette chamfer, 123 frames): IoU median 0.33 → 0.71; restarts agree ±2 cm / ±3°. Values **copied by hand** into `stack_ik_abs_visuomotor_env_cfg.py` | VERIFIED (fit json matches cfg) |
| wrist_cam (left) | fx 605.4 (MEASURED) | **HAND**: URDF-derived position plus a rotation "manually tuned in the Isaac Sim GUI" (cfg comment) | Hand-eye UNKNOWN |
| right_wrist_cam | fx 607.8 (MEASURED) | **HAND**: the same values as the left camera | Hand-eye UNKNOWN |

Failed automatic attempts (logs present, VERIFIED):
- **`wrist_handeye.py`** (AX = XB from essential matrices plus FK):
  - Left wrist: residual 1.09°, but moved **17.7 cm** from the cfg.
  - Right wrist: moved **9.5 cm / 9.9°**.
  - The two physically identical mounts disagree, so the result is not trusted.
- **`sfm_handeye.py`** (COLMAP): left moved **71.6 cm / 77.5°**, so it failed (blank walls, soft frames).
- **Extra error source (INFERRED, from the battery README):** a centimetre-level gap between real joint angles and the URDF. Kinematic calibration error would bias any FK-based hand-eye solution.

## 4. Matched real-vs-sim observation tests

**(a) Nearest joint state across datasets** (`sheet_*.jpg`). This test is weak: can pose and episode differ.

**(b) Same episode, same timestep** (`same_t_*.jpg`). Each sim episode is paired with its own source real episode, so actions and can start match. Joint residual is 0.03–0.10 rad.

| Camera | Visual verdict from (b) |
|---|---|
| body_cam | **OK.** Both arms overlay closely. The sim can stands where the real one does (validates the FK-based can start). Grasped can rides lower in the sim hand during transfer. The real can looks wider, consistent with sim 60 mm vs a real can that was never measured |
| right_wrist_cam | **Plausible.** At approach the can is at the left edge in both, and the table-edge direction agrees. During carry both fill the frame. Not quantified |
| wrist_cam (left) | **Offset suspected.** Near rest, the sim shows the pad edge in the lower third where the real camera sees wall and floor (the baked backdrop quad covers only the upper part). At hand-over the sim can sits about ⅓ frame further right. This confounds mount error, kinematic error and can-in-hand pose |

**(c) A quantitative proxy failed.** The HSV red-can centroid test (`can_centroid.json`) is invalid:
- The sim label is dark red with white text, so the threshold misses it: the sim/real area ratio comes out at about 0.12 on wrist frames that visibly fill with can.
- On body_cam, the real robot's orange tags contaminate the mask.

Measuring this needs segmentation (SAM 2, GPU, M2) or a ChArUco capture.

**Prior quantitative results (INFERRED, sysid README, not re-run):**
- DINOv2 paired distance: 0.63 → 0.55 (body), 0.88 → 0.72 (left), 0.86 → 0.75 (right).
- Real-vs-sim linear probe: AUC 1.0.
- Real-action replay carries the can in 22/50 episodes with cfg physics, and 29/50 with the contact fit.

## 5. Minimum viable scene specification (as built)

| Quantity | Value in -v1 | Origin |
|---|---|---|
| Robot kinematics | OpenArm URDF; joint calibration sign +1, offset 0 | ASSET / MEASURED (joint zero). URDF-vs-real error of a few cm reported for battery (INFERRED) |
| Actuators | kp/kd from the MuJoCo XML; optional `OPENARM_ACTUATOR_PROFILE` | ASSET; probe fit available (FITTED, one session) |
| Pad top z | 0.28 m | **HAND.** Comment "17 cm real-world + 2 cm extra, temporarily" does not add up to 0.28. Supported only indirectly by grasp FK; monocular depth disagreed by 15 cm |
| Pad size / near edge | 0.48 × 0.57 m (tape, MEASURED); near edge x = 0.03 | Near edge **HAND** ("12 cm real + 3 cm extra, temporarily") |
| Can | Lightwheel/Berkeley `chips_can_berkeley_meshes.usd` × 0.8 → ⌀60 × 200 mm, asset mass 0.205 kg | ASSET + **HAND** scale. Real can size and mass UNKNOWN. USD is gitignored and its license is UNKNOWN |
| Can start (replay) | `can_starts_use.json` = `jaw15`: right-jaw FK at first closure + 0.15 m along the tool axis | **HAND / outcome-selected** from 0.12 / 0.15 / 0.18 by lift count (8/12). The camera-based estimate (`can_starts.json`, GDINO+SAM) differs by about 4 cm (ep 0: x 0.260 vs 0.220) |
| Can spawn for new collection | x [0.2529, 0.3717] ± margin, y ≈ ±(0.012 + margin) | Taken from the **old sim demo HDF5** (`pickup_pringles_V9_annotated`), not from real data |
| Fingers | Stock OpenArm fingers; the real robot has foam fingertip pads | Geometry gap (visible in real wrist frames). Battery task uses a separate HAND/FITTED pad length |
| Contact physics | can μ 1.20, finger μ 0.224, mass 0.108 (at the 0.10 bound), grip k 636 / c 6.5 / effort 25.7 | FITTED (effective, not physical) |

## 6. Method provenance

| Component | Implementation / math | Input measurements | Human tuning | Model / API dependencies | Re-run command (offline) |
|---|---|---|---|---|---|
| Body camera fitting | `extract_real_frames` → `render_real_states` (Isaac: FK pose, mesh points) → `segment_robot_real` → `calibrate_camera`: 6-DoF mount minimises a truncated two-sided chamfer between projected sim mesh points and the real robot mask; coarse-to-fine with random restarts (**unseeded**) | D435i intrinsics (hardware query); joint states; RGB | Initial mount from earlier hand tuning; frame sampling args; result **pasted by hand** into the cfg; principal point dropped | Grounding DINO base + SAM 2.1 large (local HF, unpinned); Isaac Sim | 4 commands in the sysid README "Camera calibration" block (Isaac render step needs GPU) |
| Wrist camera fitting | Attempted: AX = XB hand-eye (`wrist_handeye`), COLMAP SfM hand-eye (`sfm_handeye`). **Used:** URDF + GUI-tuned rotation | fx from hardware query | **Fully HAND** (the fit was rejected) | pycolmap (attempt only) | Not reproducible: there is no algorithm that produces the cfg values |
| Visual appearance matching | Albedo: closed-loop per-part median colour ratio (`fit_robot_colors`, iterated by hand 0→4). Tone: emissive→pixel curve (`fit_tone`). Body backdrop: median clean plate (`build_clean_plate`) + DA-V2 metric depth scaled by a robot-pixel linear fit + RANSAC table plane (`body_cam_depth`) → mesh (`build_backdrop`). Wrist backdrops: median of rest-pose first frames as a quad (`build_wrist_backdrop`). DINOv2 metric (`eval_realism`) | Real RGB; body fit; joint states | lighting.json **HAND** (README: "by hand"; the `openarm_real_look.py` docstring wrongly says fitted). sensor.json **HAND** (no script writes it). Table forced to z = 0.28 and room flattened to a wall at x = 0.9 (**HAND** overrides of the depth model). Number of iterations and the wrist quad distance chosen by hand | Depth Anything V2 Metric-Indoor-**Large** (license expected CC-BY-NC-4.0, **verify**), DINOv2 (eval only), GDINO + SAM 2.1 | Commands are per script, not chained. `logs/real2sim/look/*` holds the intermediates (gitignored) |
| Contact physics fitting | `prepare_real_replay` (trajectories + can starts) → `fit_contact_physics`: 50 parallel open-loop replays; score = fraction of carry phase with a jaw on the can; random search + ES on eps 0–34, report on 35–49 (seeded) | Real actions; can starts; can geometry (asset) | Search bounds (mass lower bound active); score definition; can-start variant chosen by outcome | Isaac Sim (GPU, headless) | `./isaaclab.sh -p scripts/tools/sysid/fit_contact_physics.py --headless --replay logs/real2sim/replay --can_starts logs/real2sim/replay/can_starts_use.json --out …` |
| Scene output | `openarm_real_look.apply_real_look` patches the -v0 cfg from `data/real2sim/pringles_v00/*` (committed). Cameras are hard-coded in the env cfg. Physics from `contact_physics.json` | — | Pad/table/can placement, can scale, wrist mounts, spawn ranges: HAND in Python constants | Isaac Sim | `lerobot_sim_server.py --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v1 --task_mode handover --domain_randomization none` |

**AI models used (all local, open-weight):** Grounding DINO, SAM 2.1, Depth Anything V2, DINOv2.
**Paid or remote APIs:** none.
**Human or assistant intervention:**
- Every HAND item above.
- Choosing which fit to keep (rejecting the wrist hand-eye, selecting `jaw15`).
- Hand-iterating closed loops.
- Copying fitted values into the cfg.

## 7. Reproducible from documented inputs without Claude Code?

**Partly.** Every algorithmic step is a standalone script with a documented command, and none needs an assistant at runtime. But a fresh clone cannot re-derive the scene:
1. **Inputs not versioned:** `logs/` (frames, masks, poses, fits, can starts) and the can USD are gitignored (`.gitignore:58 **/logs/*`, `:12 **/*.usd`).
2. **No pinning:** HF models load by name without a revision, the Isaac Sim version is unrecorded, and the camera-fit restarts are unseeded.
3. **No pipeline driver or `run_manifest.json`:** the order, iteration counts and accept/reject decisions live in README prose and session history.
4. **Hand-authored values sit in Python constants** (wrist mounts, pad, can scale/placement, lighting, sensor) with no measurement record.
5. **Fitted → cfg transfer is manual** (body mount copied into the cfg).
6. **No held-out-scene validation:** fits used the same 50 episodes, and the sim replay dataset is in-sample.

## 8. What must change to support another task (battery evidence, INFERRED from README)

For the battery task, the body-camera mount, intrinsics, albedo, tone and sensor model were reused unchanged (same rig). The following had to be redone or written new:
- **Backdrops and clean plate:** rebuilt from the new dataset.
- **Object geometry:** new measurement code (top-face sizes from body cam + table plane; thickness from jaw opening).
- **Object texture:** `build_charger_texture.py`.
- **Assets:** a new asset module and env cfg.
- **Gripper:** a task-specific fingertip extension (`gripper.json`, which worsens Pringles).
- **Contact:** a new contact fit (`fit_battery_replay.py`).
- **Demonstrations:** real-action replay could not insert (0/9), so a **scripted expert** replaced replay.

A generic pipeline would need:
- object geometry from declared inputs (CAD or measured dims) through a common script;
- a per-task backdrop job;
- a task-agnostic contact-fit score;
- fingertip geometry as a declared measured input;
- task code (success term, expert) isolated from scene construction.

## 9. Gaps and blockers (cheapest fix first)

| # | Gap | Blocks teleop collection? | Cheapest fix |
|---|---|---|---|
| G1 | Wrist-cam mounts are hand-tuned; left looks offset | Yes, for VLA transfer of wrist views | M2: silhouette/fingertip or segmentation-based wrist fit (GPU) → if that fails, ChArUco hand-eye (≈30 min) |
| G2 | Can size and mass not measured (sim ⌀60 mm, 0.205 kg asset vs 0.108 fitted) | Partly (grasp width, contact) | Tape + scale, ≈5 min, declared input |
| G3 | Pad height and near edge hand-set with "temporary" comments | Partly | Tape measure, ≈5 min, declared input |
| G4 | Real foam fingertips not in the Pringles sim | Partly (contact) | Calipers, ≈5 min; then refit contact |
| G5 | Inputs and models not versioned or pinned; no driver or manifest | No (but blocks the paper claim) | Commit or archive the inputs (HF dataset), pin revisions, add `run_manifest.json` + one driver |
| G6 | Sim "reallook" dataset is replay, not teleop, and in-sample | No | Record fresh teleop sim_v01 (user, M3) |
| G7 | URDF-vs-real kinematic error of centimetres (battery note) | Possibly | Quantify on body-cam silhouettes per link (M2) |
| G8 | Depth Anything V2 Large license likely non-commercial | No (research) | Verify the license; consider the Small (Apache-2.0) variant or dropping the depth model (table taken from measurement G3) |

## 10. Verdict

The current implementation is a **documented engineering baseline with algorithmic components** (body-camera silhouette fit, appearance fits, bounded contact search with a held-out split). It is **not a reproducible algorithmic Real2Sim baseline**: wrist cameras and scene geometry are hand-authored, intermediates and assets are unversioned, models are unpinned, and nothing re-runs end to end from declared inputs. For the immediate goal (a sim environment for teleop collection plus no-DR Mimic), the body camera and physics are usable now. The left wrist camera (G1) and the unmeasured object/table dimensions (G2–G4) are the blockers worth fixing first.

## 11. Next approval request

"APPROVE MILESTONE 2: wrist-camera calibration and measured-input declaration." Proposed scope:
1. Allow GPU for a headless Isaac render and SAM 2.1 (both cached locally, no download).
2. Run a data-only wrist-mount fit with a held-out episode.
3. Ask the user for a ≈15 min tape/scale/caliper measurement (can, pad, fingertips) recorded as a declared-input file.
4. Propose cfg diffs only, with no commit.
