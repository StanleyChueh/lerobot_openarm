# HANDOFF: sim-free real-to-real augmentation for OpenArm (state as of 2026-11-09)

**Who this is for:** any agent (Codex, Claude) or person continuing this work. Read this first; it tells you what exists,
where it is, how it was validated, what is weak, and what to do next.

Detailed method: `PIPELINE.md`. Novelty check: `NOVELTY.md`. How to run: `README.md`.

## 1. One-paragraph summary

We turn ~10 real bimanual teleoperation demos (LeRobot v3.0, 16-D OpenArm state/action, 3 cameras) into ~100 new,
replay-checked demos per task. There is no simulator, no generative image model and no manual labels.

**How it works:**
- **Trajectories** are kinematic edits of the recorded ones: an object moves and the grasping arm follows, or the whole two-arm interaction moves.
- **Body-camera images** are real pixels moved in 3-D with the robot model (mesh-guided forward warping).
- **Wrist cameras** keep the real recorded frames, with validity flags.

**Scope.** One config-driven pipeline, **`sfagen`**, produces BOTH published task datasets with identical code and
settings; only a short task config differs. It is designed for any URDF robot but has only been run on OpenArm.

## 2. Published artifacts (Hugging Face, all public, tag `v3.0` = current)

| repo | content (v3.0) | previous revision tags |
|---|---|---|
| `ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00` | 96 synthetic eps, 24,645 frames (29 can-position, 67 hand-over-location) | `v04-65ep-two-pipelines`, `v03-55ep-mgw`, `v02-20ep-v3render`, `v01-25ep-black-wrist`, `v00-35ep-with-real` |
| `ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00` | 10 real (eps 0–9, original videos) + 96 synthetic; per-frame `is_synthetic` | `v00-75ep-two-pipelines` |
| `ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00` | 114 synthetic eps, 84,690 frames (62 battery/station-position, 52 insertion-location) | `v01-65ep-two-pipelines`, `v00-36ep-inherited-cam` |
| `ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00` | 10 real + 114 synthetic; per-frame `is_synthetic` | `v00-75ep-two-pipelines` |

**Sources:**
- Pringles: `ethanCSL/openarm_pringles_lerobot_real_10_v00` @ `cebdee98`.
- Battery: `ethanCSL/openarm_insertbattery_lerobot_real_10_v02` @ `a1c29d77`.

**Evaluation pools** (other real recordings, never used for generation):
- Pringles: `openarm_pringles_lerobot_real_v00` episodes 10–49.
- Battery: `openarm_insertbattery_lerobot_real_v02`, 45 non-source episodes.

**Other files in each Hub repo:**
- `PIPELINE.md` and `NOVELTY.md`;
- `sfagen/` (code and configs);
- `comparison_videos/` (real left / synthetic right; files named `*_unified_*` are current);
- `assets/coverage_*.png`.

**Not changed:** the older 1-camera repo `ethanCSL/openarm_pringles_lerobot_real_to_sim_10_v00` still contains 10 real + 25 synthetic episodes. A synthetic-only rebuild exists, but the push was never done.

## 3. Where things are on disk (machine `stanley`, 2× RTX 4090, 24 cores)

| path | what |
|---|---|
| `~/Stanley_ws/sfa_hf_export/sfagen/` | **current pipeline** (also committed to GitHub, see §9) |
| `.../sfagen/configs/{_openarm_robot,openarm_pringles_handover,openarm_battery_insertion}.yaml` | robot + task configs |
| `.../sfagen/configs/work/<config>/` | not in git: `calibration.json`, `table_height.json`, `cache/` (reconstructions, SAM2 masks), `candidates/`, `body/` (renders), `dataset/`, `dataset_mixed/`, `profile.jsonl`, `verification.json`, comparison videos |
| `.../sfagen/logs/` | evaluation scripts and results: `xbench.py`, `coverage.py`, `make_cards.py`, `push_unified.py`, `xbench_*_final.json`, `coverage.json` |
| `~/Stanley_ws/sfa_hf_export/{render_v4,battery,target,3cam_v3,3cam_v4,battery_v01}` | **superseded** task-specific pipelines that made the previous revisions (kept for the comparison benchmark) |
| `~/Stanley_ws/lerobot_openarm/research/sim_free_augmentation/{sfa,v2,v3,v3_1,…}` | frozen earlier research studies (read-only; Pringles v3.1 planner, SAM2 study, lid tracker) |
| `~/Stanley_ws/sfa_hf_export/videos/` | older comparison videos (previous revisions) |

**Python env:** `env -u PYTHONPATH -u LD_LIBRARY_PATH ~/Stanley_ws/lerobot_openarm/.venv/bin/python`. The LeRobot there has SmolVLA, GR00T N1.7 and π0.5.

## 4. The method (the same for every task)

Full details and formulas: `PIPELINE.md`.

**Once per rig:**
- **A. Camera.** Render-and-compare IoU fit of the URDF silhouette to a camera-independent foreground (near-static frames, fit on even episodes, held-out on odd ones).
- **B. Table height.** Median of known-size-object silhouette fits if the median IoU ≥ 0.5; else the lowest arm point − 5 mm.

**Per source demo (C, reconstruction):**
1. **Robot masks.** SAM2.1-small video segmentation, prompted every 15 frames with the projected URDF hulls of each arm, in 150-frame chunks.
2. **Clean plate.** The temporal median of robot- and object-free pixels.
3. **Objects.** Segmented as "what disappears when picked up".
   - Known-size cylinders: grasp-constrained render-and-compare. The centre is TCP + s·approach + l·closing axis; a lying cylinder's axis is ⟂ to the finger-closing axis.
   - Unknown objects: an oriented box from the segment.
4. **Holders.** Table → grasping hand → hand-over / release.
5. **Photometric time offset.** The offset that best re-renders real frames from nearby real frames.

**Variations (D):**
- **F1 object position:** ±3/±6 cm; the grasping arm follows the object; it rejoins the recording at the lift apex.
- **F2 interaction location:** ±3/±5 cm; all arms and held objects move together during the two-arm window.
- **Joints:** bounded damped-least-squares IK, σ = 3 smoothing, then contact-exact IK refinement on full-offset frames.
- **Actions:** a′ = a + Δq(t + lead).

**Replay checks (E)**, on the commanded actions:
- limits (URDF ∪ the demo's own range);
- speed ≤ 1.25× and acceleration ≤ 1.5× the real envelope;
- mesh clearance and hand distance not below the source;
- table clearance;
- start pose;
- untouched parts bit-identical;
- IK ≤ 1 mm / 0.5° at contact (≤ 5 mm / 2° while blending);
- object placement, including the robot-proximity rule.

**Rendering and output (F–H):**
- **F. Body camera:** mesh-guided forward warping of real pixels with the relit plate (σ = 10), hidden surfaces borrowed from ±3k frames, Telea fill, despeckle; `part_fill` off.
- **G. Wrist cameras:** real frames plus `wrist_view_valid` and `wrist_view_pose_offset`.
- **H. Export:** LeRobot v3.0, plus a real + synthetic mix (`add_features` + `merge_datasets`, real videos not re-encoded). Verification covers SHA-256 labels, no real-episode copies, decode checks and the `is_synthetic` flag.

**Every design choice was decided by data.** The real-ground-truth pose-transfer benchmark is run on development episodes; results are reported on held-out episodes.

| decision | evidence |
|---|---|
| fitted camera (battery) | +0.41 dB [0.26, 0.49] held-out vs inherited camera |
| plate relighting | battery +0.14 [0.12, 0.16]; Pringles +0.05 |
| SAM2 robot masks (vs mesh-only) | Pringles held-out gap to the published renderer −0.48 → −0.30 dB; battery +0.53 dB vs published |
| photometric time offset | Pringles gap −0.30 → +0.17; silhouette-based offset scans were flat (≤ 0.004 IoU) |
| part_fill off | dev: Pringles −0.09 [−0.11, −0.07] with it; battery +0.01 (n.s.) |
| contact-exact IK refinement | Pringles passing candidates 41 → 96, battery 101 → 114 (failures were 1–3 mm smoothing residuals) |

## 5. Results

**Renderer fidelity on held-out episodes.** Re-render real frame j+k from j; the scoring region is pipeline-independent; 95% episode-bootstrap CI.

| task | copy source frame | previous published renderer | **sfagen** | sfagen − previous |
|---|---:|---:|---:|---|
| Pringles (eps 0,2,3,4,6,7,9; 211 pairs) | 14.51 dB | 17.87 dB / SSIM 0.692 | **18.04 dB / 0.703** | +0.17 [−0.06, 0.41], better in 55% |
| battery (eps 3,5,8,9; 284 pairs) | 16.28 dB | 18.40 dB / 0.681 | **19.21 dB / 0.709** | +0.82 [0.64, 1.00], better in 74% |

**Coverage.** Other real recordings with a training position within 2 cm, out of 40 (Pringles) or 45 (battery). Positions by FK at gripper events.

| position | 10 sources | previous revision | **sfagen** |
|---|---:|---:|---:|
| Pringles can | 9 | 29 | 24 (lower: robot-proximity rule) |
| Pringles hand-over point | 33 | 38 | **40** (area 124 vs 35 cm²) |
| battery | 22 | 41 | 41 |
| station | 16 | 29 | **31** |
| insertion point | 34 | 34 | **43** |

**Verification** (`configs/work/*/verification.json`): all of these hold:
- the datasets load in LeRobot;
- labels SHA-256 OK;
- 0 episodes equal a real one;
- all replay checks pass;
- no black frames;
- body camera ≈ 39–41 dB vs the renders;
- mixed datasets: real labels identical to the source; `is_synthetic` correct.

**Training compatibility.** 5-step smoke tests on this LeRobot:

| policy | result |
|---|---|
| SmolVLA (`lerobot/smolvla_base`) | trains; `--rename_map` body_cam/wrist_cam/right_wrist_cam → camera1/2/3 |
| GR00T N1.7 | trains only with `--policy.lora_rank=16 --policy.tune_diffusion_model=false` (full fine-tune runs out of memory on 24 GB) |
| π0.5 | data loads; full 4 B fine-tune runs out of memory on 24 GB (needs a bigger GPU, or LoRA / a frozen backbone) |

## 6. Cost (clean profile, one RTX 4090, final code)

**Once per rig:**

| | Calibrate | Table |
|---|---|---|
| Pringles | 2.4 min, 1.3 GB | 2.8 min |
| Battery | 7.2 min, 4.4 GB | 5.6 min |

**Per source demo:**

| Stage | Pringles (249 frames) | Battery (776 frames) | Peak GPU |
|---|---|---|---|
| Reconstruction (SAM2 + photometric offset) | 3.9 min | 6.6 min | 6.9–7.5 GB |
| Plan + checks | 1.1 min / 16 candidates | 2.6 min / 24 candidates | ≤ 2.3 GB |
| Render | ~16 s per episode (0.1 s per edited frame) | 15–53 s per episode | ≈ 3 GB |

**Whole dataset on one GPU:** Pringles ≈ 1.3 h, battery ≈ 3.2 h (with build and verification). The minimum GPU is about 8 GB (SAM2 step).

## 7. Strengths

1. **One method across tasks.** The same pipeline, configuration and evaluation protocol is used for a hand-over and an insertion. The per-task input is source + objects (proxy shape) + shift grid.
2. **Real pixels only, RGB only.** No generative model (no hallucination), no depth sensor, no simulator, no CAD models, no labels.
3. **Measured fidelity.** Every choice is accepted or rejected on a real-ground-truth benchmark with held-out episodes and CIs, and the pipeline beats or matches the task-specific predecessors on both tasks.
4. **Replay-safety by construction.** Every episode passes explicit checks on the commanded action stream and is directly replayable with `lerobot-replay` (kinematically).
5. **Target-position variation.** Hand-over and insertion locations are varied, not only object positions. Target coverage improved strongly.
6. **Reproducible and extensible.** CLI, YAML configs, profiling of every stage, verification, side-by-side videos, and revision-safe Hub publishing with tags.

## 8. Weaknesses / open problems (be honest about these in any paper)

1. **No policy evidence and no hardware replay yet.** This is the most important gap. The old v4B result (−9.9% far-grasp error with real_10 + 25 synthetic) used an older renderer and episode set.
2. **Visual artifacts remain** where near-camera arm parts are re-posed or arms enter from the image edge. Fragments and smearing are visible in some frames; see the comparison videos.
3. **Narrow variation.** Translations ≤ 6 cm, no object rotations, no new viewpoints, no deformables (the station's cable moves rigidly).
4. **Wrist cameras are not re-rendered.** Before the rejoin, the wrist views lag the edited arm (flagged only).
5. **Physics unchecked.** All checks are kinematic and geometric.
6. **Pringles can-position coverage dropped** (29 → 24 of 40) because of the stricter robot-proximity rule.
7. **One robot only.** Cross-robot generality is by design (config-driven) and untested.
8. **Object proxies are approximate.**
   - The battery is too small for a confident table fit, so the table height falls back to the arm rule.
   - Box proxies for unknown objects are coarse.
   - The renderer's 3-D flow for held objects relies on proxy depth.

## 9. Novelty check (summary of NOVELTY.md)

| | |
|---|---|
| **Not new** | object-centric segment transforms (MimicGen, DemoGen); coordinated bimanual transforms (DexMimicGen); URDF-rendered robot with a calibrated camera (Phantom, Masquerade); SAM2; render-and-compare calibration (CtRNet, RoboPose) |
| **Closest competitors** | R2RDreamer (arXiv 2606.17040: 3-D edit + I2V completion, needs RGB-D + SAM2); Real2Edit2Real (CVPR 2026: multi-view + video model); R2RGen (RSS 2026: point clouds, 3-D policies); RoboSplat (Gaussian splats); ROPA (diffusion robot-pose synthesis) |
| **Defensible contribution** | the combination of RGB-only, real-pixels-only editing; measured fidelity with a public protocol; replay-safety checks; one label-free pipeline across two bimanual tasks with target-position variation |
| **Risk** | reviewers may call it incremental |
| **Needed for a top venue** | a policy study (real-only vs real + synthetic, held-out object/target positions, real robot); one learned-model baseline (R2RDreamer- or Real2Edit2Real-style); ideally a second robot |

## 10. Next steps (prioritised)

1. **Policy study.** SmolVLA, then GR00T-LoRA:
   - train on real_10 alone vs the real_and_sim datasets (also with `condition`/`family` filters);
   - evaluate on the real robot at the held-out pool positions;
   - pre-register the protocol.
2. **Hardware replay.** Replay a sample of synthetic episodes (start with ±3 cm, slow).
3. **Second robot.** Run `sfagen calibrate --search` + `table` + `generate` with a new `robot:` block.
4. **Learned baseline.** Compare against one learned model on the same benchmark.
5. **Visual quality.** Consider a constrained generative refinement only in the residual hole regions, and measure it with the same benchmark.

## 11. History (what changed and why)

- **v00–v02 (Pringles, older study code).**
  - Black wrist placeholders were replaced by real wrist frames.
  - The homography renderer was replaced by mesh-guided warping (v3, +0.98 dB).
- **Pringles v4.** Added hand-over-location episodes from the frozen v3.1 planner.
- **Battery v00 / v01.**
  - v00 used a camera inherited from the Pringles rig, 20–40 px off.
  - v01 refitted the camera (+0.41 dB), added the relit plate (+0.14 dB) and station-position variants.
- **sfagen (current).**
  - **Why:** the user asked for ONE method valid across tasks (and robots) for publication.
  - **Steps:** the first unified attempt was 0.48 dB worse on Pringles. Fixes, in order:
    - SAM2 masks;
    - photometric time offset;
    - part_fill off (decided on dev episodes);
    - contact-exact IK.
  - **Result:** at least as good on both tasks.
- **Bugs found and fixed along the way** (worth knowing):
  - the lag scan forgot to restore the default;
  - a table height from the lowest arm point is wrong for tall objects;
  - TCP ≠ grasp centre (3–7 cm on the can);
  - the plate contained the can, so segmentation is now "what disappears when picked up";
  - SAM2 ran out of memory on long episodes, so it now runs in chunks;
  - an earlier Pringles card wrongly stated the clearance gate (the actual gate is ≥ 5 mm; the published minimum is 26.0 mm).

## 12. Rules the user has set (follow them)

- **Synthetic means synthetic:** never include real episodes in `real_to_sim` repos; the mixed repos are separate.
- **Honest reporting:** no fabricated labels or metrics; report artifacts and limitations.
- **Repo hygiene:**
  - do not edit frozen studies under `research/sim_free_augmentation/{v2,v3,v3_1,…}`, or `v4a`;
  - do not commit the untracked `CODEX_*.md`, `BRANCH_AUDIT_*.md` or `artifacts/` files.
- **Permission:** commit/push only when asked; no robot operations or CAN access without permission.
- **Hub revisions:** keep the previous version under a tag before overwriting; `v3.0` is the tag LeRobot loads.
