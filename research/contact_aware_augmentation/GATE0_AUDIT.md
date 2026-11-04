# Gate 0 audit: Contact-Aware Generative Augmentation for Few-Shot VLA Learning

**Date:** 2026-11-04. **Branch:** `research/contact-aware-augmentation`, from `21388b0` (= `force-aware/gate2a-bench-prep` HEAD).
The delivery commit SHA is in `git log` and the delivery note.
**Scope:** read-only, CPU only. No GPU training, no robot, no CAN socket, no motor command, no Hub write. Datasets
were never written. Battery v02 metadata, data and one body_cam video were downloaded to a session scratch folder.

**Labels:**
* **VERIFIED:** computed in this audit from the files (`outputs/`, `test_output/`).
* **REUSED:** a prior repo finding, cited, not re-run.
* **SUPPORTED:** consistent evidence, not proven.
* **UNKNOWN:** no evidence.

Companion files:
* [DATA_PROVENANCE.json](DATA_PROVENANCE.json)
* [CONTACT_OBSERVABILITY.md](CONTACT_OBSERVABILITY.md)
* [GATE1_EXPERIMENT_PLAN.md](GATE1_EXPERIMENT_PLAN.md)
* Diagnostics: `scripts/gate0_diagnostics.py` (+ `test_gate0_diagnostics.py`, `hub_metadata.py`, `make_data_provenance.py`)

## 0. Bottom line

1. **Neither dataset contains force, torque, current, velocity or telemetry.** Both contain only 16 joint
   *positions* (state and action) and 3 RGB streams. (VERIFIED)
   * Nothing force-conditioned can be trained or evaluated on them.
   * They support geometry and phase work only.
2. **The telemetry gate has been validated offline only.** No real-hardware telemetry session, `gate2_report.json` or
   bench run exists on this machine, and the Gate-2 doc states the bench test was not run.
   * The offline suites pass at this commit (VERIFIED, §4).
   * Every hardware claim (freshness, loopback completeness, kernel timestamps, no added TX, timing impact) is UNKNOWN.
3. **Action-consistency can be measured coarsely but not yet validated.** FK projected through the fitted body_cam
   extrinsic tracks the human-clicked can positions with r = 0.97 (u) / 0.98 (v) over 50 episodes. It has a constant
   offset of (−24, +87) px and a residual of 18.6 px median after removing it (VERIFIED). The extrinsic was never
   validated on held-out robot keypoints; the wrist-camera extrinsics are hand-set and blocked (REUSED).
4. **Phases are recoverable from the gripper commands in both datasets** and are consistent across episodes
   (VERIFIED). Battery's interpretation was checked on video for 2 episodes (SUPPORTED).
5. **Recommendation:**
   * **GO** for a CPU-only Gate-1 proof of concept that tests geometric and grasp consistency on 3 Pringles
     episodes (GATE1_EXPERIMENT_PLAN.md).
   * **NO-GO** for force-conditioned generation on historical data.
   * **BLOCKED** for force-conditioned generation on new data, until the Gate-2 telemetry bench report is GO and
     calibration data D1–D5 exist (CONTACT_OBSERVABILITY.md §5).

## 1. Prior work audited and reused (not re-run)

| source | what it established | used here as |
|---|---|---|
| `research/dataset_diagnostics/RESEARCH_READINESS_REPORT.md` (2026-10-24) | real_10 = real_v00 pool eps 0–9; schema/timing/stats PASS; action leads state by 2 frames; FK grasp proxy matches sim can position (median 5 mm); can pushed ~2 cm before grasp; real gripper stalls at openness R 0.43 / L 0.35; pool eps 28–47 have late-session release habit | provenance relation (re-verified bit-exact), phase definitions, quality flags |
| `research/dataset_diagnostics/pringles_v00/real_10/episode_quality.csv` | per-episode phases (grasp, left grasp, release, overlap) and 16 quality flags: ep 2 `timing_outlier`, ep 5 `action_spike`, the rest clean | existing quality annotation; PoC episode choice |
| `research/dataset_diagnostics/annotations/real_v00.csv` | **human-verified** can start pixel (body_cam frame 0) for all 50 pool episodes | reference for the FK projection check |
| `research/generated_data_diagnosis/*` (2026-10-27) | 500 MimicGen demos ≈ 5.6 templates; hand-over replayed untransformed; tie 47 vs 49 / 100 hides a position × policy interaction; both policies fail at the same contact step (left securing the can at hand-over); action labels are correct | evidence that geometry-only generation did not cover the contact phase (prior-work comparison) |
| `research/task_aware_real2sim/REAL_DATA_AUDIT.md`, `m2a/*` (2026-10-19) | body_cam extrinsic FITTED by silhouette (IoU 0.33→0.71); wrist extrinsics HAND and **not identifiable** from the data (M2A BLOCK); ChArUco protocol written, not run; can, pad and fingertip geometry not measured | camera model used for projection; calibration requirements |
| `docs/force_aware/HARDWARE_AUDIT.md`, `CONTACT_FEASIBILITY.md`, `GATE2_HARDWARE_READINESS.md`, `TELEMETRY_SCHEMA.md`, `BATTERY_CONTACT_COLLECTION.md` | `tau` is a firmware current-based effort estimate, not external torque; the sidecar is receive-only; Gate-2 bench procedure and thresholds G1–G13/S1–S5; D1–D6 calibration list; battery GT1–GT3 protocol | telemetry status (§4), observability (CONTACT_OBSERVABILITY), calibration list |

None of these experiments was repeated. New in this audit:
* battery v02 (not audited before);
* re-verification of Hub revisions;
* video/packet counts;
* the FK→body_cam projection check;
* FK bimanual geometry;
* the event-aligned kinematic residuals.

## 2. Task 1: revisions, counts, schema, cameras, action semantics, timing (VERIFIED)

| | Pringles `real_10_v00` | Battery `insertbattery_real_v02` |
|---|---|---|
| Hub revision (main = tag `v3.0`) | `cebdee98115ab6827a4ec2655335c536b5dd085b` (2026-10-21) | `ac63facb63571215f07cb5465386b127c4b740ff` (2026-11-02) |
| local copy | HF-lerobot cache snapshot; all 9 files equal to the Hub (size + LFS sha256) | session scratch download (meta, data, body_cam file-000); sizes equal to the Hub |
| task string | "Pick up the Pringles can with the right arm, hand it to the left arm" | "Pick up the charging station with the left arm, and insert battery in it with the right arm" |
| episodes / frames | 10 / 2 575 | 55 / 41 360 |
| episode length | 241–280 frames (8.0–9.3 s nominal), median 254 | 667–890 (22.2–29.7 s), median 754 |
| codebase / robot_type | v3.0 / `openarm_umeow` | v3.0 / `openarm_umeow` |
| state, action | float32 [16], names `RJ1,LJ1,…,RJ8,LJ8` (interleaved), all `.pos`; action names = state names | identical |
| cameras | body_cam, wrist_cam (left), right_wrist_cam; 480×640 RGB, AV1 (crf 30, GOP 2), 30 fps; **no depth** | identical |
| timestamps | `timestamp` = `frame_index/30` (max err 4e-7 s); frame and global indices contiguous; metadata lengths = rows | same (max err 9e-7 s) |
| video vs data | per camera: video span = length/fps exactly; **video packets = 2 575 = frames** on all 3 cameras | spans exact; packets not counted (videos not downloaded except body_cam file-000) |
| action semantics | action = commanded joint position request (lerobot records the request before `clamp_step`; HARDWARE_AUDIT §4); **leads state by 2 frames** (mean err 0.0086 rad at k=2 vs 0.0135 at k=0) | leads by 2 frames (0.0168 vs 0.0200) |
| grippers (raw motor rad) | R open −1.298 → closed 0, L open 1.212 → closed 0; 99.3 % / 99.8 % of frames at an endpoint | R 96.0 % at an endpoint (1 251 distinct values: partial closes exist), L 99.3 % |
| true loop rate | **UNKNOWN**: timestamps are nominal; no DIAG or telemetry log of the recording sessions on this machine | UNKNOWN |

Other Hub versions of the battery task: v00 `24eb9016`, v01 `0b3509dd`. The local v01 cache (`b20b699e`) is
older than Hub v01 and was not used. Pool relation: real_10 episode *i* is bit-identical to real_v00 pool
episode *i* for i = 0–9 (state arrays).

## 3. Task 2: measured torque, telemetry or positions only? (VERIFIED)

Data columns of both datasets: `action, observation.state, timestamp, frame_index, episode_index, index,
task_index`. Every vector name ends in `.pos`. A keyword scan over columns and names (torque, tau, effort, current,
force, wrench, vel, temp, telemetry, contact) finds nothing.

* **Positions only.**
* No telemetry sidecar was found on this machine for either recording.
* Pringles v00 predates the telemetry code (2026-10-28), so it cannot have one.
* Battery v02 (pushed 2026-11-02) could in principle have been recorded with `--robot.telemetry_dir` on another
  machine. **UNKNOWN**, so it must be asked.
* The prior reports were produced under `/home/csl/`; this host is `stanley-Z790`.

What positions alone show is covered in CONTACT_OBSERVABILITY.md: jaw-blocking, load-induced tracking offsets and
candidate event residuals. **None of it is force.**

## 4. Task 3: the telemetry gate (offline tests vs hardware validation)

Run here at `21388b0`, with a `sitecustomize` guard that raises on any `PF_CAN` socket. No CAN interface was up on
this host (`ip -brief link` shows none). Logs are in `test_output/`.

| suite | result | what it proves | what it does **not** prove |
|---|---|---|---|
| `plugins/tests/test_telemetry.py` (checks 1–20) | all PASS | decode bit-equal to installed openarm_can 1.2.8; no transmit path; arms see only getter calls; commands and safety hold identical with telemetry on/off; R1–R4 fixes (fd ≥ 1024, ramp overflow, untimed frames, frames log); analyzer logic on fake sessions | anything about the real gs_usb adapter, kernel timestamps, loopback, freshness, real timing |
| `plugins/tests/test_telemetry_fixture.py` (F1–F7) | all PASS | committed fixture matches the producer; **F7: fixture is mock data** | — |
| `plugins/tests/test_telemetry_record.py` (A–C) | all PASS (G6 UNKNOWN by design) | official `lerobot-record` main() with telemetry joins the dataset bit-exactly, on a **fake** bus | real-bus alignment |
| `tools/dataset_diagnostics/tests` | 39 tests OK | reused loader/FK helpers | — |
| `scripts/test_gate0_diagnostics.py` | 11 tests OK | projection math vs pinocchio, lag, event detection, synthetic end-to-end | — |
| not re-run here | `test_robot_guard`, `test_return_lag`, `test_quest_safety`, `test_teleoperate_mock`, `test_record_mock`, `test_record_gate`, `test_rerun_status` | Gate-2 doc: `test_quest_safety` P fails 43/44, cause UNKNOWN | — |

**Hardware validation status:**
* No bench session was run.
* No `gate2_report.json` exists.
* The only `openarm_telemetry/1` sessions on disk (`/tmp/openarm-gate1-evidence/supplied-*`, 2026-10-28) come from the
  fake-bus tests.

So HARDWARE_AUDIT items 5–9 (firmware PMAX/VMAX/TMAX, `tau` scale, value freshness S1, loopback completeness, kernel
timestamps) and Gate-2 checks G4–G13, B1 and S1–S5 are all **UNKNOWN**. The Gate-2 decision stands: contact
collection is NO-GO until a bench report says GO.

## 5. Task 4: handover phases and quality annotations

Phases come from the binary gripper commands (openness < 0.5), with hold levels from the measured state
(`outputs/events_*.csv`).

**Pringles real_10** (VERIFIED, n = 10):

| phase | frames (min / median / max) |
|---|---|
| right close (grasp) | 82 / 86.5 / 101 |
| left close (take-over) | 156 / 163.5 / 188 |
| right open (release) | 164 / 168.5 / 198 |
| command overlap (right open − left close) | 4 / 5 / 10 |

* One close per arm in every episode.
* The left arm holds to the end.
* Jaw stall while holding: R openness 0.40–0.43, L 0.34–0.38. It is blocked by the can in 10/10 for both arms.
* The existing quality table agrees on the state-based events (grasp 83–102, left grasp 158–190). Only ep 2
  (`timing_outlier`) and ep 5 (`action_spike`) are flagged.

**Battery v02** (VERIFIED for the event structure; the interpretation was checked visually on episodes 0 and 7,
`outputs/figures/battery_phases_body_cam_ep{0,7}.jpg`, so it is SUPPORTED for the rest):

| event | frames (min / median / max) | interpretation |
|---|---|---|
| left close | 85 / 117 / 182 | left grasps the charger; one close in 55/55; jaws blocked (openness 0.35–0.46) |
| right close 1 | 236 / 286 / 361 | right grasps the AA cell |
| right open 1 | 381 / 440 / 638 | releases the cell over the bay |
| right close 2 | 407 / 463.5 / 563; 12–39 frames (median 27.5) after open 1 | re-closes with the cell no longer held, then presses on it (visual, 2 eps) |
| right open 2 | 428 / 656 / 808 (53 eps) | final release |
| left open | 560 / 654 / 808 (54 eps; within ±13 frames of right open 2 except one) | charger set down |

* Right closes per episode: 2 in 52/55, 3 in 2, 1 in 1 (ep 22).
* **The right jaws close fully on the AA cell:** stall openness −0.05, blocked in 0/55. Jaw position therefore
  cannot even confirm that the cell is held.
* Battery has no existing quality annotation. The repo's handover quality tool does not apply to this task
  structure.

**Bimanual geometry from FK** (VERIFIED, `outputs/fk_bimanual_geometry.csv`):
* **Pringles at take-over:** TCP distance 0.114–0.151 m (median 0.133). Right TCP in the left TCP frame: std
  (1.3, 1.3, 0.9) cm.
* **Battery at right open 1** (54 eps; ep 22 excluded): TCP distance 0.046–0.077 m. Right-in-left std
  (0.8, 1.2, 0.6) cm, relative rotation 168–180°.

The bimanual relative pose at the contact phase is tight. That makes it a usable consistency reference.

## 6. Unknowns and blockers

| # | item | blocks | cheapest resolution |
|---|---|---|---|
| U1 | body_cam extrinsic accuracy on robot keypoints (fitted on silhouettes only; residual vs can clicks 18.6 px after a constant bias) | any pixel-level action-consistency claim | annotate ~6 robot keypoints on 18 frames (Gate 1), or the ChArUco capture (M2A §7, ~30 min, operator) |
| U2 | wrist-camera extrinsics (HAND; M2A BLOCK) | any wrist-view consistency metric | ChArUco hand-eye capture (operator) |
| U3 | camera frame age vs joint state (unmeasured; HARDWARE_AUDIT §5) | contact-onset timing metrics below ~2 frames | LED/flash sync test at the next recording |
| U4 | true recording loop rate (timestamps nominal) | velocity/acceleration-based features | telemetry or DIAG log at the next recording |
| U5 | gains, `gripper_squeeze_tau` and code SHA at recording time (not stored in the datasets) | interpreting tracking residuals as effort | record them in every new dataset card; ask the operator for v02 |
| U6 | whether a telemetry sidecar exists for battery v02 on the recording PC | force-aware use of v02 | ask the operator |
| U7 | physical dimensions and masses (can, charger, AA cell, foam pads, pad height) | metric object models, payload subtraction | tape, calipers and scale, ~15 min |
| B1 | no force/torque in any dataset | force-conditioned generation on existing data | **cannot be resolved retroactively** |
| B2 | telemetry never validated on hardware | force-conditioned generation on new data | Gate-2 bench (approval) → D1–D5 |
| B3 | battery wrist/right-wrist videos not inspected | battery visual PoC | download (read-only) when needed; Pringles is the Gate-1 target |

## 7. Files and reproduction

```bash
cd ~/Stanley_ws/lerobot_openarm
R="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python"
P=~/.cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/cebdee98115ab6827a4ec2655335c536b5dd085b
POOL=~/.cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d
B=<folder with insertbattery_real_v02 @ ac63facb: meta/, data/ (+ videos/... body_cam file-000 for the phase sheets)>
env -u PYTHONPATH .venv-v2/bin/python -I research/contact_aware_augmentation/scripts/hub_metadata.py research/contact_aware_augmentation/outputs/hub_metadata.json  # online, GET only
$R research/contact_aware_augmentation/scripts/gate0_diagnostics.py --pringles-root $P --pool-root $POOL --battery-root $B \
   --hub-meta research/contact_aware_augmentation/outputs/hub_metadata.json --out research/contact_aware_augmentation/outputs
python3 research/contact_aware_augmentation/scripts/make_data_provenance.py
$R -m unittest research/contact_aware_augmentation/scripts/test_gate0_diagnostics.py -v
```

Runtime ≈ 5 s CPU. Outputs:
* `summary.json`
* `provenance_local.json`
* `events_*.csv`
* `tracking_residual_windows.csv`
* `event_aligned_residual.json`
* `fk_bimanual_geometry.csv`
* `fk_projection_vs_annotation.csv`
* `figures/`
