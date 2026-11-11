# Workspace-wide object-position augmentation: implementation and validation

Date: 2026-11-10. Position/trajectory generation, continuous-time rendering and full-length export are implemented and tested locally. This is an engineering completion report, not evidence of real-robot task success or a training-ready three-camera release. Public datasets and historical research files were not overwritten or pushed.

## Source of truth

- Shared repository: `/home/stanley/Stanley_ws/lerobot_openarm`, branch `docs/sfagen-visual-walkthrough`, HEAD `c23726d4a96489616ec3206a281560296a66e957`. No tracked implementation changes were made in that repository.
- The implemented package is `/home/stanley/Stanley_ws/sfa_hf_export/sfagen`, which has no Git repository. Release code is identified by per-file SHA-256, not by the unrelated documentation commit. Original Claude files were backed up before editing.
- Pringles source: `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98115ab6827a4ec2655335c536b5dd085b`; battery source: `ethanCSL/openarm_insertbattery_lerobot_real_10_v02@a1c29d77dabcaf5a0a9ff658ba742a2d941e1570`. Sources remain episodes 0–4.
- Source file hashes, download etags and exact state/action lineage were checked. Pringles subset episodes map to pool episodes 0–9; battery subset episodes map to 15,20,21,22,23,30,32,36,45,53. All ten are excluded from the 40-/45-recording coverage pools. Battery pool upstream revision was not retained locally; its exact parquet files are frozen by SHA-256 rather than inventing a Hub revision.
- Existing rig calibration, table estimates, reconstruction and SAM caches were reused. A single motion-source result therefore does not mean that the entire calibration/reconstruction setup was learned from one recording. No new training, dynamics simulation, CAN access or robot operation occurred.

## Implemented corrections

1. Support targets come from the estimated source support surface, using a 1 cm grid, positive camera depth, projected visibility margin, >=95% footprint support, object separation, robot-start overlap and translated-grasp IK. Whole physical-table coverage is not claimed.
2. Both tasks use the same 48-target rule: 24 centroid/extreme/farthest-point targets plus up to 24 from the nearest distance quartile, with global fill if that quartile is too small. Object names/geometry/arms remain task configuration; no task-specific XY target grid was added. All 15 source/object banks were unique, in their inferred region, and retained the identical 24-wide-target prefix of the preceding 3D pilot.
3. The full near-object reach and grasp translate with the object. Proximity is measured in 3D against object bounds, and travel starts after the final sustained rest. In battery source 1, the old XY rule incorrectly started the rigid approach at frame 18 even though the hand was overhead; the repaired schedule starts travel at 166 and the rigid approach at 218, before grasp command 265.
4. A generic vertical transit detour avoids a straight blended reach through the relocated object. The shortest passing clock is selected from protected transit/return, smoothly slowed pickup-window and whole-recording alternatives. Later handover/insertion positions rejoin the source; timing can change and is recorded. Interaction-location augmentation is disabled in these configs.
5. Actions preserve the source command/future-state residual, with an independently computed reference. Gripper commands use zero-order hold. Actions are never used as their own reference and are not clipped to disguise a state/action disagreement.
6. The pre-grasp proximity guard ends at source grasp minus 15 source frames, mapped through tau; it includes the rigid approach. The earlier shortened guard was corrected and its extra rejections retained.
7. The renderer maps source RGB through continuous exposure time. Held objects use FK of interpolated joints at the exact exposure, avoiding attachment errors from separately interpolating object translations. Source frames are copied only when source time AND exposed geometry agree.
8. Export loops use generated length, remap wrist footage, retain pose offsets/validity flags, record source clocks/hashes and refuse stale rendering or existing export replacement. Nonstreaming encoding is followed by per-episode/per-camera PTS and frame-span checks.
9. Exact bounding-box pruning accelerates mesh minimum-distance checks without changing thresholds: zero difference against all-pairs on 180 real/generated poses, including five colliding poses, plus an independent primitive-geometry unit test. This does not extend the inherited collision-pair scope.

## Final geometric candidates

These are nominal trajectory bundles passing the implemented replay checks; only six representative bundles were fully rendered/exported.

| Object | Accepted / attempts, sources 0–4 | Max shift from source | Estimated centre span X × Y |
|---|---:|---:|---:|
| can | 171 / 240 | 15.80 cm | 28.0 × 20.0 cm |
| station | 163 / 240 | 22.56 cm | 30.0 × 24.0 cm |
| battery | 107 / 240 | 25.83 cm | 33.0 × 24.0 cm |

Per-source acceptance counts:
- can: 0 → 36/48, 1 → 39/48, 2 → 15/48, 3 → 42/48, 4 → 39/48.
- station: 0 → 34/48, 1 → 41/48, 2 → 45/48, 3 → 43/48, 4 → 0/0.
- battery: 0 → 16/48, 1 → 39/48, 2 → 13/48, 3 → 15/48, 4 → 24/48.

## Strict single-motion-source test

All final source-0 native spatial paths were retimed and fully checked against source 0 alone, without using the five-source speed/acceleration envelope. Shared rig calibration is disclosed above. No targets were selected according to evaluation outcomes.

| Object | Accepted source-0 variants | Other-real coverage <=2 cm: old raw / old label-checked / new |
|---|---:|---:|
| can | 36 / 48 | 4 / 4 / 22 |
| station | 34 / 48 | 15 / 15 / 21 |
| battery | 15 / 48 | 12 / 9 / 19 |

Thus one motion demonstration can produce many distinct estimated initial positions under these checks. This does not establish VLA learning benefit, reliable contact, or complete tabletop coverage.

## 1–5-source coverage screen

Coverage uses FK at gripper command closes, a position proxy rather than independently measured can/station poses. The recordings are reused exploratory development data. Each row uses the first N sources. The main bank was retimed with five sources, then each subset was rechecked with only its own speed/acceleration/table envelope. These small-N rows are conservative reuse estimates, not direct replanning experiments; the strict one-source experiment above is separate. More sources also change sample count and replay envelopes, so this is not a causal source-diversity study.

| Sources | Can /40: old → new | Station /45: old → new | Battery /45: old → new |
|---:|---:|---:|---:|
| 1 | 4 → 11 | 15 → 21 | 12 → 7 |
| 2 | 10 → 18 | 23 → 33 | 24 → 25 |
| 3 | 13 → 24 | 24 → 42 | 32 → 29 |
| 4 | 23 → 32 | 26 → 43 | 38 → 34 |
| 5 | 23 → 32 | 39 → 44 | 38 → 34 |

At five sources, label-checked historical baselines cover can 22/40, station 39/45 and battery 35/45, versus new 32/40, 44/45 and 34/45. **Battery nearby-position coverage is not improved at five sources**, even after removing inconsistent old labels. Larger maximum displacement is not evidence of universal dataset superiority. Different target locations and the optional TCP hand-floor policy also prevent interpreting these numbers as an isolated algorithm effect. Both final and dense banks have 48 attempts per object/source.

## Preserved negative findings and integrity corrections

- The intermediate XY-approach/24-target version covered battery 31/45 versus the old raw 38/45. It is retained under the original Codex work directories and `horizontal_v1_*` reports. The 3D segmentation repair and balanced target bank were engineering responses, not a preregistered confirmatory experiment.
- Some old passing archives clipped nominal actions while retaining unclipped planned states. Across historically passing object-position candidates from the first five sources, the independent audit found:
  - pringles: 2 / 24 mismatches, maximum 0.00593460 rad.
  - battery: 15 / 112 mismatches, maximum 0.01784039 rad.
- Original archives/results were preserved. Future legacy generation now retains the exact label and rejects range failures; export rejects inconsistent old labels. Raw historical coverage is still shown rather than replaced.
- A part-fill/narrow-erosion diagnostic did not resolve missing-surface artifacts. Conservative renderer settings were retained; the diagnostic is not presented as a successful visual repair.
- The published battery missing-wrist-frame issue was not silently patched on the Hub. New local encoding/span checks prevent accepting that defect; public repair still requires a separately versioned rebuild/tag.

## Independent numerical and export checks

| Task | Accepted archives rescored, all rows | Max nominal action error | Max contact-window FK position / rotation |
|---|---:|---:|---:|
| pringles | 171 | 1.19e-07 rad | 0.1453 mm / 0.0572 deg |
| battery | 270 | 2.37e-07 rad | 0.1237 mm / 0.0573 deg |

FK and attachment were recomputed from saved state arrays, not accepted from stored summary errors. These tests validate construction consistency, not physical contact. Fresh planning reproduced saved states/actions/tau exactly for one wide can and one wide battery example; repeated grasp-frame rendering was bit-identical in those checks. Fifteen unit tests cover timing, command transitions, corrupted action channels, duplicate IDs, missing/overlapping video spans, exact distance pruning and fractional-exposure attachment.

| Local sample dataset | Episodes | Frames per camera | Camera spans checked | LeRobot actual image/state reads |
|---|---:|---:|---:|---:|
| pringles | 3 | 1466 | 9 | 9 |
| battery | 3 | 3607 | 9 | 9 |

All source/action hashes, per-camera decode counts and PTS spans passed; no all-black frame was found. These exports contain 16-D state/action labels and three camera features. Codec PSNR versus the renderer input is 39.36 dB for Pringles and 41.09 dB for battery; it measures encoding preservation, not fidelity to real counterfactual views.

## RGB and physical limitations

**These are engineering previews, not a verified three-camera VLA training release.** Actual decoded MP4 frames show relocated objects, but also stretched/fragmented robot surfaces, dark replacement regions and residual ghosts. Some target objects are occluded at the start. Proxy visibility in the image bounds is not a guarantee of informative RGB. The existing low-quality battery object fit (source 1 silhouette IoU about 0.093) and approximate camera calibration limit physical pose claims.

The body camera is transformed. Wrist frames are time-remapped copies of the source, not new wrist views; invalid edited/time samples are flagged, and unconfigured wrist lags are explicitly assumed rather than validated. A default three-camera loader that ignores those flags is unsuitable. Body-only observation selection avoids that particular mismatch but does not solve body-image artifacts.

Discrete mesh checks inherit excluded links and lack full robot/object/contact physics. Attachments and grasp phases derive from source geometry and commands, not measured force. Window/uniform retiming preserves gripper command order but does not validate changed actuator-response duration. No claim is made about executable hardware replay, external torque, force conditioning, VLA improvement or real task success.

## Practical next step

Keep initial-position variation as the primary augmentation axis. Before bulk RGB release or VLA training, add an independently inspected visibility/appearance gate and source-only rejection-aware target filling: within a fixed candidate budget, select additional region points according to distance from accepted points, rather than letting rejected extremes leave coverage holes. Preserve valid near-source examples and evaluate the remaining battery gap. Do not tune coordinates to these reused evaluation recordings. Three-camera use additionally requires rendering or valid evidence for both changed wrist views. Grasp/interaction variation remains a separate later axis.

## Reproduction and evidence

From the installed package, using the existing environment:

```bash
cd /home/stanley/Stanley_ws/sfa_hf_export/sfagen
PY=/home/stanley/Stanley_ws/lerobot_openarm/.venv/bin/python
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$PY" -B -m unittest discover -s tests -v
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$PY" -B -m sfagen verify configs/openarm_pringles_handover_workspace.yaml
env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 "$PY" -B -m sfagen verify configs/openarm_battery_insertion_workspace.yaml
```

Use a fresh work directory and copy the matching rig/calibration/reconstruction caches before regenerating; existing exports are deliberately not overwritten. For a one-source rerun use `*_workspace_strict1.yaml`, not merely `--episodes 0` with a five-source config. The exact tested orchestration, source-budget commands and input hashes are retained in:

`/home/stanley/Stanley_ws/lerobot_openarm/artifacts/codex_workspace_f1w_20261110`

- `source_lineage.json`, `source_file_hashes.json`, `source_reconstruction_quality.json`
- `target_bank_checks.json`, `legacy_label_audit.json`, `independent_checks.json`, `distance_regression.json`
- `release_source_sha256.json`, `starting_sha256.json`, `starting_files/`, `logs/`
- `work/*_workspace_final/source_budget.json` contains all 31 source subsets, not only the convenient nested subset.

Local previews and copied validated results:

- [Pringles source-vs-generated MP4](configs/work/openarm_pringles_handover_workspace/compare.mp4)
- [Battery/station source-vs-generated MP4](configs/work/openarm_battery_insertion_workspace/compare.mp4)
- [Pringles decoded frames](configs/work/openarm_pringles_handover_workspace/decoded_mp4_examples.png)
- [Battery/station decoded frames](configs/work/openarm_battery_insertion_workspace/decoded_mp4_examples.png)
- [Pringles estimated position coverage](configs/work/openarm_pringles_handover_workspace/workspace_coverage.png)
- [Battery/station estimated position coverage](configs/work/openarm_battery_insertion_workspace/workspace_coverage.png)

No policy training, robot operation, physics-engine generation or external upload was performed.
