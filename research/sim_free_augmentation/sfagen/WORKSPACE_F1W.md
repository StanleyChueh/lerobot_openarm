# Workspace-wide object-position augmentation

This path prioritizes initial object position. It is shared by all tasks; only the existing object names, geometry and grasping-arm configuration differ. Interaction-location augmentation is disabled in the workspace example configurations.

## What changes

1. Infer an estimated connected support region from the source clean plate. Require at least 95% support under the object's projected footprint, positive camera depth, a 10-pixel image margin, separation from other objects, no initial robot overlap, and a reachable recorded grasp orientation. Sample a 1 cm grid; choose 48 target positions: 24 covering the centroid/extremes, then up to 24 covering the nearest distance quartile (remaining slots fill globally). Both tasks use this same rule; no task-specific XY offset list is supplied. These are camera/geometry estimates, not surveyed physical table positions.
2. Locate the final travel bout after a sustained rest. Measure approach distance in 3D against the object bounds, so an overhead hand does not count as being at the object. Translate the full near-object approach and grasp rigidly with the object. Blend only the preceding travel and return. A generic vertical transit detour is added if the straight blend crosses the relocated object's approach corridor. Continue rejecting interference with the other arm, joint-limit failures and poor IK.
3. Select the shortest passing retiming from the configured scale bank. First try the protected transit/return clock, a constant-slowed pickup window with smooth speed ramps, and uniform retiming as a last option. The latter also changes interaction pace and is recorded explicitly. Repeated frames are expected when slowing recordings.
4. Resample gripper commands by zero-order hold. Preserve the recorded command lead and residual relative to future state for every arm, then add the generated arm's future joint displacement. Independently reconstruct labels for validation; candidate actions are never their own reference.
5. Render at the continuous output exposure time and select source RGB through the inverse source clock. Compute held-object poses from FK at the exact fractional camera exposure; use SO(3) interpolation for other moving poses. Copy source frames only when the scene and source time are unchanged. Candidate hashes and rendering fingerprints prevent stale exports. The pre-grasp guard uses source grasp time, including the entire rigid approach, rather than ending at its onset.
6. Export the full generated duration. Use nonstreaming episode video encoding and decode/check every episode/camera span, including PTS and episode boundaries. Existing exports are never silently replaced.

## Usage

Run with the project's existing Python environment and this directory on `PYTHONPATH`:

```bash
python -m sfagen generate configs/openarm_pringles_handover_workspace.yaml --episodes 0,1,2,3,4
python -m sfagen render configs/openarm_pringles_handover_workspace.yaml --episodes 0 --keys CAN_KEY
python -m sfagen build configs/openarm_pringles_handover_workspace.yaml --repo-id local/pringles_workspace
python -m sfagen verify configs/openarm_pringles_handover_workspace.yaml
python -m sfagen video configs/openarm_pringles_handover_workspace.yaml --speed 2
```

For a strict single-source run, set `source.episodes: [0]` in a copied config and use a fresh `work_dir`; the command computes its replay envelope from that subset only. Rig calibration is separate evidence and may have used additional recordings. If using an existing rig, copy its calibration/table files and its reconstruction/SAM caches into the fresh work directory; do not refit calibration accidentally against another dataset.

`generation_manifest.json` records source state/action hashes, the pinned dataset revision, actual replay envelope, camera calibration, reconstruction hash, code hashes and candidate hashes. An incomplete generation manifest prevents rendering/building partially replaced candidates.

## Important training limits

- This is kinematic augmentation of recorded RGB, not physics simulation or proof of collision-free/contact-successful robot execution. Replay checks sample collision geometry and inherit source/reconstruction imperfections.
- Larger translations can reveal occluded surfaces and produce warping, copying, stretching and inpainting artifacts. Inspect generated frames; geometric acceptance is not an image-quality certificate.
- The current renderer transforms the **body camera**. Wrist RGB is time-remapped source footage, not a new viewpoint rendering. Changed scenes/time samples are marked invalid in `wrist_view_valid`, and camera-pose offsets are exported. Unconfigured wrist timing offsets remain unvalidated. Train on body RGB alone, or explicitly mask invalid wrist inputs. Do not treat an unmodified three-camera loader as valid synthetic multiview training.
- Geometric coverage uses FK-at-gripper-command proxies; neither object-position accuracy nor VLA success is established by those coverage numbers.
- Source-budget reports filter each subset using only that subset's speed, acceleration and table-clearance envelope. If candidates were initially retimed with a larger configured source set, small-subset yield is conservative; a separate strict one-source generation run gives the direct test.
- Public datasets are unchanged. Before any future replacement, preserve each current Hub revision with a tag and retain the previous dataset card/provenance. No upload is part of local validation.

## Validation scope

See `WORKSPACE_F1W_RESULTS.md` for actual final measurements, source hashes, 1–5-source tables and reproduction commands. The first horizontal-only prototype and its negative battery coverage result are retained separately in the Codex artifact directory. Development coverage measurements influenced engineering iteration; they are exploratory and are not an untouched evaluation or a controlled VLA experiment. Whole-table coverage is not guaranteed: support masks, visibility, recorded grasp orientation and the planner restrict the estimated admissible region.

`fast_reject: true` avoids expensive checks only after a necessary IK/limit/speed/action test has already failed; it never issues a passing verdict. Bounding-box pruning computes the same exact minimum mesh distance as the original all-pairs method, including penetration cases; it changes runtime rather than collision thresholds. Collision checks still inherit the existing mesh-pair exclusions and discrete sampling. Do not treat them as full-scene physical certification.

Window/uniform slowing also changes observed gripper response duration. Command transitions stay discrete and ordered, but actuator response after retiming is not physically validated. The generated state/action pairing is nominal supervision, not a measured robot replay.

Legacy variation definitions remain available, but nominal action clipping is corrected: range failures are rejected rather than paired with an inconsistent planned state. Existing clipped archives must be regenerated before export; historical published data/results are preserved. See the results report for the independently measured legacy mismatches.
