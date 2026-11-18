# R05 protocol: robust arm / object segmentation (pre-registered 2026-11-16, before any R05 run)
User direction: fix the segmentation the pipeline already has (one SAM2 mask per arm, one mask per object) so that
occlusion by the object and the background do not damage the arm, especially the gripper; the fix must be general
across tasks, embodiments and benchmarks. No new mask or track. Contact-aware ideas are out of scope.
Approved plan: `~/.claude/plans/read-home-stanley-claude-plans-don-t-mod-lucky-sky.md`. Method change #3.

## Baseline (measured before this protocol, read-only, cached masks, 2 episodes per task)
Reference = URDF silhouette through the true RoboTwin head camera. Gripper coverage by raw SAM2, far / near / holding:
cup 0.82 / 0.37 / 0.37; stack 0.83 / 0.30 / 0.28; hand-over 0.00 / 0.00 / 0.00 (fitted-vs-true silhouette IoU 0.25).

## Changes (all default off; robot- and task-agnostic; no link, robot or task names)
1. `reconstruction.prompt_object_cut: depth` (`segment.py:arm_prompt`). The arm prompt loses only the pixels where an
   object proxy is in front of the arm in the joint z-buffer (5 px rim at 640 px width), instead of a 25-px
   neighbourhood of every object.
2. `reconstruction.mask_repair: 0.6` (`recon.py:_repair`). Per frame, on the same arm masks:
   (a) drop pixels farther than 25 px (at 640 px) from the arm's projection; (b) drop pixels where an object proxy is
   in front; (c) for every prompted link whose visible silhouette is covered < 0.6, add that silhouette eroded 2 px.
   Object masks are cut after the repair.
3. `calibration.fit_focal: false` (`calibrate.py:run`). The focal length is not optimised when the dataset supplies
   intrinsics.
Constants 0.6, 25 px, 5 px, 2 px are fixed here and are not tuned per dataset. Pixel constants scale with width / 640.

Deviation from the approved plan, decided before any run: no shadow rule in the object mask. A rule "darker with the
same chroma is shadow" would also delete dark or grey objects, and a contact shadow should move with its object.

## Conditions
- A: baseline (existing caches). B: changes 1 + 2 with the existing camera, same reconstruction cache (isolates the
  masks). C (RoboTwin only): change 3, then 1 + 2 on a new reconstruction.
- Development: OpenArm Pringles demos, RoboTwin `place_empty_cup`, `stack_blocks_two`, `handover_block`.
- Held-out, run once after the code is frozen: RoboTwin `handover_mic`, `stack_bowls_two`, `move_can_pot`, 2 episodes
  each, no objects configured, condition C settings.

## Hypotheses and accept rule
- H1 (masks, RoboTwin, truth reference): final arm-mask coverage of the gripper links >= 0.8 (median) when near or
  holding, in B for cup and stack and in C for all three; mask pixels > 10 px outside the true arm not higher than A.
- H2 (camera): `fit_focal: false` lowers the median reprojection error against truth on all three rigs.
- H3 (images): on the same candidates, B/C renders show fewer arm / gripper fragments and ghosts than A
  (side-by-side sheet; single rater, not blinded).
- H4 (no regression): OpenArm held-out pose-transfer PSNR (`sfagen bench`, episodes 0,2,3,4,6,7,9) with B not lower
  than A (95 % episode-bootstrap CI of the difference includes or exceeds 0); unit tests pass; the OpenArm guard is
  bit-identical with the flags off.
- H5 (generality): unit tests on three synthetic URDFs pass; held-out tasks meet H1 with zero code or setting change.
A change that fails its hypothesis is reported and left off by default. Negative results are kept.
Truth (true camera) is used for scoring only and never enters an sfagen run. No dataset is rebuilt or published.
