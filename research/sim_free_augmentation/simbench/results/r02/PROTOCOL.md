# R02 protocol — pipeline v1 "joint-space transit blend" (pre-registered 2026-11-15, before any generation)

Approved by the user (plan `~/.claude/plans/ancient-moseying-clarke.md`). One §7 method change. Hashes in `FREEZE.json`.

## Change
`plan.blend_space: joint` (default `cartesian` = pipeline v0, unchanged): exact IK only on full-offset frames
(w >= 0.999), each contiguous block warm-started from the source joints; every blending frame gets
q_src + w * dq(nearest full-offset frame). Applies to object_workspace (F1W), object_position (F1) and
interaction_location (F2). `ik_blend` then applies only to full-offset frames (blend frames have no Cartesian target);
recorded per candidate as `ik_blend_scope`. All other gates, thresholds, the grid, the selection rule and the 1.000x
length rule are unchanged. No retiming.

## Hypotheses and decision rules
- **H1 (RoboTwin):** place_empty_cup object_workspace yield (all gates, groups L+R pooled) >= 10 % (v0: 0/339).
  Fail -> stop and report (S1 stop rule).
- **H2 (OpenArm regression, same code, loose v03r2 config, generate only):**
  (a) the `cartesian` run reproduces the v0 grid result 92/416 exactly (sanity; else stop);
  (b) `joint` grid yield >= `cartesian` grid yield;
  (c) `joint` interaction_location + interaction_timing passes >= `cartesian` passes.
  H2 fails -> report; the user decides whether pipeline v1 is adopted for OpenArm. RoboTwin S1 may continue only
  if H1 passes (pipeline v1 is applied to every robot, so an OpenArm regression is reported, not hidden).
- Reported, not gated: gripper (TCP) peak/mean speed ratios, J5 speed ratio (RoboTwin), first failed gate per
  candidate, mesh-clearance and object rejections, changed OpenArm candidates.

## Fixed settings
`rel`-free (no threshold); w >= 0.999 = full offset (existing convention); ik() tolerance unchanged (0.1 mm / 1e-3 rad).
Negative results are kept. No sensitivity variants.

## Result of the registered variant (kept as a negative result)
Cup run `artifacts/simbench_runs/s1b` (joint blend as registered): object_workspace **8/339 (2 %) -> H1 FAILED**
(first failed gate: speed 174, accel 139, object 12, mesh 6; `yield_cup_joint.json`). The start-pose wrist swing is
gone (no failure lies in the early ramp); failures sit at the blend edge t_approach.

## Amendment 1 (2026-11-16, written before any run of it; outcome-driven, disclosed)
Diagnosis (read-only on the s1b candidates): (i) the blend had a velocity kink at the exact-segment edge (smoothstep
arrives with zero slope while the exact offset is already moving); (ii) the exact pose at t_approach (object still
10 cm away, arm mid-reach) genuinely needs a wrist-yaw offset (converged IK gives the same value) that vanishes as the
arm reaches its grasp pose, moving J5 faster than the demos ever do. Changes, both robot- and task-agnostic:
1. the joint-space blend is a cubic Hermite from (0, slope 0) at the last unedited frame to (dq, dq') at the
   exact-segment edge (mirrored for the ramp-out): velocity-continuous edges;
2. for object families the exact segment starts at max(first full-offset frame, t_grasp - 15), i.e. sfagen's existing
   F1 convention (f1_plan uses t_grasp - 15); F2 keeps its interaction window. The `ik` gate (1 mm / 0.5 deg) applies
   on the exact segment; `ik_blend_scope` records this per candidate.
Offline estimate on the s1b candidates (speed+accel+exact IK): 315/339. Not a gate change; thresholds unchanged.

Hypotheses for Amendment 1 (fresh run dirs; decision rules as above):
- **H1-A1:** RoboTwin place_empty_cup object_workspace yield (all gates) >= 10 %.
- **H2-A1 (user requirement: OpenArm must not get worse):** on the OpenArm loose v03r2 config, same code,
  `joint` >= `cartesian` for (a) grid passes, (b) grid targets with >= 1 passing candidate, (c) interaction_location
  passes, (d) interaction_timing passes; and `cartesian` grid = 92/416 (sanity). Any family worse -> report, do not adopt
  `joint` for that family without the user's decision.
- **H3-A1 (cross-robot/task):** handover_block and stack_blocks_two each reach >= 10 % yield with the same recipe; the
  per-robot adaptation is limited to the URDF-derived robot block and per-task `objects` + arena (counted in R02).

## Amendment 1 results (kept)
- Guard (default `cartesian`): 28/28 candidates, 4/4 bodies bit-identical (`guard_blendspace_a1`).
- **H1-A1 PASS:** RoboTwin cup object_workspace 71/339 (21 %), 41/54 targets (`s1b2`).
- **H2-A1 FAIL (narrowly):** OpenArm loose, same code: cartesian 92/416 (sanity OK), 39 targets; joint 111/416 but
  38 targets (8 lost, 7 gained); interaction_location 62 vs 60; timing 11 vs 11. Per candidate: 49 pass both, 43 only
  cartesian (joint first gate: object 39 [robot points near the can +6..+20 vs source, the approach below 10 cm is no
  longer the shifted Cartesian path], limits 3, other 1), 62 only joint (cartesian first gate: speed 32, accel 30).

## Amendment 2 (2026-11-16, before any run of it; outcome-driven, disclosed) — pipeline v1 = joint-space FALLBACK
The two blends fail for opposite reasons, so v1 keeps v0 and adds the joint blend only as a fallback:
`plan.blend_space: joint_fallback` -> plan every variation with the v0 Cartesian blend; only if that candidate fails a
gate, plan the joint blend (Amendment 1) and keep it if it passes; otherwise keep the v0 record. Gates, thresholds,
grid, selection unchanged. Consequences fixed before running: every v0-passing candidate is bit-identical to v0, so
OpenArm cannot get worse by construction; robots/tasks where v0 fails are rescued where the joint blend passes.
Predicted from the two A1 runs (exact union): OpenArm grid 154/416, 46 targets; RoboTwin cup 71/339.
- **H2-A2 (must hold exactly):** OpenArm fallback run: every candidate passing in the cartesian run has an identical
  candidate_sha256 in the fallback run; grid passes >= 92 and targets >= 39; interaction_location >= 60; timing >= 11.
- **H1-A2:** RoboTwin cup >= 10 %. **H3-A2:** handover_block and stack_blocks_two >= 10 % each with the same recipe.
