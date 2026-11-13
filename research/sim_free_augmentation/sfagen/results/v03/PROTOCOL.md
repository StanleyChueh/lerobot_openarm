# v03 protocol: even arena coverage with a pre-registered body-camera distortion threshold (Pringles)

Frozen 2026-11-13 by Claude before any v03 generation or rendering ([FREEZE.json](FREEZE.json)). Plan approved by the
user the same day. This is development work on the same 10 source demos as v00–v02b, not an untouched test.

## Question
Can the 10 real demos yield synthetic can starts that cover the arena evenly (corners, edges, middle)? How many
episodes survive a distortion cap at the level already tolerated in v00/v02 (strict) or at twice that level (loose)?

## Inputs (unchanged from v02)
- Source `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`, 10 episodes.
- The v02 rig calibration, table, reconstruction caches, render settings (`tm_erode 21`, `part_fill false`) and checks.
- Configs: `configs/openarm_pringles_v03{_gen,,_wristrender}.yaml`, from `artifacts/sfagen_v03_run/make_configs.py`.

## Generation
- **Arena grid:** x 9–36 cm × y −9 to 15 cm, 3 cm step, 90 targets (a superset of the v02 grid), in the same world
  frame as the source positions.
- **Planner:** F1W (`workspace_plan.plan_with_time_scale`: rigid near-object reach, vertical transit detour,
  shortest passing clock from the existing bank) via `object_targets.planner: f1w`.
- **Sources:** every source whose reconstructed can position is 0.5–12 cm from the target. 416 attempts.
- **Gates:** every existing gate unchanged (limits, speed 1.25×, accel 1.5×, clearance, hands, table, IK, untouched
  frames, placement: in view, support footprint ≥ 95 %, object gap, robot-near = 0).
- **Interaction episodes:** v02's interaction-location (60) and interaction-timing (15) candidates are copied unchanged.

## Distortion score (body camera)
`sfagen/distortion.py`, from per-frame diagnostics added to `render.py` (image output unchanged):

  s(frame) = (miss + spill) / tmask

- **tmask:** the visible target silhouette of the moved parts.
- **miss:** silhouette pixels without real pixels (warped, borrowed or atlas).
- **spill:** warped pixels more than 3 px outside the silhouette.
- **Ignored frames:** frames with tmask < 2000 px and copied frames score 0.
- **Episode score:** the 95th percentile over frames.

A provenance-only score (borrowed + inpainted fraction) was tried first and rejected before freezing. It scored a
visibly broken 12 cm F1W preview at 0.002, below ordinary v02 episodes, because missing fingers are neither warped nor
counted as filled.

## Thresholds (rule fixed here, numbers computed from v02 only)
1. Re-render all 104 exported v02 episodes with the instrumented renderer. The images are identical to v02's; only the
   diagnostics are added. Compute their scores.
2. **Validity check, before thresholds are used.** Show the 5 highest- and 5 lowest-scoring v02 episodes at their
   worst frame. If high scores do not look visibly worse, the metric is reported as failed and no threshold is applied.
   Generation still proceeds and exports are labelled "unfiltered".
3. **strict** = the maximum score among v02 episodes with |shift| ≤ 4 cm, the regime of v00, which has policy evidence
   (real-10 10/20 → mixed 15/20, p = 0.19). **loose** = 2 × strict.
4. The numbers are written to `thresholds.json` before any v03 grid candidate is rendered.

## Selection and exports
- **Per grid target:** render the 2 passing candidates with the smallest shift. If neither scores ≤ loose, render the
  next ones in order of shift, up to 4 per target.
- **Kept per threshold:**
  - every interaction episode with score ≤ threshold;
  - per target, up to 2 rendered candidates with score ≤ threshold, smallest shift first.

  `loose` ⊇ `strict`.
- **Exports,** for each threshold × wrist condition (copied wrist; rendered right wrist, i.e. the v02 wrist config):
  - synthetic-only;
  - mixed (10 real + synthetic);
  - 8 local LeRobot v3.0 datasets in all.
- **Checks on every export:** `sfagen verify`, `tools/quality_report.py`, `results/v02/audit_claude/check_exports.py`.
- No upload, training or robot use without the user's approval.

## Reporting
- **Coverage map:** every grid cell as covered under strict, covered under loose only, rejected by distortion, or
  rejected kinematically (with the first failed gate). The worst-nearest distance is given per eval position, and
  corners / edges / middle are reported separately.
- **Visual review:** the 10 worst kept episodes per threshold, plus contact sheets.
- **Negative findings** stay in the report: holes, the failure of the provenance score, and the artifact types.
- **Recording recipe** (`tools/recording_plan.py`): r = the shift at which the median v03 score crosses strict, and
  k-center can positions so that every arena cell is within r of a real demo. Discrete modes (battery slots, object
  types, grasps, orientations) need ≥ 2 real demos each.

## Known limitations fixed in advance
- The score does not capture a ghost of the object at its old place: the ghost-lid crescent is outside the target
  silhouette and not warped. It is reported by visual review only.
- Wrist images are not scored. The rendered right wrist is an ablation, and its gate failed on Pringles earlier.
- Kinematic checks are nominal; there is no physics or hardware validation.

## Amendment 1 (2026-11-13, after the first generation, before any v03 render or threshold)

The first generation (`artifacts/sfagen_v03_run/gen`, preserved, table in `amendment1/gen_v1_table.json`) passed 43/416
attempts at 28/83 targets. Two causes are planner or measurement defects, not physical limits:

1. **`on_table`, 168 rejections.** F1W's support model keeps the table pixels whose saturation/value are within 30 of
   the ring around the source can, then the connected component.
   - On this rig the black mats have a lighting gradient. The mask is a ragged patch and leaves most of the
     continuous table out ([amendment1/support_mask_src8_src0.jpg](amendment1/support_mask_src8_src0.jpg)).
   - Change: `checks.support_check: appearance` applies the v00/v02/v02b on-table rule (median footprint
     saturation/value within 30 of the source ring) to F1W too.
   - This rule is the one every released dataset used, not a new tolerance.
2. **`limits`, 95+ rejections; every attempt in rows x = 9–15 cm.** This is the same shifted-action joint-limit issue
   fixed for F1 in v02b. F1W lacked that fix.
   - Change: `plan.action_aware_bounds: true` is passed to F1W's IK (shared helper `plan.state_bounds`).
   - The limits gate itself is unchanged.

Every gate threshold is unchanged. The second generation goes to `artifacts/sfagen_v03_run/gen_a1`. Both generations
are reported.

## Amendment 2 (2026-11-13, after the amended generation, before any v03 render or threshold)

Generation `gen_a1` passed 125/416 attempts at 40/83 targets (table `amendment1/gen_a1_table.json`).
- The remaining `limits` rejections (62 alone, more combined) exceed the envelope by only 0.07–0.10° on right joint 6,
  in the action label.
- Cause: F1W retiming interpolates actions after the IK.
- Change: `plan.action_bound_margin_rad: 0.005` (0.29°, about 3× the residue). The action-aware IK bound sits that
  far inside the envelope, never tighter than the recorded pose (`plan.state_bounds`, unit-tested).
- The limits gate is unchanged. Generation goes to `gen_a2`, and all three generations are reported.

## Amendment 3 (2026-11-13, after gen_a2, before any v03 render or threshold)

`gen_a2`: 137/416 attempts pass at 40/83 targets ([amendment3/gen_a2_grid_on_image.jpg](amendment3/gen_a2_grid_on_image.jpg)).
The rear-left eval corner, where real eval cans were placed at about (15, −2.5) cm, is rejected by `on_table` for
every source.

- **Cause:** the HSV rule compares saturation. On this near-black mat (ring V ≈ 41/255), saturation is numerically
  unstable. At the rejected cells the saturation difference is 40–78 while the brightness difference is 0–24.
- **Effect:** the rule effectively forbids moving a can across the seam between the two mats, which differ slightly in
  tint ([amendment3/lab_vs_hsv.jpg](amendment3/lab_vs_hsv.jpg)).
- **Change:** `checks.plate_appearance_space: lab`. The same tolerance, as a fraction of range (30/255 → 11.8 on a
  0–100 scale), is applied to CIELAB lightness and chroma.
- **Discrimination:** it still rejects the window-lit front-left table region (ΔL 12–19).
- This is informed by the data. Every grid target admitted only under amendment 3 is tagged in the report, so an
  HSV-rule-only dataset can be rebuilt from `gen_a2` if preferred.

Generation goes to `gen_a3`.
