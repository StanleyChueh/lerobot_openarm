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

## Amendment 4 (2026-11-14, after v03 rev 1 was published; before any rev-2 generation): same duration and speed as the source

**Finding (user inspection of the published rev-1 data):**
- Real demos last 241–280 frames (8.0–9.3 s at 30 fps). v00 synthetic episodes are exactly 1.000× their source
  (96/96, checked on the Hub).
- Rev-1 grid episodes are 1.09× their source at the median, but 20/84 (loose) are >1.5× and the worst is 3.56×
  (998 frames, 33 s). Among all 189 passing rev-1 candidates, 98 are >1.3×.
- Cause: `plan_with_time_scale` passes the speed/accel gate by slowing the clock (`window` / `uniform`, up to 6×).
  The robot then reaches and lifts in slow motion. The user rates this as more harmful to VLA training than mild
  visual distortion.

**Changes** (stricter; no rev-1 gate is loosened; rev-1 data are kept as a negative result, tag `v03-rev1`):
1. **Identity clock:** `object_targets.time_scales: [1.0]`. No time scaling.
2. **Wide spatial blend** (`object_targets.wide_blend: true`, `workspace_plan.f1w_schedule(wide=True)`). The shift
   ramps in from the end of the previous gripper event +12 frames, or frame 0 if there is none, and blends back out
   at `next_event − 10`. The robot absorbs the shift over the longest event-free span instead of moving faster.
3. **New hard gate `duration`** (`checks.fixed_duration`): the synthetic frame count equals the source frame count.
4. **New hard gate `speed_similarity`** (`checks.speed_similarity: [0.8, 1.25]`): for every moved arm, the TCP
   (Cartesian) peak speed and the mean speed over the edited frames are 0.8–1.25× of the same arm in its source episode.
   The per-joint speed ≤ 1.25× / accel ≤ 1.5× envelope gates are unchanged.
5. **Applies to every exported episode.** The hand-over variants (reused from v02) are re-checked from the exported
   state by `tools/quality_report.py`. Any that fail are excluded before export, and the exclusion is reported.
6. **Minimal-distortion selection:** per grid target, all passing candidates are rendered. The up to 2 with the
   lowest distortion score are kept per threshold, with ties broken by the smaller shift. Rev 1 used the 2 smallest
   shifts instead.
7. **Unchanged:** thresholds (strict 2.289 / loose 4.578, `thresholds.json`), grid, sources, every other gate,
   the coverage metric and the reporting.

Rev 2 goes to `artifacts/sfagen_v03r2_run/`, with results in `results/v03r2/`. Configs:
`configs/openarm_pringles_v03r2*.yaml`. The published repos keep their names and are overwritten after tagging the
rev-1 revision `v03-rev1` (user decision).

## Amendment 5 (2026-11-14, after the rev-2 body renders and before any rev-2 export): remove the ghost lid

**Finding:** a visual audit of all 67 loose rev-2 grid renders (`results/v03r2/audit_sheets/`) shows a ghost
crescent, lid or duplicate can at the source can's position during grasp and lift in most episodes. It is present in
nearly all episodes with score ≥ 1. The distortion score does not measure it, because the ghost lies outside the
target silhouette.

**Cause:** the source can's plate-difference region (mostly the pale lid) extends beyond the proxy-pose mask, so it is
not replaced by the plate.

**Existing fix:** `render.object_mask_grow: 60` (`sfagen/render.py`, from the can-fix study
`results/canfix/RESULTS.md`). It grows the object mask over the connected plate-difference region. It was not adopted
for v01 only because the registered whole-can PSNR gain was below +0.5 dB; that rule is insensitive to a small ghost.

**Dev check:** 6 rev-2 episodes, 5 with ghosts plus 1 clean one, compared frame by frame in
`results/v03r2/amendment5/grow60_dev.jpg`.
- The ghost is removed in 4/5. Source 7 / target 038 keeps a small sliver.
- The clean episode is unchanged.
- The distortion score rises by 0.07–0.40, because the larger removal region is counted.

**Change:** rev 2 uses `object_mask_grow: 60` for every body render, grid and hand-over alike. All body renders are
redone, and selection is re-run with the unchanged thresholds. These are now slightly stricter, since scores rise.

**Unchanged:** wrist renders (their fingerprint does not include body render settings), thresholds, and all gates.
The base-renderer rev-2 renders are kept in `artifacts/sfagen_v03r2_run/body_base/` and reported.
