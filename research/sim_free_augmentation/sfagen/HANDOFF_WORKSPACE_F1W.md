# Workspace-position implementation status — completed offline validation, 2026-11-10

The continuation is documented in [WORKSPACE_F1W.md](WORKSPACE_F1W.md) and [WORKSPACE_F1W_RESULTS.md](WORKSPACE_F1W_RESULTS.md). The package now handles longer generated trajectories throughout planning, rendering, export and comparison videos. The shared source-based rule uses 3D approach proximity and balanced near/wide workspace targets. See the results for actual coverage, rejection counts and important RGB/physical-validation limits.

Public datasets were not pushed or replaced. The old published battery wrist-frame deficit is not silently repaired; new local exports use nonstreaming encoding and strict per-episode camera-span checks. The visual-walkthrough Git branch is unchanged.

## Historical paused handoff (preserved)

# Handoff: workspace-wide object-position variation (F1w): IN PROGRESS, untested

Date: 2026-11-10. The work stopped mid-implementation at the user's request.

## User's goal

- **Vary object positions across the whole reachable table area** (centre and corners), not just a few cm around the demo's own spot. Object position matters most for VLA success; the hand-over / insertion location is secondary; grasp variation comes later.
  - Pringles: the can.
  - Battery: the battery and the charging station.
- **Make the variation logic task-independent:** no hand-set shift grids per task.
- **Answer:** can 1–5 source demos produce varied object positions?

## Why the current datasets vary so little (measured on the ±9 cm dense candidates, all 10 sources)

Pass rates of the object-position family by shift size:

| object | ≤ 4.3 cm | ≤ 6.7 cm | ≤ 9.5 cm | > 9.5 cm |
|---|---|---|---|---|
| Pringles can | 36/80 | 16/120 | 2/160 | 0/120 |
| Battery | 47/80 | 44/120 | 39/160 | 22/120 |
| Station | 20/80 | 19/120 | 20/160 | 9/120 |

Rejection causes, and whether each is an artifact of the method:

1. **`place:robot_near`.** The shift is blended in from motion onset to grasp − 15, so a half-shifted arm sweeps through the new object spot. Method artifact.
2. **`speed` / `accel`.** The extra distance has to fit into the original approach time.
3. **`hands`.** Hand distance may not drop more than 5 mm below the source's, even when the hands are 30 cm apart. The rule is too strict.
4. **`place:on_table`.** Blind offsets land off the table mat.
5. **`ik`.** Real reachability limits.

## Design (implemented in part)

- **Support region per source and object** (`sfagen/workspace_plan.py::support_targets`). Candidate spots on a 1 cm grid are kept if the object is:
  - in view;
  - on the same-looking support surface (same HSV rule as `Checker.placement`);
  - at least 1 cm from other objects;
  - free of the robot at the start pose;
  - graspable there (IK of the translated grasp pose).

  Targets are then chosen by farthest-point sampling seeded at the region's centroid, so the centre comes first, then the corners, then fill (`n_targets` = 24).
- **Object-centred approach** (`f1w_schedule`/`f1w_plan`). The full shift applies from `t_approach` onward, where `t_approach` is the first frame after which the TCP stays within the object half-width + 10 cm of the original object. Everything near the object is therefore the recorded hand–object motion translated rigidly. Only the transit (motion onset → `t_approach`) and the return (grasp + 10 → lift apex) are blended.
- **Time scaling** (`time_map`, `plan_with_time_scale`). The transit and return are stretched by the smallest `s` in {1, 1.25, …, 3} that keeps commanded speed and acceleration within the existing envelope. Everything else in the recording stays identical, offset by a whole number of frames.
  - Actions keep the command lead: `a'(n) = A(τ_n) + [q'(n+k) − S(τ_n + k)]`.
  - Candidates store `tau` (output state index → source time), `ref_state` and `ref_action` (the time-scaled recording).
- **Checks** (`sfagen/checks.py`, patched):
  - time-scaled candidates are compared against `ref_state`/`ref_action`, and object tracks are taken at `tau`;
  - placement accepts the `object_workspace` family;
  - new optional `checks.hand_floor_m`: hand distance must be ≥ min(source, floor) − 5 mm, the same form as the clearance rule. Without this key, the old behaviour is unchanged.

## File state in this working copy (`~/Stanley_ws/sfa_hf_export/sfagen`, not in Git)

| file | state |
|---|---|
| `sfagen/workspace_plan.py` | NEW, complete, **never run** |
| `sfagen/checks.py` | PATCHED (original: `~/.claude/jobs/7a472154/tmp/checks.py.orig`) |
| `sfagen/generate.py` | PATCHED: family `object_workspace` in `variation_list`/`run_episode`; saves `tau`/`ref_state`/`ref_action` (float32) and `candidates/ep*/workspace.json`; `load_candidate` returns them (original: `.../tmp/generate.py.orig`) |
| `sfagen/render.py`, `render_stage.py`, `build.py`, `compare_video.py` | **NOT yet patched**. They still assume output frame i = source frame i. The patch was written and interrupted before it was applied |

The four unpatched files need:

1. **`render.py`:** add `source_frame_map(rec, c)` (output frame → nearest source frame via `tau(i + lag) − lag`) and `candidate_frames_tw` yielding `(i, j, q, g, objs)` over `len(c["state"])` frames.
2. **`render_stage.py`:**
   - `unchanged_rows` for `tau`: state == `ref_state`, `tau` an integer, objects == `rec.obj_pose(k, tau)`;
   - copy `rec.F[j]` when the frame is unchanged and exact;
   - otherwise `R.render(j, ...)`.
3. **`build.py`:**
   - loop over `len(c["state"])`;
   - wrist frames `ep.frames[w][jm[t]]`;
   - wrist pose offsets against `ref_state`;
   - provenance adds `time_scale`, `frames_added`, `source_frame_index`.
4. **`compare_video.py`:** left panel uses `source_frame_index[t]`.

## Config to add (task-independent)

```yaml
variations:
  object_workspace: {objects: [can], n_targets: 24}        # battery: [battery, station]
  interaction_location: {...}                              # optional, secondary (e.g. +-3 cm only)
checks:
  hand_floor_m: 0.10
```

Reuse the rig calibration, table height and caches through symlinks, as `configs/*_dense.yaml` / `work/*_dense` do.

## Next steps

1. Apply the remaining patches.
2. Test on Pringles source 0 and battery source 1:
   - plot each support region with its chosen targets;
   - list the rejection reasons;
   - look at rendered corner frames, where artifacts are expected for large arm motion.
3. Run sources 0–4 for both tasks and score coverage of the unseen real recordings for N = 1…5. Compare with the dense grid; the existing tool is `tools_source_budget.py`.
4. Only then rebuild or push the datasets. Before overwriting any Hub repo, tag its current revision; the budget repos `_1_`…`_9_` are currently public at `v3.0`.

## Do not forget

- **Verify per-episode spans.** The streaming encoder dropped a wrist frame in published battery episode 91 (see `lerobot_openarm/research/sim_free_augmentation/sfagen_visual_guide/diagnostics/battery_wrist_cam_frame_deficit.md`), so check `span == length` per episode before pushing anything.
- **Visual guide branch.** `docs/sfagen-visual-walkthrough` @ `c23726d` in `lerobot_openarm`; local only, not pushed.
