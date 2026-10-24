# LeRobot Dataset Readiness & Coverage Analyzer

A read-only check of a LeRobot dataset before VLA training. It produces four separate statuses (PASS / WARN / FAIL / UNKNOWN), never one "ready to train" verdict:

| section | what it checks |
|---|---|
| **technical** | `codebase_version` (v2.1 / v3.x), Hub v3.0 tag vs `main`, info.json ↔ parquet schema and shapes, NaN/inf, empty episodes, frame gaps, timestamp grid, durations, camera streams and video spans, `stats.json` vs a recomputation from the data. With `--policy-checkpoint`: state/action dims vs `max_*_dim`, stale `input_features`, cameras after `rename_map`, normalizer stats vs data, `chunk_size` vs episode length. |
| **coverage** | initial object position per episode (sources A–D below), occupancy grid, smoothed density, distance to the nearest demo, near-duplicate starts, kernel effective n, coverage of a target distribution (uniform or the evaluation distribution), leave-one-out fragility, per-region counts, and recommended next positions with reasons. |
| **quality** | task adapter. OpenArm reuses `analyze_dataset_quality.analyze()` unchanged (gripper, hand-over, timing, action spikes, tracking, near-duplicate trajectories). Without an adapter this section is UNKNOWN. |
| **real_sim** | for pairs in a manifest: identical names, order and cameras, fps and robot_type, per-dimension mean/std shift, gripper hold levels, per-frame joint speed (a 20 Hz-labelled-30 fps signature), action→state lead, start pose. |

Nothing writes into a dataset directory. Hub datasets resolve to a commit SHA (the codebase-version tag, as `lerobot-train` does) and are read from LeRobot's own cache when a complete copy exists there. Checkpoints are read for their JSON and normalizer files only. Model weights are never downloaded.

## Position sources (best first)

| | source | flag | frame |
|---|---|---|---|
| A | recorded object pose | `recorded:<csv>` (episode_index,x,y[,z],frame_id) · `isaac-hdf5:<h5>` (OpenArm/IsaacLab) | metric |
| B | validated camera estimate | `transform:<json>` applied to annotations; the JSON must carry held-out evidence (`fit-transform`) | metric |
| C | human click in the start frame | `annotations:<csv>` (`annotate --interactive`) | image |
| – | colour-blob proposal (unverified) | `detect:<json>` | image, **provisional only** |
| D | nothing | (default) | → UNKNOWN |

Image (u, v px) and metric (x, y m) coordinates are separate columns. A metric map needs a transform with ≥ 6 correspondences and a leave-one-out error ≤ `max_allowed_m`, or a recorded pose. Robot joint states are never treated as object positions. The OpenArm adapter's **grasp proxy** (forward kinematics of the right TCP at the first commanded close) is reported separately, as the *grasp* position. It never replaces an object start position.

Only reachable cells count: a cell outside the declared boundary or inside a keep-out box is never "missing". When a workspace config says `"provenance": {"boundary_measured": false}`, no recommendations are produced.

## CLI

```bash
# run from the lerobot_openarm root; .venv-v2 (fork) or .venv (lerobot 0.6.2) both work
# with ROS sourced, prefix `env -u LD_LIBRARY_PATH -u PYTHONPATH` (pinocchio for the grasp proxy)
python -m tools.dataset_diagnostics analyze \
    --repo-id ethanCSL/openarm_pringles_lerobot_sim_10_v00 --role sim --output outputs/readiness/sim_10 \
    --adapter openarm \
    --policy-checkpoint ethanCSL/openarm_pringles_lerobot_sim_smolvla_10_v00 \
    --object-pose-source isaac-hdf5:$HOME/Stanley_ws/IsaacLab/logs/demos/openarm_pringles_lerobot_sim_10_v00.hdf5 \
    --workspace-config tools/dataset_diagnostics/configs/openarm_pringles_sim_spawn_metric.json \
    --grasp-workspace-config tools/dataset_diagnostics/configs/openarm_grasp_metric.json

python -m tools.dataset_diagnostics analyze --root /path/to/any/lerobot_dataset --output outputs/readiness/x

# several datasets + real/sim comparison
python -m tools.dataset_diagnostics analyze --manifest research/dataset_diagnostics/pringles_manifest.json

# human annotation of start positions (pre-filled by the detector, every point confirmed by a click/Enter)
python -m tools.dataset_diagnostics annotate --repo-id ethanCSL/openarm_pringles_lerobot_real_10_v00 \
    --camera observation.images.body_cam --output annotations/real_10.csv \
    --propose tools/dataset_diagnostics/configs/pringles_can_detect_real.json --interactive --annotator <name>

# pixel -> table-plane transform from measured correspondences (u,v,x,y), refused unless held-out error is small
python -m tools.dataset_diagnostics fit-transform --correspondences body_cam_points.csv \
    --camera observation.images.body_cam --frame-id openarm_world --max-error 0.02 --output body_cam_table.json
```

Outputs (per dataset in `<output>/<label>/`, combined in `<output>/`): `dataset_report.html`, `readiness.json`,
`provenance.json`, `episode_quality.csv`, `position_annotations.csv`, `recommended_positions.csv`,
`position_coverage_*.png`, `position_density_*.png`, `nearest_demo_distance_*.png`, `coverage_*.json`,
`start_frame_proposals.png`, `grasp_positions_*.png`, and for pairs: `real_sim_comparison_*.png`,
`state_distribution_*.csv`, `real_sim_checks.csv`, `coverage_overlay_*.png`.

## Layout

```
loader.py      read-only v2.1 / v3.x access (local, Hub, LeRobot cache), frame decoding
technical.py   schema / timing / camera / stats checks
checkpoint.py  dataset vs policy checkpoint (SmolVLA padding rules; small files only)
positions.py   sources A-D, annotation CSV, validated homography, colour-blob proposals
coverage.py    workspace config, occupancy / density / nearest-demo fields, duplicates, recommendations
compat.py      real/sim compatibility
report.py      statuses, HTML/JSON
openarm.py     OPTIONAL adapter (analyze_dataset_quality reuse, Isaac HDF5 poses, FK grasp proxy)
configs/       Pringles workspace + detector configs (each states its provenance)
tests/         python -m unittest discover -s tools/dataset_diagnostics/tests -t .
```

The core (everything except `openarm.py` and `configs/`) imports nothing OpenArm-specific.
