# Reusing the OpenArm real-to-sim pipeline across tasks

The unit of reuse is the robot rig and its recording protocol: 16 motor-radian action/state values,
30 Hz, the three named RGB cameras, Quest teleoperation, and the sim bridge. A *task cfg* supplies
movable objects, collision shapes, resets, and success logic. A *real-look profile* supplies the
dataset-fitted static scene, robot appearance, lighting, camera response and optional contact fit.
The same renderer reads either profile; it does not decide what a task means.

| Layer | Reuse across tasks | Fit or implement per task |
| --- | --- | --- |
| Robot and recording | CAN/Quest action convention, calibration, 30 Hz stepping, image keys | Check any changed mount or robot hardware |
| Visual scene | Backdrop mesh renderer, material recolouring, sensor response code | Static room images/depth, object textures, camera alignment |
| Physics | Contact application API, actuator model | Object collision geometry, mass, friction, gripper contact, insertion tolerances |
| Task | LeRobot record loop and server protocol | Spawn/reset distribution, grasp/insert stages, success condition |

## What the research suggests

* [SimFoundry](https://research.nvidia.com/labs/gear/simfoundry/) reconstructs scene backgrounds and
  objects as separate components, then produces related scene variants. This supports a reusable
  reconstruction pipeline with task-specific assets, rather than a rendering function written for one
  object. Its reported real-to-sim evaluation also motivates checking policy outcomes, not images alone.
* [TwinAligner](https://arxiv.org/abs/2512.19390) aligns visual appearance and interaction dynamics
  separately. A convincing image is insufficient evidence that a grasp or insertion will work.
* [Real2Render2Real](https://proceedings.mlr.press/v305/yu25a.html) shows that reconstructed visual
  assets can generate useful VLA data without dynamics simulation. That route is useful for visual
  pretraining or augmentation; a task requiring physical contact still needs collision geometry.
* [RialTo](https://www.roboticsproceedings.org/rss20/p015.pdf) uses real observations to construct a
  digital twin and tests whether training there improves real robustness. This supports a loop of
  reconstruct, measure, train, and validate on held-out real episodes.
* [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) uses structured variations of clutter, light,
  background, table height and language. Randomization should be scoped to a measured task envelope;
  changing light or table height independently of a baked photo backdrop can create impossible images.

These papers motivate the separation above; they do not show that the present OpenArm profiles have
the same transfer performance.

## Adding another dataset

1. Compare the real dataset's metadata and task text with the intended sim recording. Run
   `compare_sim_real_schema.py` on completed datasets. That checks the schema, not physical units.
2. Create a task cfg with the real task's objects, reset distribution and success condition. Keep
   robot and camera names from the shared OpenArm rig where possible.
3. Create `source/isaaclab_assets/data/real2sim/<task_profile>/profile.json` with the dataset repo ID,
   compatible task IDs, randomization profiles, camera names, required asset files and static scene entities hidden by the backdrop. Add fitted
   `backdrop_*.npz`, `albedo.json`, `lighting.json`, `tone.json` and `sensor.json` as available. A
   contact fit is optional and must name its manipulated scene asset in `contact_asset`.
   For a future multi-object fit, `contact_physics.json` can instead contain
   `{"targets": [{"asset": "charger", "params": {"object_friction": 0.5}},
   {"asset": "battery", "params": {"object_mass": 0.027}}]}`. These numbers are only an example;
   fit the actual materials and object masses before using them.
4. Set `real_look_dir` in the task cfg. The server loads and validates that profile and reports its
   source dataset in the connection handshake. The recorder can require it with
   `--robot.expected_real_dataset=<repo_id>`.
5. Compare sim renders at **the same recorded robot/object states** against held-out real frames for
   each camera. Then replay held-out real actions in the sim and inspect contact and task success.
   Only then vary appearance and physics within plausible bounds and measure real policy rollouts.

## Current profiles and physical limits

* **Pringles handover:** `pringles_v00` has a fitted visual scene and a contact fit for `can`. The
  pick-up task's `-v1` cfg selects it. Mimic source collection can remain deterministic, while
  generated visual randomization is a separate experiment.
* **Battery insertion:** `battery_v01` has a dataset-fitted visual scene and charger top texture.
  The leftmost slot has an approximate open collider and a lift-plus-seat success condition. A
  drop test verifies the cavity is physically open. Image-aligned replays of eight real episodes
  lifted the charger in seven and the battery in one under the best bounded contact fit, but did
  not complete insertion. A complete image-aligned replay of real episode 0 lifted neither object
  and returned `any_sim_success: false` (see `artifacts/replay_battery_ep0_imagefit/summary.json`).
  A separate 160-candidate search on that episode, keeping object corrections within 1.5 cm and
  8 degrees, found a charger lift but no battery lift (see
  `artifacts/battery_fit_logs/fit_ep0_imagebounded.json`). The eight-episode search is in
  `artifacts/battery_fit_logs/fit_imagebounded.json`.
  The slot geometry, gripper pads and real-to-sim alignment require
  measurement and contact validation before an insertion dataset can be called successful.
* **Pringles handover:** A local LeRobot replay dataset contains three episodes (778 frames) from
  real source episodes 0, 4 and 5 that passed the simulator's handover success condition. Its
  `validation.json` identifies the source and final object pose for each episode. The action/state
  schema and video streams match the real dataset, but gripper state statistics still differ.

Keep these datasets and policy evaluations separate until each task's physical interaction is
validated. The shared robot schema makes cross-task training possible, but it does not make their
success conditions or visual distributions interchangeable.
