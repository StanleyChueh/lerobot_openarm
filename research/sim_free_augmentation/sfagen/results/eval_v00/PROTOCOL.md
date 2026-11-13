# Step 1 diagnosis protocol — 2026-11-12

Frozen before calculating the new image/kinematic results. Exploratory diagnosis;
no policy-performance acceptance gate, training, robot operations or publication.
Code base: `efda1cbe4b2e896aacb92bac34b788a91433b8d5`.

- Evaluation inputs: real `de1bbf51fdac10218dd5602260ad65dd9da2aca5`, mixed
  `af877eb5294e1e747a0eee113e824bf888bd625e`, 20 episodes each. The training
  mixture is `ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00`
  at `13d7231a48866069001f4a0635259dd7d80b1e17`, not today's v00 tag.
- Position groups: episodes 0–3 front-right, 4–7 rear-right, 8–10 rear-left,
  11–15 front-left, 16–19 middle. Outcomes are user-reported aggregates;
  Hub evaluation success labels are null. Do not infer per-trial success from
  gripper events alone or treat the trials as paired successes.
- Detect the initial red can by colour components in the body frame, inspect
  every trial's overlays, and back-project its lid centre at table + known can
  height. Explicitly flag missing/ambiguous detections and calibration limits.
  Report centimetres and sensitivity to ±5 pixels / ±1 cm lid height.
- Grasp proxy: first right action-gripper downward crossing of 0.5 openness
  after opening; also record measured-state crossing and action/state delay.
  TCP error is planar distance to initial can centre, with TCP-to-can offset
  estimated from the ten real source grasps. Report raw and offset-adjusted
  errors; this is a geometric proxy, not force/contact measurement.
- Approach proxy: first TCP entry within 5 cm planar distance and ±5 cm of
  source median grasp height before close. Can-motion proxy: persistent image
  displacement >5 px before close; loss/occlusion is unknown, not movement.
  Save per-trial event times, trace diagnostics and contact sheets containing
  initial, approach, pre-close, close, post-close and hand-over views.
- Coverage: ten real sources, exact v00 synthetic positions, v01 targets, all
  forty measured evaluation starts. Report nearest-neighbour distances and
  source/synthetic convex-hull inclusion. Do not identify FK grasp positions
  with object centres without documenting the offset.
- Wrist audit: exact v00 labels/provenance, source-time alignment, phases
  approach (0.3–0.6 grasp time), final approach (0.6 grasp to grasp−5),
  grasp (grasp±5), hand-over (left close±5). Quantify right camera translation,
  rotation and projected can-relative displacement using calibrated wrist FK.
  Joint invalidity flags do not establish camera-relative object error.
- Renderer diagnosis: same source/frame, ±6 cm target and target joints,
  legacy / sfagen / sfagen with legacy lid-track pose. Use the R1 cross-episode
  aligned pairs and scoring masks for a bounded comparison; preserve the
  original R1 gate. Cluster uncertainty by source/target episodes, not frames.
  If only a subset can run, label it diagnostic and do not claim a full R1 win.
- Statistics: exact binomial 95% intervals and two-sided Fisher test for
  user-reported aggregate outcomes; bootstrap episodes for geometric summaries
  (2,000 draws, seed 0). Existing exposure confound: batch 8 versus 16.
- Save input/code hashes, commands, failures, rejected/missing observations,
  PNG contact sheets and machine-readable outputs beside DIAGNOSIS.md.

Stop after Step 1. Decisions for v02 remain conditional on evidence; no new
dataset generation, training, Hub cards or outward-facing writes in this step.
