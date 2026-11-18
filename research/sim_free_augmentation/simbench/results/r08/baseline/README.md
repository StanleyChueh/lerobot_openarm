# R08 baseline: OpenArm Pringles v04 (copied 2026-11-18 from artifacts/simbench_runs/r04_openarm)
- `candidates/ep*/checks.json`, `generation_manifest.json`: replay checks of all 239 candidates (v04 code ab657f7).
- `body/*.json`: per-frame renderer diagnostics (tmask/miss/spill/borrowed/inpainted px) of every candidate.
- `selection.json`, `verification.json`: the N=100 selection and dataset verification of the published v04.
- `body_clearance_v04_mixed.txt`: right-arm vs `openarm_body_link0` URDF mesh distance (coal, every 3rd state frame)
  for the 110 episodes of the local v04 mixed dataset (100 synthetic + 10 real, real = episodes 100-109).
  Columns: episode, synthetic, source demo, |shift| cm, shift xy cm, min distance mm, frames < 5 mm, frames < 20 mm, link.
  34/100 synthetic and real demos 0 and 1 reach 0 mm. The real arm never touched the body (user), so the right-arm model is wrong.
  The checker skipped every arm-body pair (`skip_collision_links: [link0, link1]` matches `openarm_body_link0`).
