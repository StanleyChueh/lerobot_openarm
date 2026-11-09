# Agent notes (Codex / Claude)

**Start here for the simulation-free data augmentation work (OpenArm, Nov 2026):**
`research/sim_free_augmentation/sfagen/HANDOFF.md`

It describes:
- the single pipeline (`sfagen`) that generates the four published Hugging Face datasets;
- the method (`PIPELINE.md`) and the novelty check (`NOVELTY.md`);
- results, strengths, weaknesses and next steps.

**Rules:**
- **Frozen code.** Do not modify the frozen studies under `research/sim_free_augmentation/{sfa,v2,v3,v3_1,v3_2,v4,v4a,v4b1,sd10}`.
- **Untracked files.** Do not commit the untracked `CODEX_*.md`, `BRANCH_AUDIT_*.md` or `artifacts/` files.
- **Datasets.** Synthetic-only datasets must never contain real episodes; real+synthetic mixes live in separate `*_real_and_sim_*` repos.
- **Hub.** Before overwriting a Hub dataset revision, tag the current one; LeRobot loads the `v3.0` tag.
- **Robot.** No robot / CAN operations without explicit permission.
