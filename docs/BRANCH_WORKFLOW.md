# Shared baseline and task branches

`official-lerobot-quest` is the integration branch. `main` is a compatibility mirror: advance both to the same commit after baseline changes. Do not develop separate features on `main`.

The shared baseline includes the evaluation fixes originally delivered in `8e43f49` and `21388b0`:
- background observation sending;
- reuse of the recorder's observation;
- timing-option metadata;
- a default observation-similarity threshold of zero, to avoid action-queue starvation during small movements.

Target control rate is `--fps=30`; measure the achieved rate from evaluation logs on each device. These changes do not guarantee a particular hardware rate.

| Branch | Additional purpose | Writing device |
|---|---|---|
| `force-aware/gate2a-bench-prep` | Telemetry integration and hardware bench preparation | Assign to robot-connected device |
| `research/factr2-next-openarm` | NEXT exporter/model and force estimation research | Confirm owner before handoff |
| `research/sfagen-workspace-f1w` | Active simulation-free generator development (`sfagen`) | `stanley-Z790-AORUS-ELITE-AX` |
| `research/sfagen-unified-pipeline` | Code and walkthrough of the published sfagen datasets; change only by fast-forward from reviewed work | `stanley-Z790-AORUS-ELITE-AX` |
| `research/contact-aware-augmentation`, `research/sd10-g1-generation` | Frozen sim-free study history (Gate 0/1/1R, v1–v4B.1, SD10); README updates only | `stanley-Z790-AORUS-ELITE-AX` |

**Syncing branches.**
- Every active task branch must contain the current integration baseline.
- Preserve research additions by merging the baseline into existing task branches; avoid rewriting shared history.
- New work starts from `origin/official-lerobot-quest`, or from the research branch it extends.
- Historical `archive/*` refs and the frozen study branches are exempt from baseline synchronization.

**READMEs.**
- Only `official-lerobot-quest` and `main` carry the robot execution README.
- Every other branch replaces the root `README.md` with its own purpose, results (with exact commits and dataset revisions) and reproduction commands.
- When the baseline is merged into a task branch, `README.md` will conflict. Keep the task branch's version, and update the branch map on the default branch instead.

**Before working:**
1. Inspect `git status --short --branch`.
2. Fetch with `git fetch --prune origin`.
3. Inspect `git branch -vv`.

Use one writing device per task branch. On a clean tracking branch, use `git pull --ff-only`; investigate divergence rather than force pushing. Use separate worktrees for concurrent agents, and verify that editable Python installs point at the intended checkout.

**Promoting fixes.**
- Promote general evaluation/runtime fixes to the integration branch and validate them.
- Then advance `main` to the same commit and merge that baseline into active task branches.
- Keep force-specific changes and experiment outputs on their task branches.
- Push before handing work to another device, and record the pushed SHA and next step.
- Untracked files are not backed up by a branch push.

**Regression checks.**
- Frequency: `plugins/tests/test_eval_obs_reuse.py` uses a fake robot and sender. It checks observation reuse, read/action thread ownership, and the slow-action fallback.
- Matched evaluation: `research/generated_data_diagnosis/matched_evaluation/tests/`.

No hardware session is part of these checks.
