# Shared baseline and task branches

`official-lerobot-quest` is the integration branch. `main` is a compatibility mirror: advance both to the same commit after baseline changes. Do not develop separate features on `main`.

The shared baseline includes the evaluation fixes originally delivered in `8e43f49` and `21388b0`: background observation sending, reuse of the recorder's observation, timing-option metadata, and a default observation-similarity threshold of zero to avoid action-queue starvation during small movements. Target control rate is `--fps=30`; measure achieved rate from evaluation logs on each device. These changes do not guarantee a particular hardware rate.

| Branch | Additional purpose | Writing device |
|---|---|---|
| `force-aware/gate2a-bench-prep` | Telemetry integration and hardware bench preparation | Assign to robot-connected device |
| `research/factr2-next-openarm` | NEXT exporter/model and force estimation research | Confirm owner before handoff |
| `research/contact-aware-augmentation` | Contact-verifier history, generation feasibility, and current simulation-free dataset generation | `stanley-Z790-AORUS-ELITE-AX` |

Every active task branch must contain the current integration baseline. Preserve research additions by merging the baseline into existing task branches; avoid rewriting shared history. New work starts from `origin/official-lerobot-quest`. Historical `archive/*` refs are frozen snapshots and are exempt from baseline synchronization.

Before working, inspect `git status --short --branch`, fetch with `git fetch --prune origin`, and inspect `git branch -vv`. Use one writing device per task branch. On a clean tracking branch, use `git pull --ff-only`; investigate divergence rather than force pushing. Use separate worktrees for concurrent agents and verify editable Python installs point at the intended checkout.

Promote general evaluation/runtime fixes to the integration branch, validate them, then advance `main` to the same commit and merge that baseline into active task branches. Keep force-specific changes and experiment outputs on their task branches. Push before handing work to another device and record the pushed SHA and next step. Untracked files are not backed up by a branch push.

For frequency regression checks, `plugins/tests/test_eval_obs_reuse.py` uses a fake robot and sender; it checks observation reuse, read/action thread ownership, and the slow-action fallback. Matched-evaluation regressions live in `research/generated_data_diagnosis/matched_evaluation/tests/`. No hardware session is part of these checks.
