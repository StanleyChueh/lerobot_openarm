# v02b frozen protocol: rear-row rerun with action-aware IK bounds (2026-11-13, Claude)

Scope: local generation, render and export only. No training, hardware/CAN, physics, publication, commits or pushes.
Base: research/sfagen-v02 worktree (77e52d4 + v02 local changes). v02 results, datasets and gates stay untouched.
Requested by the user ("audit Steps 1–2; rerun anything you want"). Hypothesis formed **after** seeing v02 rejections:
this is a development rerun, not an untouched test.

## Why

v02 accepted 18/35 grid targets and added no coverage at evaluation position ② rear-right
(worst nearest can 6.63 cm, same as real-only; v00 3.91 cm). The rear row (x = 12/15 cm) failed mainly on
`limits`: the exact shifted *action* label on `openarm_right_joint6` exceeds URDF ∪ source by ≤ 0.97°.
`plan._retarget` bounds the IK *state* only; the label A(t) + dq(t+k) inherits the source's action–state gap.
v00 (13d7231a) has 0/65 episodes outside the same envelope (audit_claude/v00_limits.json), so v00's rear coverage
did not rely on out-of-limit labels.

## Single change

`plan.action_aware_bounds: true` (F1 object-position planner only). For each moved arm and frame t', the IK upper bound
becomes min(previous bound, q(t') + max(URDF hi, max source action) − max{A(t) : min(t+k, T−1) = t'}), and symmetrically
for the lower bound. It is never tighter than the recorded pose, so untouched frames stay bit-identical. This is a
constraint inside the solve; labels are not clipped afterwards. If the pose becomes unreachable, the existing IK/blend
error gates reject the candidate. Flag off = v02 behaviour, unchanged code path.

## Attempts and budget

Only the 17 grid targets with no accepted v02 attempt: indices 0–11, 14, 15, 20, 21, 34. The nearest-1–3-source
assignment is unchanged from `results/v02/planned_targets.json`: **35 attempts**, IDs `can_target_NNNb`.
One pass. Preserve every rejection. No interaction-location or timing variants (v02's are reused).
Budget: 6 GPU-hours total.

## Gates (identical to v02, not loosened)

Every v01/v02 gate and threshold is unchanged: limits, speed ≤ 1.25×, accel ≤ 1.5×, IK ≤ 1 mm/0.5°,
blend ≤ 5 mm/2°, clearances, placement (`extra == 0`), start/untouched/rejoin, codec PSNR ≥ 30 dB.
The same fixed-duration and independent label checks apply.

## Stop rule

If no target with x ≤ 15 cm passes, stop after generation and report; do not render.
Otherwise:
- render the new passes only (body and right wrist);
- build v02b = v02's 104 accepted + new passes, in new work dirs, for both wrist conditions plus mixtures;
- run `sfagen verify`, `tools/quality_report.py`, the independent export audit and the per-position coverage table vs real/v00/v02.

## Reporting

Report all 35 outcomes and their rejection causes. Report the per-position worst nearest distance for real, +v00,
+v02 and +v02b, and how many v02b episodes the flag changed. The coverage measure is a development-informed geometric
proxy, not policy evidence.
