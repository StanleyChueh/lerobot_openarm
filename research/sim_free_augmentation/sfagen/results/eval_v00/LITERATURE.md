# Focused literature refresh — 2026-11-12

Read primary abstracts before implementing diagnostic tools. These motivate
measurements; none establish the cause of these OpenArm failures.

- [DemoGen (2502.16932)](https://arxiv.org/abs/2502.16932): limited spatial
  generalisation motivates object-position edits. Measure actual evaluation
  placements against the exact training revision instead of attributing a
  failure to wrist views from appearance alone.
- [WARP-VLA (2610.11508)](https://arxiv.org/abs/2610.11508), October 2026:
  wrist-camera mounting changes affect fine geometric cues. Measure camera
  pose and camera-relative can motion separately; a translated camera and can
  can cancel in the relative view.
- [Cross-View Action Consistency (2608.06965)](https://arxiv.org/abs/2608.06965),
  August 2026: masks the wrist stream to isolate scene-view robustness.
  A matched wrist ablation is necessary for causal attribution; saved rollout
  diagnostics alone cannot prove wrist copying caused late closure.

No new architecture or training intervention is designed in Step 1. Earlier
renderer literature and negative findings remain in ../LITERATURE.md and
../canfix/RESULTS.md. Abstract-level review only; no numerical claims from these
papers are used as experimental evidence here.
