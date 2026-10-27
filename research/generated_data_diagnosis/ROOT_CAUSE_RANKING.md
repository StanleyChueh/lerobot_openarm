# Why 500 generated demos did not beat 10 manual ones — root-cause ranking

Ranked by measured evidence × severity, then ease of testing. Details and numbers in the four audits.

## Bottom line

47/100 vs 49/100 is not a difference (Fisher p = 0.89; repeat runs vary by 3–5 points). It is also not "no effect":
the two policies fail in **different places**, and the tie is two coverage effects cancelling. The generated data
helped where it has starts (x 0.20–0.27: 30/47 vs 16/46, p = 0.007) and hurt where it has none (x < 0.20: 6/37 vs
25/43, p = 0.0002; policy × x interaction p = 0.006). Neither dataset improved the step where both policies fail —
the left hand securing the can during the pass (51/51 and 51/53 failures).

## Ranking

| # | cause | evidence | status | severity | ease of testing |
|---|---|---|---|---|---|
| 1 | **Generation's spawn distribution ≠ the evaluation's.** Generation samples uniform x ∈ [0.203, 0.422]; the evaluation samples N(0.235, 0.07) ∩ [0.12, 0.40], so 34 % of evaluation spawns have no generated start, and 28 % of generated starts are in regions the evaluation never spawns. Nearest-object source selection then almost never used the 4 near-base sources (13/500 episodes, source 2 never). | position-structured success with a significant interaction; spawn code read; Monte Carlo of the exact sampler | **CONFIRMED** (effect measured); causal mechanism SUPPORTED HYPOTHESIS until tested on fixed poses | high: explains why the extra data cancels out | easy: fixed-pose evaluation of existing checkpoints; region-filtered subset training |
| 2 | **Generation adds no information about the failing phase.** In hand-over mode only the right arm's approach is re-anchored; the left arm and the pass replay the source untransformed. 500 episodes ≈ 5.6 templates × a can offset (473/500 near-duplicates). The dominant failure (left loses the can within ~3 frames of taking it) is identical for both policies. | source attribution 5.5 mm; diversity metrics; stage analysis | **CONFIRMED** (structure); "this is why both plateau near 50 %" SUPPORTED HYPOTHESIS | high: caps both policies | medium: needs pass-segment variation (later generation) or a stage-targeted eval |
| 3 | **Evaluation measurement.** Wall-clock 15 s = 270–321 simulated steps, fewer in successful episodes; success OR-ed over any step (including the unrecorded return); can poses unseeded and unlogged; one run per checkpoint; ±10 points of binomial noise at n = 100. | sim-server/robot code; openarm_eval.json; repeat runs 52 vs 49, 45 % vs 47 % | **CONFIRMED** | high for any conclusion drawn from 47 vs 49 | easy |
| 4 | **Episode horizon ≠ demo horizon.** Manual demos end at step 210–265; evaluation runs to ~290; the manual policy opens the left hand at step 229–282 (median 250) in 43/49 successes. "Held at the end" would read 3 vs 47. | gripper commands; visual spot-check | **CONFIRMED** (behaviour); horizon interpretation SUPPORTED HYPOTHESIS | high for the metric, not for 47 vs 49 | easy |
| 5 | **Visual domain: 0/500 generated episodes look like the evaluation** (camera pose, background, lighting, can size randomized; evaluation has no randomization). | frame correlation 0.40 vs ≥ 0.9997 on all three cameras | **CONFIRMED** (gap); its effect on success UNKNOWN — the generated policy still beat the manual one where it had starts, so it is not the dominant cause | medium / unknown | medium: needs a no-DR generation or an eval with DR (later) |
| 6 | **Generated hand-over differs**: both hands closed 10–17 frames (manual 1–10), can jolt > 1 m/s in 22 %, tilt beyond any manual demo in 44 %. | can ground truth; `check_mimic_demos.py` agrees (108/500) | **CONFIRMED** (artefact); effect on success SUPPORTED HYPOTHESIS at most — the pass-failure rate is the same for both policies | low–medium | easy: train on the filtered set (C) |
| 7 | **Slower generated timing** (+20–40 frames in data; +45 steps in closed loop; successes finish at step 227–278 vs 181–227). | events, eval stages | **CONFIRMED**; currently binding in ≤ 1 episode | low now, high under a stricter step budget | easy |
| 8 | Unequal training budget (8× samples, 9.5 vs 67.7 frame passes); "undertraining" | configs; no loss curves or 20k checkpoint available | budget difference CONFIRMED; undertraining **UNKNOWN**, not supported by closed-loop evidence | unknown | easy if the 20k checkpoint exists |
| 9 | Action labels = smoothed next state / wrong lag / wrong gripper semantics | frame-exact match to joint targets; identical lead k = 2; same units and order | **DISPROVEN** | — | — |
| 10 | Generated demos are largely invalid | 1/500 proven invalid for the eval controller (#91); 500/500 end held | **DISPROVEN** | — | — |
| 11 | Normalization / camera mapping / chunking differences | identical recipe, normalizers = own data | **DISPROVEN** | — | — |

## What this means for the research question

"More generated demonstrations" did not help because their usefulness is **local**: they improve the policy only
where their start states and phases overlap the evaluation, and the generator (a) spent its budget away from a third
of the evaluation distribution, (b) duplicated the hand-over instead of varying it, and (c) rendered every episode in
a visual condition the evaluation never uses. That is a measurable, testable account; a heatmap of starts alone
would not have shown it — the policy × position interaction and the phase-level duplication are the evidence.
