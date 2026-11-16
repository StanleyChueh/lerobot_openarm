# R04 protocol — even per-position quota (pre-registered 2026-11-16, before any selection is run)

Motivation (user, real-robot evaluation of `ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_loose`): front
and middle ~100 %, rear-left / rear-right still fail sometimes. Measured training density (episodes, real + synthetic,
within 4 cm of the eval starts, mean per region): front-right 20.5, front-left 21.6, middle 20.2, rear-right 4.5,
rear-left 6.7 (max/min 4.8); real demos near the rear: 0. User choice: even quota + fill the rear, for all robots/tasks.
Method change #3 (of at most 4).

## Rule (selection only; generation = pipeline v1: joint_fallback + auto/explicit targets + partial visibility)
- Object position of an episode: real demo -> its source position; object_workspace -> its target; interaction
  location/timing -> its source position (object not moved). Cell = nearest point of the 3 cm lattice.
- Quota Q = 2 episodes per cell in the final source + synthetic set. Real demos are always kept and count first
  (a cell with >= Q real demos gets no synthetic episode).
- Fill order: cells with the fewest passing+rendered candidates first; within a cell prefer the least-used source demo
  so far, then the lowest distortion score, then the smaller shift. Candidates must pass every gate (unchanged).
- The synthetic-only dataset = the selected synthetic episodes (same set as in the mixed one).

## Hypotheses
- H1 (OpenArm, pipeline v1 candidates `r03_openarm_v1`): max/min of the mean training density within 4 cm of the 40
  eval starts across the 5 regions <= 1.5 (v03r2 loose: 4.8). Rear-left and rear-right density each >= 2x v03r2's.
- H2 (all tasks): per-cell counts are all <= Q (by construction); report covered cells, holes (lattice cells within
  12 cm of a source but uncovered), per-source counts (max/mean), total episodes, and what was trimmed (incl. handover
  variants), vs R03.
- H3: every selected episode passes all gates; length 1.000x; verify passes.
Negative results are kept. The published v03r2 / R03 repos are not modified.

## Q=2 results (kept; builds stopped, superseded)
RoboTwin `s1f`: cup 122 selected / 145 candidates (98 cells), stack 186 / 315 (118 cells), handover 85 / 244 (58 cells;
only 3 of 63 handover location/timing variants survive: they sit at real-demo cells). Totals differ per task.

## Amendment 1 (2026-11-16, user proposal, before any N=100 selection): fixed N = 100 synthetic episodes per dataset
Same position/cell definitions, real demos count first. Fill 100 slots in rounds: round r adds the r-th episode to
cells, cells ordered by (current count, then -distance from the cell to the nearest already-occupied position: real
demos + episodes selected so far); within a round, cells are taken one at a time, recomputing that order (farthest-point
sampling), so if the slots run out inside a round the chosen cells are spread evenly. Within a cell: least-used source,
then lowest distortion, then smaller shift. Shortfall (< 100 candidates) is reported, never padded.
H1 (OpenArm) and H2/H3 unchanged; additionally report per-cell count distribution and the region densities.
