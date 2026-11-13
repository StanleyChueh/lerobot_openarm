# Claude audit of Codex Step 1 (diagnosis) and Step 2 (v02), 2026-11-13

Scope: read-only checks plus independent recomputation. Evidence is in `audit_claude/`. Base: worktree
`research/sfagen-v02` @ 77e52d4 + Codex local changes. Verdict: **Codex's implementation and reporting are sound. No
numerical error was found.** The v02 result misses its main goal. Some limitations of the planner and the diagnosis are
recorded below.

## Reproduced (PASS)

| Check | Result |
|---|---|
| Fisher two-sided, real-10 10/20 vs mixed 15/20 | p = 0.1908 (Codex 0.19079) |
| Unit tests (`unittest discover -s tests`) | 53/53 OK (`audit_claude/tests.log`) |
| `FREEZE.json` vs current files | all OK except `build.py`, the documented packaging amendment (`audit_claude/freeze_check.txt`) |
| Exports, independent reader (`audit_claude/check_exports.py`, own URDF XML parse) | both conditions: 104 ep / 26,781 frames synthetic; 114 / 29,356 mixed; `is_synthetic` 0 on real, 1 on synthetic; 0 synthetic episodes identical to a real one; 0 duration mismatches; 0 URDF ∪ source envelope violations; mixed synthetic arrays equal synthetic-only |
| Coverage, recomputed from `eval_v00/coverage.json` + `v02/planned_targets.json` | real 6.63 cm, +v00 3.91 cm, +v02 6.63 cm worst nearest; matches |
| `checks.py` change (limit envelope = URDF ∪ original source) | matches protocol; stricter for timing variants, identical for spatial |
| `build.py` amendment | nulls heterogeneous codec/pix_fmt metadata only; MP4 bytes unchanged |

## Material finding: v02 does not cover the position it was built for

Worst nearest can start (cm) per evaluation position:

| | ① front-right | ② rear-right | ③ rear-left | ④ front-left | ⑤ middle |
|---|---:|---:|---:|---:|---:|
| real | 2.98 | **6.63** | 5.59 | 3.30 | 3.33 |
| +v00 (13d7231a) | 1.12 | **3.91** | 2.81 | 1.44 | 1.23 |
| +v02 | 2.84 | **6.63** | 3.90 | 1.37 | 1.68 |
| full planned grid | 1.03 | 1.09 | 0.89 | 1.37 | 1.68 |

- **② rear-right gains nothing.** That position went 0/4 → 2/4 with v00 in the user's evaluation. v02 is worse than v00 at
  both rear positions.
- The rear row (x = 12/15 cm, targets 000–011) is rejected:
  - 30 attempts fail `limits`, where the exact shifted **action** on `openarm_right_joint6` exceeds URDF ∪ source by
    0.11–0.97°;
  - 26 attempts fail placement (robot points inside the inflated can box, `extra == 0` gate), mostly mid-grid.
- Root cause: `plan._retarget` bounds the IK *state*. The label A(t) + dq(t+k) inherits the source's action–state
  gap, so it exceeds the bound. This is a planner limitation, not a Codex error. Codex preserved the gate and reported it.
- **v00 is not "cheating"** (`audit_claude/v00_limits.json`): 0/65 v00 synthetic episodes exceed the same envelope.
  v00 covered ② with source 8 shifted (0, −3) cm, ep 66, 2.61 cm from the nearest rear-right start. v02 assigned the same
  source to target 006 with shift (−1.16, −3.58) cm, and the sfagen planner rejected it on joint 6.

## Smaller concerns (no rerun needed)

1. The coverage proxy moved to Step 1 measured source centres after generation (00:46). It is applied identically to
   real, v00 and v02, so the ranking is unaffected. It is still a post-hoc choice.
2. Rejected attempts stop at the first failed gate, so their other gates are unevaluated. The rejection-cause counts are
   lower bounds per cause.
3. Codex continued rendering and exporting for about 10 h after generation showed 18/35 targets. The protocol allowed it,
   but the coverage miss could have been raised before spending GPU time.
4. **Step 1 wrist finding is close to true by construction.** F1 translates the TCP and the can by the same d with fixed
   orientation, so the can-relative wrist geometry is invariant (camera rotation ~1e-15°, camera translation 3.00 cm,
   can displacement < 0.1 px). It rules out can-relative mismatch only. The copied wrist image still shows the table,
   plate and background 3 cm off for the new can position. That cue is untested, and it is exactly what the matched
   copied vs rendered-wrist ablation (v02 both conditions) can test.
5. Right-wrist validity is 0 in 59.8% of v00 synthetic frames. SmolVLA training does not read `wrist_view_valid`, so
   these invalid frames are trained on as if valid.

## Rerun

The planner limitation is addressed by a pre-registered development rerun, v02b (`../v02b/PROTOCOL.md`): action-aware
IK bounds on the 17 uncovered targets only, with gates unchanged.
