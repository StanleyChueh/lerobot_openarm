# Gate 1R specification (frozen before the benchmark run)

* **Spec version:** `gate1r-1.0` (`gate1r/spec.py`).
* **Freeze:** this file and all `gate1r/` code are committed **before** `run_gate1r.py` is executed for the first time
  on recorded data. The commit SHA is the freeze record. It is self-attested chronology, not independent registration.
* **What may change afterwards:** after the benchmark, a change to any constant, window, rule or criterion requires a new
  spec version and a new holdout. Bug fixes that do not change a decision rule are recorded under §6 "Deviations".

## 1. Scope

* **Task:** a bounded recovery of the Gate 1 verifier, after Codex's review of `1069ddb` found three critical silent
  passes.
* **Constraints:** CPU only; no generator, simulation, hardware or training.
* **Episodes:** development episodes 0, 4, 8 (leave-one-episode-out). Episodes 1, 2, 3, 5, 6, 7, 9 stay sealed, and
  `run_sealed()` refuses to score them until human ground truth exists.

## 2. Verifier

| component | checks | can return |
|---|---|---|
| **A** action-state | wrapper contracts (`max_tracking_error` 0.5 rad, `max_command_jump` 0.25 rad, URDF limits ± 0.05 rad); per-joint command-response model `Δs = α(a[t−d]−s) + β·Δs_prev`, fitted on dev, per-joint RMS threshold; gripper commands must move the measured jaw within 10 frames; no large uncommanded jaw move | PASS / FAIL |
| **B** object–gripper | per hold: the attachment point is fitted on a 15-frame **calibration** segment only, then **predicted** on the next frames (never refit). Gates: calibration coverage; predicted-pixel 1σ ≤ 0.5·τ (else UNKNOWN); attachment within 0.30 m of the TCP and in front of the camera (else FAIL). Decision: p90 prediction residual ≤ τ and no run of 5 frames above 2τ | PASS / FAIL / UNKNOWN |
| **C** bimanual phase | C1: command order and **measured** jaw order (receiver closes before giver opens). C2: the **observed** track over the whole transition [receiver close − 10, giver open + 8] against the frozen attachments of whichever hands the measured jaws show holding; 3-frame running median, max ≤ τ. C3: after release the object departs from the giver. Attachments for C come from each hand's whole hold (calibration + prediction), and C never fits a frame it scores | PASS / FAIL / UNKNOWN |
| **D** evidence | tracker-ok coverage ≥ 0.80 in **each** required segment (giver calibration and prediction, transition, receiver calibration and prediction, release) | PASS / UNKNOWN |

* **Combined:** FAIL if any component FAILs; else UNKNOWN if any is UNKNOWN; else PASS.
* **What PASS means:** kinematic-visual consistency was checked and held. **Absolute contact** (object between the pads)
  is UNKNOWN for every sample, because no measured object/pad geometry exists and G1 calibration is pending.

**Segments** (frames, from command events):

| segment | frames |
|---|---|
| giver calibration | [g_close+10, +25) |
| giver prediction | [g_close+25, r_close−10) |
| transition | [r_close−10, g_open+8) |
| receiver calibration | [g_open+8, +23) |
| receiver prediction | [g_open+23, g_open+38) |
| release | [g_open+8, g_open+25) |

There is no unscored frame between the giver calibration start and g_open+38.

**Thresholds:** frozen per fold from the two development originals: τ = max(1.5 × dev max, floor), or dev min / 1.5
for the departure check. Floors are a priori: response 0.004 rad/frame; B 10 px (the 8 px G1 budget plus margin); C 12 px.

* **Lag:** frozen per fold from development fits; it is a model-dependent nuisance parameter, not a measured camera age.
* **Camera:** the inherited body_cam model, hashed, never refit. Its calibration overlaps the pool imagery and is
  unvalidated.

## 3. Benchmark samples (per held-out episode)

| label | families |
|---|---|
| valid | original; hard negatives: 10 % random frame dropout, 1.5 px track noise |
| invalid | action offsets 0.05 / 0.1 / 0.25 / 0.5 / 1 rad (all 14 arm channels; RJ4 alone); action time shift ±3 / 6 / 10 frames; action L/R swap; gap excursion 15 / 30 / 60 px (Codex's window); transition excursion 1 / 2 / 4 cm (3D); missing receiver evidence 25 / 50 / 75 / 100 % and Codex's Gate 1 window; missing transition; missing giver hold; video–state offsets ±3 / 6 / 10; world displacement 1 / 2 / 4 cm; hand-over slip 2 / 4 cm; transfer failure; phase errors as gripper-**command** edits (giver releases 10 frames early, receiver window 15 frames early, L/R roles swapped in state and command); cross-episode pairs |
| ambiguous | hand-frame rigid displacement 1 / 2 / 4 cm; constant 20 px offset. These are indistinguishable from a different legitimate grasp without measured geometry, so they are reported but never counted |

**Baselines on the same samples:** Gate 1's C and C_rel, with their own frozen Gate 1 fold thresholds. Ablations remove
one of A–D from the combined verifier.

## 4. Falsification criteria (combined verifier, development folds)

| id | criterion |
|---|---|
| K1 | 0/3 originals FAIL and ≤ 1/3 UNKNOWN; ≤ 1 FAIL among the 6 hard negatives |
| K2 | +1 rad action: FAIL 3/3; 60 px gap excursion: FAIL 3/3; Codex's missing-receiver case: never PASS (3/3) |
| K3 | video–state offsets 6 and 10 frames, both directions: FAIL 3/3 each |
| K4 | all-channel action offsets ≥ 0.1 rad: FAIL 3/3 each |
| K5 | no missing-evidence sample whose minimum segment coverage is below 0.80 is PASS |

## 5. Stop rule and verdict ceiling

* **Single evaluation:** the benchmark is evaluated **once**.
* **If any of K1–K5 fails:** NO-GO for this verifier line. No further threshold or window tuning in a new round.
* **If all hold:** the verdict is at most **PARTIAL**. The sealed test needs human ground truth, which does not exist, and
  absolute contact is UNKNOWN by design.
* **PASS is unattainable in this round,** and that is stated before the run.

## 6. Deviations

* **Design revisions before the freeze** (made on the synthetic fixture only; no recorded-data scoring had run):
  1. The observability gate changed from attachment-parameter σ (15 mm) to the predicted-pixel σ ≤ 0.5·τ.
     Parameter σ along the camera ray barely moves the projection.
  2. C's attachments use the whole hold instead of the 15-frame calibration. The transition prediction could not
     resolve its threshold at 1.5 px noise.
  3. The receiver evidence window is 1 s (30 frames), because the lid leaves the image 1–2 s after release in the
     pilot.
* **After the freeze:** see GATE1R_RESULTS.md §Deviations. Bug fixes only.
