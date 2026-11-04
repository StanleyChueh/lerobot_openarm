# Gate 1R results: bounded recovery of the contact-consistency verifier

**Verdict: FAIL → recommendation NO-GO for this verifier line. Gate 2 stays NO-GO.**

* **The failed criterion:** pre-registered criterion **K3** (time offsets of 6 and 10 frames detected in 3/3, both
  directions) is not met. A 6-frame *lead* is UNKNOWN in 3/3 episodes.
* **Stop rule:** under the rule frozen before the run (`GATE1R_SPEC.md`), this ends the line. No further threshold or
  window tuning. All other criteria (K1, K2, K4, K5) hold.
* **Sealed test:** episodes 1, 2, 3, 5, 6, 7, 9 were **not** tested. No human ground truth exists, so the annotation
  blocker applies.

Date 2026-11-04. Spec `gate1r-1.0`, frozen in commit `db02192` before the benchmark ran. The results commit follows it.
CPU only: no training, simulation, hardware, generator or pushes.
Companions:
* [GATE1R_DIAGNOSIS.md](GATE1R_DIAGNOSIS.md)
* [GATE1R_SPEC.md](GATE1R_SPEC.md)
* `outputs/gate1r/`

## 1. Codex's three critical failures (tasks 1–2)

| failure | Gate 1 verifier (frozen) | why it passed | Gate 1R combined |
|---|---|---|---|
| +1 rad on all 14 arm actions | PASS 3/3, all scores bit-identical | no action field in `Sample` | **FAIL 3/3** (A) |
| 60 px excursion in the transfer gap | C_rel PASS 3/3, unchanged to 1e-10 | gap frames in no scored window; take-over compares two fits | **FAIL 3/3** (C2) |
| receiver evidence removed | C_rel finite PASS 3/3 | NaN components skipped by `max` | **UNKNOWN 3/3** (D) |

Details, frame ranges and code paths are in GATE1R_DIAGNOSIS.md §1. Reproduction:
`gate1r/reproduce_failures.py` → `outputs/gate1r/reproduction.json`, and tests `Gate1BlindSpots` and
`RealPilotCounterexamples`.

## 2. The verifier (tasks 4–6)

| component | what it checks |
|---|---|
| **A** action-state | wrapper contracts, a dev-frozen command-response model, gripper command → measured jaw |
| **B** object–gripper | calibrate, then predict: no refit on scored frames; observability and attachment priors |
| **C** bimanual phase | measured jaw order; the observed track over the whole transition; departure from the giver |
| **D** evidence | per-segment coverage ≥ 0.80 → otherwise UNKNOWN |

* **Combined:** FAIL > UNKNOWN > PASS. **Absolute contact is UNKNOWN for every sample** (no measured geometry; G1
  pending). No single rigidity score is used as proof.
* **Calibration separated from evaluation:**
  * the camera is the inherited body_cam model, hashed `camera_hash` in `results_full.json`, and never refit;
  * thresholds and lags are frozen per fold from the two development episodes only;
  * the sealed episodes are never loaded, and `run_sealed()` refuses.
* **Calibration validity:** the camera fit overlaps the pool imagery. It is **unvalidated** until G1 (human robot
  landmarks) passes.

## 3. Benchmark (tasks 7–8): 3 folds, 180 samples (60 per held-out episode)

Cells are **FAIL / UNKNOWN / PASS** counts. For invalid samples FAIL is the desired outcome and PASS a silent miss; for
valid samples PASS is desired.

| family (n) | **combined** | A | B | C | D | Gate 1 C | Gate 1 C_rel |
|---|---|---|---|---|---|---|---|
| valid originals (3) | **0/1/2** | 0/0/3 | 0/1/2 | 0/0/3 | 0/1/2 | 2/0/1 | 0/0/3 |
| valid hard negatives (6) | **1/2/3** | 0/0/6 | 1/2/3 | 0/1/5 | 0/2/4 | 4/0/2 | 0/0/6 |
| action offset, all channels (15) | **15/0/0** | 15/0/0 | 0/5/10 | 0/0/15 | 0/5/10 | 10/0/5 | 0/0/15 |
| action offset, RJ4 only (15) | **15/0/0** | 15/0/0 | 0/5/10 | 0/0/15 | 0/5/10 | 10/0/5 | 0/0/15 |
| action time shift (18) | **18/0/0** | 18/0/0 | 6/4/8 | 5/3/10 | 0/3/15 | 14/0/4 | 6/0/12 |
| action L/R swap (3) | **3/0/0** | 3/0/0 | 3/0/0 | 3/0/0 | 0/3/0 | 3/0/0 | 3/0/0 |
| gap excursion px (9) | **8/1/0** | 0/0/9 | 0/3/6 | 8/0/1 | 0/3/6 | 6/0/3 | 0/0/9 |
| transition excursion 3D (9) | **7/1/1** | 0/0/9 | 0/3/6 | 7/0/2 | 0/3/6 | 7/0/2 | 3/0/6 |
| missing evidence (21) | **0/21/0** | 0/0/21 | 0/19/2 | 0/21/0 | 0/21/0 | 14/0/7 | 1/0/20 |
| video–state offset (18) | **13/5/0** | 0/0/18 | 11/7/0 | 7/11/0 | 0/3/15 | 18/0/0 | **18/0/0** |
| world displacement (18) | **11/4/3** | 0/0/18 | 9/6/3 | 5/0/13 | 0/6/12 | 15/0/3 | 10/0/8 |
| hand-over slip + transfer failure (9) | **9/0/0** | 0/0/9 | 9/0/0 | 6/0/3 | 0/3/6 | 9/0/0 | 8/0/1 |
| phase errors as command edits (9) | **9/0/0** | 9/0/0 | 5/2/2 | 8/0/1 | 0/3/6 | 9/0/0 | 9/0/0 |
| cross-episode pairs (6) | **3/3/0** | 0/0/6 | 3/3/0 | 0/6/0 | 0/2/4 | 6/0/0 | **6/0/0** |
| *ambiguous*: hand-rigid displacement (18) | 0/6/12 | — | — | — | — | 15/0/3 | 0/0/18 |
| *ambiguous*: constant 20 px (3) | 0/1/2 | — | — | — | — | 2/0/1 | 0/0/3 |

Gate 1 C's raw FAIL counts are inflated: it false-rejects 2/3 originals, and every corruption of those episodes inherits
the rejection.

**By magnitude** (combined, of 3 folds):

| corruption | detected (FAIL) |
|---|---|
| action offset 0.05 / 0.1 / 0.25 / 0.5 / 1 rad (all channels; and RJ4 alone) | 3 / 3 / 3 / 3 / 3 (same for RJ4) |
| action time shift ±3 / ±6 / ±10 frames | 3 in every direction and size |
| gap excursion 15 / 30 / 60 px | 2 (1 UNKNOWN) / 3 / 3 |
| transition excursion 1 / 2 / 4 cm | 1 (1 UNKNOWN) / 3 / 3 |
| missing receiver 25 / 50 / 75 / 100 % / Gate-1 window; missing transition; missing giver hold | UNKNOWN 3/3 each; never PASS |
| video–state lag 3 / 6 / 10 | 3 / 3 / 3 |
| video–state lead 3 / 6 / 10 | **2 / 0 / 2** (UNKNOWN 1 / 3 / 1) |
| world displacement 1 cm x / y; 2 cm x / y; 4 cm x / y | 1 / 1; 3 / 1; 3 / 2 |
| slip 2 / 4 cm; transfer failure | 3 / 3; 3 |

**Ablations** (combined minus one component):
* A is the only detector of action-offset corruptions: 30 → 0 without it.
* C is the only detector of excursions: gap px 8 → 0, 3D 7 → 0.
* B adds the 2 cm slip (3 → 0 without it) and x-axis world displacements.
* D changes no family; B and C already enforce coverage.

Full per-variant tables are in `outputs/gate1r/summary_tables.json`.

## 4. Falsification criteria (pre-registered, combined verifier)

| criterion | result | met |
|---|---|---|
| K1 originals: 0 FAIL, ≤ 1 UNKNOWN | 0 FAIL, 1 UNKNOWN (ep 4: receiver coverage 0.73) | yes |
| K1 hard negatives: ≤ 1 FAIL / 6 | 1 FAIL (ep 8, 1.5 px noise: giver p90 16.8 vs τ 15.5), 2 UNKNOWN | yes (at the limit) |
| K2 +1 rad action: FAIL 3/3 | 3/3 | yes |
| K2 60 px gap excursion: FAIL 3/3 | 3/3 | yes |
| K2 missing receiver: never PASS | UNKNOWN 3/3 | yes |
| **K3 video–state ±6, ±10: FAIL 3/3 each** | lag 6: 3, lag 10: 3, **lead 6: 0 (UNKNOWN 3)**, **lead 10: 2 (UNKNOWN 1)** | **no** |
| K4 action offsets ≥ 0.1 rad: FAIL 3/3 | 3/3 at every magnitude, including 0.05 rad | yes |
| K5 no silent PASS below coverage | 0 of 21 | yes |

**Why K3 fails** (GATE1R_DIAGNOSIS.md §3):
* The frozen observability rule scales the attachment covariance by the calibration-fit residual.
* A time-shifted track misfits its calibration segment: 21–45 px, against 4–6 px for originals.
* That inflates the predicted uncertainty and turns detectable inconsistency into abstention. This is a design defect of
  the frozen rule, not an implementation bug.
* The same mechanism halves cross-episode detection (3/6).
* On exactly these two families, the repaired verifier is **worse than Gate 1's C_rel** (temporal 13 vs 18;
  cross-episode 3 vs 6).

## 5. Human annotation workflow (task 3) and the sealed-test blocker (task 9)

* **Tool:** `gate1r/annotation/annotate_gate1r.py`, blind, resumable, with two tasks:
  * **keypoints:** lid centre and bottom; two pad tips per gripper; wrist tags. Each with visible / occluded /
    out-of-frame and confidence 1–3. The manifest has 147 frames: key frames, dense grasp/lift and transition windows,
    and 19 repeats.
  * **events:** right first contact, lift-off, left first contact, right release. Each with confidence, or marked not
    observable.
* **Provenance on every row:** annotator, session, tool SHA-256, repository commit, dataset revision, video SHA-256,
  SHA-1 of the exact decoded frame, UTC time.
* **Instructions:** `ANNOTATION_INSTRUCTIONS_GATE1R.md`.
* **Evaluator:** `evaluate_gate1r_annotations.py`. Its rules were fixed in advance:
  * ground truth needs visible, confidence ≥ 2 and annotator agreement;
  * G2 enforces ≥ 90 % tracker coverage of human-visible frames per window;
  * G1 needs ≥ 20 points per gripper.
  It returns **PENDING (exit 3)**. No automatic prediction is substituted for ground truth.
* **Not exercised:** the GUI loops were not run in this session. The session logic, ground-truth rules and evaluator are
  unit-tested.
* **Blocker:** `sealed_ready` requires G1, G2 and EV to pass on the pilot **and** human annotations of the sealed
  episodes' required windows. Neither exists, so the sealed test is not run, and `run_sealed()` refuses.

## 6. Recommendation (task 10)

* **NO-GO for the visual contact-consistency verifier line,** as frozen.
  * It closes all three of Codex's holes: action corruption, transfer excursion, missing evidence.
  * But it fails a pre-registered detection criterion, and regresses on time-offset and pairing errors.
  * The false-rejection margin is thin (one hard-negative FAIL at 1.5 px noise).
  * A fix exists on paper (GATE1R_DIAGNOSIS.md §3: residual-consistency FAIL before observability; pose-only
    observability). Under the stop rule it is a new spec version needing a new holdout and human ground truth, not a
    tweak to this round.
* **Gate 2 (synthetic generation): NO-GO.**
* **What is salvageable as engineering:**
  * Component A detects every action-label corruption tested, down to 0.05 rad on one joint, with 0 FAIL on 9 valid
    samples. It is a sound dataset sanity check, independent of vision.
  * The annotation kit and the counterexample fixtures are reusable.
* **What must exist before any contact-consistency claim:**
  * human pilot ground truth (G1, G2, EV);
  * measured object and pad geometry (for absolute contact);
  * calibration evidence independent of the pool imagery;
  * a new frozen spec evaluated once on the sealed episodes.
* **No force claim** of any kind is made or supported.

## 7. Reproduction and tests

```bash
cd ~/Stanley_ws/lerobot_openarm
R="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python -W ignore::DeprecationWarning"
D=research/contact_aware_augmentation
$R $D/gate1r/reproduce_failures.py --out $D/outputs/gate1r/reproduction.json
$R $D/gate1r/run_gate1r.py --out $D/outputs/gate1r        # ~20 s
python3 $D/gate1r/tabulate.py
$R -m unittest discover -s $D/gate1r/tests -t $D/gate1r/tests -v
```

**Tests** (logs in `outputs/gate1r/test_output/`), all OK:
* Gate 1R: 19, including synthetic counterexamples and the real-pilot fixture `tests/fixtures/pilot_inputs.npz`
* Gate 1: 17
* Gate 0: 11
* dataset diagnostics: 39

**Deviations after the freeze:** none to the verifier or the spec. Added afterwards, for analysis only:
`gate1r/tabulate.py` (reads `samples.jsonl`; no re-scoring). Dates in the Gate 0/1 documents were corrected from
2026-11-05 to 2026-11-04, the actual run date (Codex's provenance note).
