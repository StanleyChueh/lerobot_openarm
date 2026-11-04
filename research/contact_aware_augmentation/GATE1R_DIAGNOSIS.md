# Gate 1R diagnosis: why the Gate 1 verifier passed Codex's three failures, and why the repair still fails

Date 2026-11-04.

* **Inputs:** [CODEX_GATE1_REVIEW.md](../../CODEX_GATE1_REVIEW.md) (review of `1069ddb`), Gate 0 and Gate 1 reports.
* **Evidence:**
  * `outputs/gate1r/reproduction.json` (`gate1r/reproduce_failures.py`, frozen Gate 1 code and folds);
  * `outputs/gate1r/samples.jsonl` and `summary_tables.json` (frozen Gate 1R benchmark);
  * tests in `gate1r/tests/`.

## 1. The three failures, reproduced (3/3 each, real pilot episodes 0/4/8)

### F1: +1 rad on all 14 arm action channels passes

* **Reproduced:** all eight Gate 1 method scores are identical to the original on every episode. C_rel passes 3/3.
* **Cause (structural):**
  * `gate1_core.Sample` has the fields `name, family, valid, joints_ep, video_ep, T_hand, events, uv, ok, meta`. There is
    no action field.
  * `T_hand` comes from the measured state, and `events` from the gripper commands only.
  * No code path reads an arm action, so no arm-action corruption of any size can change a score.

### F2: a 60 px excursion during the transfer passes

* **Reproduced:** with a smooth excursion peaking at 60 px over frames 174–188 (ep 0), 161–175 (ep 4) and 158–171
  (ep 8), C_rel scores are unchanged to 1e-10.
* **Cause (window design):**
  * C_rel scores three windows: the giver window `[g_close+10, r_close−3]`, the receiver window
    `[max(g_open, r_close)+8, tail]`, and the release window `[g_open+8, g_open+25]`.
  * The frames `r_close−2 … g_open+7` (15 / 15 / 14 frames) belong to none of them. That is the actual transfer.
  * The only component touching the transfer, `takeover_px` (in C, not C_rel), compares the two *fitted predictions*
    `π(T_g p_g)` and `π(T_r p_r)`. It never compares them with the observed track, so it is unchanged too.
* **Side note:** the phase-free mixture B_rel scores every frame and does react (Codex: flagged eps 0 and 4). The blind
  spot was introduced by the phase-window selection.

### F3: no receiver evidence at all passes

* **Reproduced:** with `ok = False` over the whole receiver window, `res_recv` is NaN with n = 0. C_rel returns a finite
  score below 1 in 3/3 (`unevaluable_component` appears only in the explanation list).
* **Cause (aggregation):**
  * `method_score` skips non-finite components (`continue`) and takes the maximum of the rest.
  * The summary counts a sample as unevaluable only if *every* component is NaN.
  * So missing evidence silently becomes a pass.

### Common root cause

The Gate 1 verifier answered a single question: *does some rigid point explain the frames we chose to look at?* It had:
1. no action input;
2. window choices that left the most informative interval unobserved;
3. evidence-agnostic aggregation.

A low rigidity residual was being read as certification. These are design properties, not tuning errors, and the tests
`Gate1BlindSpots` (synthetic) and `RealPilotCounterexamples` (recorded fixture) now pin them.

## 2. What Gate 1R changed (frozen in `gate1r-1.0` before any recorded-data run)

| failure | repair | result on the frozen benchmark |
|---|---|---|
| F1 | **A:** actions are part of every sample. Contracts: request–state gap ≤ 0.5 rad and jump ≤ 0.25 rad (the wrapper's safety-hold limits), URDF limits. A per-joint command-response model is frozen on dev. Gripper commands must move the measured jaw | +1 rad: **FAIL 3/3**. All offsets from **0.05 rad**, on all channels or one joint, and action time shifts ±3/6/10: FAIL 3/3 each |
| F2 | **C2:** the observed track over the whole transition `[r_close−10, g_open+8)` is compared, frame by frame, with the frozen attachments of whichever hands the *measured* jaws show holding. Segments now tile the manipulation with no unscored frame | 60 px gap excursion: **FAIL 3/3**; 30 px 3/3; 15 px 2/3 (1 UNKNOWN); 3D 2 and 4 cm 3/3; 1 cm 1/3 |
| F3 | **D:** each required segment needs ≥ 80 % tracker-ok coverage. Any missing component makes the combined verdict UNKNOWN; a combined PASS needs every component PASS | Codex's case: **UNKNOWN 3/3**. All 21 missing-evidence variants: UNKNOWN, 0 PASS |
| Codex blind spot: the attachment is refit on the scored hold | **B:** the attachment is fitted on a 15-frame calibration segment and predicted forward. C uses whole-hold fits and never fits a frame it scores | slip 2 and 4 cm: FAIL 3/3 each (C_rel: 2/3, 3/3) |
| Codex: rigid offsets cannot be certified | absolute contact is UNKNOWN for every sample | hand-rigid displacement and constant offset are reported as ambiguous, never counted |

## 3. Why the repaired verifier still fails its own criterion (K3)

* **Pre-registered K3:** video–state offsets of 6 and 10 frames, both directions, must FAIL in 3/3.
* **Observed:** lag 6: 3/3, lag 10: 3/3, **lead 6: 0/3 (UNKNOWN 3/3)**, **lead 10: 2/3 (UNKNOWN 1)**. Lead 3 is 2/3, and
  cross-episode pairs are only 3/6 FAIL (3 UNKNOWN).
* **Comparison:** Gate 1's C_rel detected all of these: temporal 18/18, cross-episode 6/6.
* **Mechanism** (from `samples.jsonl`; every UNKNOWN reason is `prediction_cannot_resolve_threshold`,
  `transition_prediction_cannot_resolve_threshold`, or its consequence `transition_needs_both_attachments`):
  1. A time-shifted or mismatched track does not fit a rigid point on the calibration segment. The giver calibration RMS
     is **21.6–37.5 px** for lead 6 (lead 10: 38–45 px; cross-episode up to 57 px), against **4.2–6.3 px** for the
     originals and 4.6–5.6 px for the 1.5 px-noise controls.
  2. The specification sets the attachment covariance as `σ² (JᵀJ)⁻¹`, with **σ² = median(calibration residual)²**.
     It then abstains when the propagated prediction σ exceeds 0.5·τ.
  3. The inconsistency therefore inflates its own uncertainty estimate. Evidence of failure is converted into
     abstention, and the larger the inconsistency, the more likely UNKNOWN.

This is a **design defect in the frozen rule**, not an implementation bug. The code does exactly what `spec.py`
states, so it cannot be repaired under the stop rule.

**What a correct rule would need** (recorded for the next spec version; NOT implemented or evaluated here):
* a calibration-segment consistency test that FAILs on a large calibration residual, with its own development
  threshold, evaluated *before* observability;
* an observability measure from motion excitation alone (pose-only Jacobian conditioning), not scaled by the residual
  of a possibly inconsistent fit.

Whether that would pass K1–K5 is unknown, and finding out would be another round on the same three episodes. Testing it
requires a new spec version, a new holdout and human ground truth.

## 4. Secondary findings

* **False-rejection margin is thin.** The 1.5 px-noise hard negative of ep 8 FAILs. Its giver prediction p90 is 16.8 px
  against τ = 15.5 px, a threshold frozen from two development episodes whose own p90 values are 10.4 and 8.4 px.
  K1 allows one such FAIL among six; it would not survive a larger sample.
* **Coverage abstention on a real episode.** The ep 4 original is UNKNOWN: receiver prediction coverage is 0.73, because
  the lid moves fast and then leaves the image. This is the abstention Codex asked for, and it shows the 1 s receiver
  window is already at the limit of what body_cam sees.
* **Component necessity (ablations):**
  * removing **A** loses all 30 action-offset corruptions (FAIL 30 → 0) and most action time shifts (18 → 7);
  * removing **C** loses every excursion;
  * removing **B** loses the 2 cm slip (3/3 → 0/3) and the x-axis 2 and 4 cm world displacements (3/3 → 1/3);
  * removing **D** changes no family, because B and C already enforce their own segment coverage. D is redundant as
    implemented, but it states the contract explicitly.
* **Lag:** per-fold lags are −1.25, −1.5, −1.5 frames (Gate 1: −1.5, −1.5, −1.75). This is model-dependent, as Codex
  noted, and the change of windows moves it.

## 5. Blind spots that remain by construction

* **Rigid wrong attachment and constant image offset:** indistinguishable from a different legitimate grasp without
  measured object/pad geometry. Absolute contact is UNKNOWN.
* **A single lid-centre point:** cannot validate object orientation, pad contact surface, force closure, or motion
  along the camera ray.
* **Measured-state errors that project the TCP to the same pixel** escape object geometry. G1 robot-landmark
  reprojection is required, and it is PENDING.
* **The camera model** is the inherited silhouette fit on overlapping pool imagery. It is hashed and never refit, but
  it is unvalidated.
* **The tracker's colour band** was chosen on the pilot episodes, so pilot scores are development results.
