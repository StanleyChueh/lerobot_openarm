# v4B.1 interpretation and limitations

**Status: Executed — Pending Independent Codex Review.** Numbers come from `V4B1_RESULTS.md` and `results.json`.

All conclusions here concern open-loop action error on reused development recordings, with 3 paired seeds and 1,000 updates. Causal wording applies only within these training interventions. Labels marked *exploratory* go beyond the frozen §7 rules.

## A. Did generated RGB improve learning? (H1)

**Not established under the frozen rule.**
* Far-grasp incremental value of generated synthetic RGB over the missing-camera treatment: +2.8% [−1.2, +7.3]. All three seeds are positive, but the effect is below 3% and the CI includes 0.
* Secondary metrics (*not the registered H1 endpoint*): near grasp +4.6% [+2.7, +6.7], approach +4.5%, carry +2.0%, phase-balanced +1.8% [+1.1, +2.6]. Each has a CI above 0.

Read together, generated RGB gives a **small, broad, consistently positive** contribution. That contribution is not large enough on the registered far-grasp endpoint to pass.

Most of the far-grasp gain comes **without** synthetic images: NoSynthRGB vs R is +7.2% [+3.4, +10.9], and the retained-gain fraction is 0.73 [0.38, 1.14]. The synthetic state/action trajectories, i.e. the kinematic retargeting, carry most of the measured far-grasp benefit.

**Limitation (unchanged from the registration):**
* Missing-camera availability identifies synthetic origin, so NoSynthRGB also learns a separate image-free mode.
* The contrast is "this renderer's RGB vs this missing-image treatment". It is not a content-only ablation, and it cannot attribute the effect to render artefacts or fidelity.

## B. Did PhaseMask improve handover compared with MatchedRandMask? (H2-D)

**No.**
* Handover with PhaseMask vs MatchedRandMask: −0.12% [−0.21, −0.03]; seeds −0.18 / −0.10 / −0.09. The difference is very small and, if anything, against the hypothesis.
* The contrast is low-powered by design: masks can differ only in the 19.25% of synthetic draws whose chunk crosses the grasp boundary. A larger true effect in either direction is not excluded by this design. The result is not evidence of equivalence.

*Exploratory:* PhaseMask beats matched random on **grasp** instead: far +1.7% [+0.9, +2.4] and near +1.7% [+1.3, +2.1], tightly consistent across seeds. With an equal number of retained targets, contiguous early (grasp-phase) targets help grasp prediction more than scattered targets do. This concerns grasp, not the handover question H2 asked.

## C. Was the original far-grasp advantage preserved?

**Yes vs R. Not resolved vs RpPos.**
* PhaseMask vs R: +13.5% [+8.3, +19.5], so P2 holds.
* PhaseMask vs RpPos: +0.3% [−2.0, +2.0].
* MatchedRandMask vs RpPos: −1.4% [−3.2, +0.0].
* NoSynthRGB keeps +7.2% vs R, and RpBoth keeps +9.9% (frozen).

## D. Did the handover guardrail pass?

**No, for both H2 arms.**
* Handover degradation vs R: PhaseMask 7.1% [5.8, 8.6]; MatchedRandMask 7.0% [5.6, 8.5]. The v4B RpPos degradation was 3.8%, so both masks made it worse.
* NoSynthRGB has 1.9% [0.8, 3.1] degradation. That passes the 2% rule on its point estimate only; the CI permits 3.1%.
* None of these new screens changes the v4B record:
  * v4B primary (RpPos) remains **FAIL**;
  * v4B secondary (RpBoth) remains **PASS**;
  * the original v4B.1 pre-training NO-GO at `1f80f2e` and the corrected GO at `d29013e` remain as recorded.

## E. Are the observed improvements larger than rerun variability?

The registered rerun arms are **bit-identical** to the frozen v4B checkpoints (same model SHA-256; prediction differences exactly 0.0). Rerun variability is therefore 0. Every nonzero effect formally "exceeds replication noise", and the spec §8 GPU-nondeterminism confounder is excluded.

This does **not** quantify retraining or seed variability; the rerun arms were never designed to. The relevant uncertainty is the paired per-seed spread together with the recording bootstrap, which is conditional on the three fitted seeds:

| effect | per-seed values | reading |
|---|---|---|
| H1 far grasp | +1.0 / +3.2 / +4.1 | spread of about 3 points, comparable to the effect itself |
| H2-A handover | −3.4 / −3.7 / −2.5 | consistent sign |
| PhaseMask vs R handover | −7.8 / −8.4 / −5.3 | consistent sign |
| D handover | −0.18 / −0.10 / −0.09 | consistent sign, magnitude about 0.1% |
| D far grasp | +1.6 / +1.6 / +1.8 | consistent |

Three seeds cannot estimate between-seed variance reliably.

## F. What remains unidentifiable

1. **Whole late-context removal.** Both H2 arms remove direct loss from about 60% of synthetic rows and raise real-target coefficients by about 1.52×. That bundle is shared, so its effect is not separable from the masks. The bundled contrast MatchedRandMask vs RpPos combines whole-row removal, real reweighting **and** within-row random thinning. Its −3.1% handover effect cannot be attributed to any single one of these.
   *Wording correction (re-audit non-blocking item 1): A1.3's phrase "context-removal and reweighting share" should be read as this three-part bundle.*
2. **Input-side effects.** Masked synthetic rows still provide synthetic observations and noised action tokens as model inputs. Loss masking is not deletion of the data. Retained later targets can be conditioned on earlier excluded tokens (A1.4).
3. **Why handover regresses at all with position augmentation.** v4B.1 rules out one explanation: direct supervision on synthetic carry, handover and release targets is not what drives the regression, because removing it makes handover worse. The cause is not identified. Candidates include:
   * fewer real rows per batch (8 vs R's 16);
   * input-distribution or state shift from synthetic rows;
   * interaction between the synthetic early-phase motion and later prediction.
   None is tested here.
4. **Image-free mode in H1.** We cannot separate "generated RGB carries useful content" from "a missing-image mode is a worse training domain".
5. **The replication guard's scope.** *Wording correction (re-audit non-blocking item 2):* the noise guard applies to the effect classifications (H1 useful/harmful, H2 A/D and change vs RpPos). The original P2 vs R and the point-estimate 2% guardrail keep their registered formulas, as `results.json` reports them.
6. **Generalisation.** Recordings 10–49 have been reused across studies. Only 2 far recordings are covered by generated positions. There is no fresh locked evaluation, no closed-loop or robot result, and no force evidence.

## G. Implications for the simulation-free generator

These are *exploratory* recommendations, not established findings.

* **Synthetic late-phase supervision appears useful, not harmful.** Withholding it worsened handover by about 3 points for both masks. Filtering or phase-masking synthetic targets is therefore **not** a fix for the handover regression. It would also not be a novel contribution (STEP 9).
  The direction this supports: generate *genuinely valid* post-grasp carry, handover and release trajectories, with changed bimanual object relations and consistent views. That is the v4C path (interaction-aware generation vs strong G+), rather than copying or suppressing source late-phase segments.
* **Kinematic trajectory generation is the main source of the measured grasp gain.** The current body-camera renderer adds a small, broad improvement: about +2–5% on several metrics, and handover closer to R. RGB fidelity (v4A: shared object clock, texture, depth and provenance) is a plausible lever but **not yet a demonstrated bottleneck**. A fidelity-improved renderer should be tested with this same missing-camera H1 contrast, which can be reused.
* **Target placement matters for grasp.** At an equal target count, contiguous grasp-phase targets beat scattered ones. Phase-aware weighting may be a useful training detail for a generator paper, but on its own it is not a contribution.
* **Remaining requirements before any publication claim:**
  * 1/3/5-demo data-efficiency;
  * fresh locked recordings (R+Both remains the confirmation candidate);
  * closed-loop real-robot evaluation;
  * independently validated generated RGB/action data.
  v4A, v4C and v4D remain open.
