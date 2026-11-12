# A: can-appearance fix, frozen protocol (Pringles body camera)

**Frozen before any new rendering**, on 2026-11-12. Its sha256 is recorded in `summary.json`.

## Question
Does a renderer change improve how the **moved object** looks in the body camera, without hurting the rest of the image?
R1 (`../r1_relocation/`) found that the can region falls from 14.9 dB (0–3 cm) to 11.5 dB (15–21 cm), and the lid edge
tears.

## Data and frames
- **Sources:** `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`, episodes 0–9, with the cached reconstructions
  of `configs/openarm_pringles_handover.yaml`.
- **Pairs:** the R1 cross-episode pose transfer (render A's frame with B's real joints and can pose; ground truth = B's
  real frame).
- **Frames:** R1's 4 phases × 3 offsets.
- **Regions:** R1's xbench regions (`can`, `robot`, `revealed`, `other`, `all`).
- **Splits** (by episode; pairs never cross a split):
  - **dev:** ordered pairs with A, B ∈ {0, 1, 2, 3, 4} (20 pairs, 240 frames);
  - **held-out:** ordered pairs with A, B ∈ {5, 6, 7, 8, 9} (20 pairs, 240 frames);
  - **identity:** same-episode re-render (R1's `REF_K = 8`) for the held-out episodes (60 frames).

## Conditions
- **Baseline:** today's render config (`tm_erode 21, despeckle, part_fill off, plate_adapt 10`).
- **Candidates:** new `render:` keys that default to off, alone or combined:
  - `object_erode`: the silhouette erosion for object pixels;
  - `object_texture`: a per-proxy-point colour atlas from the episode's own frames;
  - `object_shadow`: move the object's contact shadow on the table plane;
  - an optional object relight.

## Metric
- Pair-level mean PSNR per region (the mean over the pair's frames).
- Gain = candidate − baseline, on identical frames.

## Choice (dev only)
- Among the candidates whose dev mean whole-image gain is ≥ −0.1 dB, choose the one with the highest dev median
  can-region gain.
- Dev may be re-run as often as needed. Held-out is rendered **once**, for the chosen config only.

## Adoption rule (held-out)
All three must hold:
1. The median pair-level can-region gain is ≥ +0.5 dB, **and** its 95 % CI lower bound is > 0. The CI is an
   episode-cluster bootstrap: resample the 5 held-out episodes, weight each pair by the product of its two episodes'
   multiplicities, 2000 resamples, seed 0.
2. The mean whole-image gain has a 95 % CI lower bound ≥ −0.1 dB.
3. The identity (same-episode) mean whole-image PSNR changes by ≥ −0.1 dB.

**Outcome:**
- **Pass:** the candidate keys are enabled in the new dataset configs.
- **Fail:** the baseline renderer is kept, and the result is reported as a negative finding.

## Not claimed
- PSNR is a proxy. The can poses are reconstructions, so they are ground truth up to reconstruction error.
- Episodes 0–9 are reused development data.
