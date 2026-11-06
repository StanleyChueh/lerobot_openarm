# v4B.1 results (scripts/score_v4b1.py, Amendment A1 contract)

Mode: **final**. Open-loop action error on development recordings 10–49 (not robot success).

## Original images, seed-averaged error (lower is better)

| arm | far grasp | near grasp | approach | grasp | carry | handover | release | phase-balanced | gripper timing |
|---|---|---|---|---|---|---|---|---|---|
| R | 0.4110 | 0.3106 | 0.2821 | 0.3483 | 0.4654 | 0.3713 | 0.4448 | 0.3824 | 5.08 |
| RpBoth | 0.3705 | 0.2862 | 0.2667 | 0.3178 | 0.4713 | 0.3749 | 0.4451 | 0.3752 | 4.91 |
| RpPos | 0.3565 | 0.2907 | 0.2671 | 0.3154 | 0.4753 | 0.3855 | 0.4511 | 0.3789 | 5.13 |
| RpBoth_NoSynthRGB | 0.3812 | 0.3000 | 0.2795 | 0.3305 | 0.4812 | 0.3784 | 0.4416 | 0.3822 | 5.04 |
| RpBoth_rerun | 0.3705 | 0.2862 | 0.2667 | 0.3178 | 0.4713 | 0.3749 | 0.4451 | 0.3752 | 4.91 |
| RpPos_MatchedRandMask | 0.3614 | 0.2924 | 0.2660 | 0.3183 | 0.4745 | 0.3974 | 0.4481 | 0.3809 | 5.29 |
| RpPos_PhaseMask | 0.3553 | 0.2875 | 0.2650 | 0.3129 | 0.4739 | 0.3979 | 0.4474 | 0.3794 | 5.21 |
| RpPos_rerun | 0.3565 | 0.2907 | 0.2671 | 0.3154 | 0.4753 | 0.3855 | 0.4511 | 0.3789 | 5.13 |

## Paired contrasts (x vs ref; positive = x better), recording bootstrap 95% CI

| contrast | far grasp | near grasp | approach | handover | carry | release | pb | per-seed far grasp | per-seed handover | timing Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| RpBoth vs RpBoth_NoSynthRGB | +2.8% [-1.2, +7.3] | +4.6% [+2.7, +6.7] | +4.5% [+2.9, +6.1] | +0.9% [-0.2, +2.0] | +2.0% [+0.6, +3.5] | -0.8% [-2.6, +1.2] | +1.8% [+1.1, +2.6] | +1.0 / +3.2 / +4.1 | +0.6 / +2.2 / -0.0 | -0.14 |
| RpBoth_NoSynthRGB vs R | +7.2% [+3.4, +10.9] | +3.4% [-0.1, +6.5] | +0.9% [-0.8, +2.7] | -1.9% [-3.1, -0.8] | -3.4% [-5.2, -1.6] | +0.7% [-1.4, +2.6] | +0.0% [-1.0, +1.1] | +10.0 / +6.6 / +5.2 | -1.7 / -3.7 / -0.3 | -0.04 |
| RpBoth vs R | +9.9% [+6.2, +13.7] | +7.9% [+5.6, +10.1] | +5.4% [+3.2, +7.8] | -1.0% [-2.5, +0.6] | -1.3% [-2.3, -0.3] | -0.1% [-0.6, +0.4] | +1.9% [+1.1, +2.6] | +10.9 / +9.6 / +9.1 | -1.1 / -1.5 / -0.3 | -0.18 |
| RpBoth_rerun vs RpBoth | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0 / +0.0 / +0.0 | +0.0 / +0.0 / +0.0 | +0.00 |
| RpBoth_rerun vs RpBoth_NoSynthRGB | +2.8% [-1.3, +7.1] | +4.6% [+2.7, +6.8] | +4.5% [+3.0, +6.1] | +0.9% [-0.2, +2.0] | +2.0% [+0.6, +3.5] | -0.8% [-2.6, +1.2] | +1.8% [+1.1, +2.6] | +1.0 / +3.2 / +4.1 | +0.6 / +2.2 / -0.0 | -0.14 |
| RpPos_PhaseMask vs RpPos | +0.3% [-2.0, +2.0] | +1.1% [+0.3, +1.8] | +0.8% [+0.3, +1.2] | -3.2% [-4.9, -1.6] | +0.3% [-0.4, +1.0] | +0.8% [-0.5, +2.1] | -0.1% [-0.9, +0.6] | -0.2 / +1.1 / +0.1 | -3.4 / -3.7 / -2.5 | +0.08 |
| RpPos_PhaseMask vs R | +13.5% [+8.3, +19.7] | +7.5% [+5.0, +9.9] | +6.1% [+3.8, +8.5] | -7.1% [-8.6, -5.7] | -1.8% [-3.2, -0.6] | -0.6% [-1.8, +0.5] | +0.8% [-0.2, +1.7] | +13.3 / +14.1 / +13.3 | -7.8 / -8.4 / -5.3 | +0.12 |
| RpPos_MatchedRandMask vs RpPos | -1.4% [-3.2, +0.0] | -0.6% [-1.4, +0.1] | +0.4% [-0.0, +0.8] | -3.1% [-4.8, -1.4] | +0.2% [-0.5, +0.8] | +0.7% [-0.7, +1.9] | -0.5% [-1.3, +0.2] | -1.8 / -0.5 / -1.8 | -3.2 / -3.6 / -2.4 | +0.17 |
| RpPos_PhaseMask vs RpPos_MatchedRandMask | +1.7% [+0.9, +2.4] | +1.7% [+1.3, +2.1] | +0.4% [+0.2, +0.5] | -0.1% [-0.2, -0.0] | +0.1% [-0.2, +0.5] | +0.1% [+0.1, +0.2] | +0.4% [+0.2, +0.5] | +1.6 / +1.6 / +1.8 | -0.2 / -0.1 / -0.1 | -0.09 |
| RpPos_MatchedRandMask vs R | +12.1% [+6.7, +18.6] | +5.9% [+3.4, +8.3] | +5.7% [+3.5, +8.1] | -7.0% [-8.5, -5.6] | -2.0% [-3.4, -0.6] | -0.7% [-1.9, +0.4] | +0.4% [-0.6, +1.4] | +11.8 / +12.7 / +11.7 | -7.6 / -8.3 / -5.2 | +0.21 |
| RpPos_rerun vs RpPos | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0% [+0.0, +0.0] | +0.0 / +0.0 / +0.0 | +0.0 / +0.0 / +0.0 | +0.00 |
| RpPos_PhaseMask vs RpPos_rerun | +0.3% [-2.0, +2.0] | +1.1% [+0.4, +1.8] | +0.8% [+0.3, +1.2] | -3.2% [-4.8, -1.6] | +0.3% [-0.5, +1.0] | +0.8% [-0.5, +2.1] | -0.1% [-0.9, +0.6] | -0.2 / +1.1 / +0.1 | -3.4 / -3.7 / -2.5 | +0.08 |
| RpPos vs R | +13.2% [+7.1, +20.8] | +6.4% [+4.0, +8.9] | +5.3% [+3.0, +7.8] | -3.8% [-4.9, -2.8] | -2.1% [-3.6, -0.6] | -1.4% [-2.0, -0.9] | +0.9% [-0.2, +2.1] | +13.4 / +13.1 / +13.2 | -4.2 / -4.5 / -2.7 | +0.04 |

## v4B screening rule vs R (≥5% far grasp, CI>0, ≤2% handover degradation)

| arm | far grasp | handover degradation | PASS |
|---|---|---|---|
| RpBoth | +9.9% [+6.2, +13.6] | 1.0% [-0.6, +2.5] | True |
| RpPos | +13.2% [+6.9, +20.8] | 3.8% [+2.8, +4.9] | False |
| RpBoth_NoSynthRGB | +7.2% [+3.5, +11.0] | 1.9% [+0.8, +3.1] | True |
| RpBoth_rerun | +9.9% [+6.3, +13.7] | 1.0% [-0.6, +2.5] | True |
| RpPos_MatchedRandMask | +12.1% [+6.7, +18.6] | 7.0% [+5.6, +8.5] | False |
| RpPos_PhaseMask | +13.5% [+8.3, +19.5] | 7.1% [+5.8, +8.6] | False |
| RpPos_rerun | +13.2% [+6.9, +21.1] | 3.8% [+2.8, +4.8] | False |

## Inference-time image conditions (same condition for both models)

| contrast | condition | far grasp | grasp (40) | handover (40) |
|---|---|---|---|---|
| RpBoth vs RpBoth_NoSynthRGB | original | +2.8% [-1.2, +7.1] | +3.8% [+1.7, +6.0] | +0.9% [-0.2, +2.1] |
| RpBoth vs RpBoth_NoSynthRGB | blank | -1.4% [-6.9, +3.1] | +0.1% [-2.9, +3.0] | -3.3% [-4.5, -2.3] |
| RpBoth vs RpBoth_NoSynthRGB | wrong_episode | -15.8% [-22.2, -10.9] | -24.0% [-29.9, -18.7] | -0.5% [-1.6, +0.7] |
| RpBoth vs RpBoth_NoSynthRGB | object_only | -1.7% [-6.1, +2.7] | -2.8% [-5.9, +0.2] | -1.2% [-2.4, +0.0] |
| RpBoth vs RpBoth_NoSynthRGB | background | +5.2% [+0.9, +10.3] | +4.6% [+2.5, +6.8] | -0.2% [-1.4, +1.0] |
| RpBoth vs RpBoth_NoSynthRGB | gripper_only | +5.5% [+2.6, +8.9] | +5.6% [+4.0, +7.5] | +0.1% [-1.2, +1.3] |
| RpBoth_NoSynthRGB vs R | original | +7.2% [+3.5, +10.9] | +5.1% [+2.4, +7.6] | -1.9% [-3.1, -0.8] |
| RpBoth_NoSynthRGB vs R | blank | +13.1% [+8.7, +18.2] | +10.4% [+7.5, +13.5] | +4.2% [+2.7, +5.6] |
| RpBoth_NoSynthRGB vs R | wrong_episode | +15.0% [+12.6, +17.5] | +17.9% [+15.1, +20.8] | +1.7% [+0.1, +3.2] |
| RpBoth_NoSynthRGB vs R | object_only | +10.9% [+8.0, +14.0] | +9.8% [+6.8, +12.7] | +0.4% [-0.9, +1.6] |
| RpBoth_NoSynthRGB vs R | background | +7.2% [+4.4, +10.3] | +4.4% [+2.1, +6.7] | -1.2% [-2.6, +0.0] |
| RpBoth_NoSynthRGB vs R | gripper_only | +7.1% [+4.1, +10.3] | +4.6% [+2.2, +6.7] | -1.3% [-3.0, +0.3] |
| RpBoth vs R | original | +9.9% [+6.3, +13.7] | +8.7% [+6.8, +10.9] | -1.0% [-2.5, +0.6] |
| RpBoth vs R | blank | +11.8% [+7.6, +16.0] | +10.5% [+8.1, +13.0] | +1.0% [-0.6, +2.5] |
| RpBoth vs R | wrong_episode | +1.6% [-1.6, +4.2] | -1.8% [-4.2, +0.4] | +1.3% [-0.3, +2.8] |
| RpBoth vs R | object_only | +9.4% [+5.6, +12.9] | +7.3% [+5.1, +9.4] | -0.7% [-2.5, +1.0] |
| RpBoth vs R | background | +12.0% [+8.8, +16.0] | +8.8% [+6.6, +11.1] | -1.4% [-3.1, +0.2] |
| RpBoth vs R | gripper_only | +12.2% [+8.3, +16.5] | +9.9% [+7.8, +12.2] | -1.2% [-2.9, +0.5] |
| RpPos_PhaseMask vs RpPos | original | +0.3% [-2.0, +2.0] | +0.8% [-0.3, +1.6] | -3.2% [-4.9, -1.5] |
| RpPos_PhaseMask vs RpPos | blank | -0.1% [-2.9, +2.2] | +3.5% [+1.8, +5.1] | -2.3% [-3.8, -0.7] |
| RpPos_PhaseMask vs RpPos | wrong_episode | -0.1% [-0.9, +0.6] | +0.9% [+0.2, +1.6] | -4.0% [-5.6, -2.4] |
| RpPos_PhaseMask vs RpPos | object_only | +0.8% [-0.9, +2.1] | +2.6% [+1.4, +3.7] | -2.7% [-4.7, -0.8] |
| RpPos_PhaseMask vs RpPos | background | +0.1% [-2.7, +1.9] | +1.1% [-0.0, +2.0] | -2.9% [-4.7, -1.2] |
| RpPos_PhaseMask vs RpPos | gripper_only | +0.2% [-2.0, +1.8] | +0.8% [-0.2, +1.7] | -3.4% [-5.3, -1.5] |
| RpPos_PhaseMask vs RpPos_MatchedRandMask | original | +1.7% [+0.9, +2.4] | +1.7% [+1.3, +2.1] | -0.1% [-0.2, -0.0] |
| RpPos_PhaseMask vs RpPos_MatchedRandMask | blank | +2.2% [+1.8, +2.5] | +2.9% [+2.4, +3.3] | +0.3% [+0.1, +0.5] |
| RpPos_PhaseMask vs RpPos_MatchedRandMask | wrong_episode | +1.8% [+1.4, +2.1] | +2.3% [+1.9, +2.7] | -0.3% [-0.5, -0.1] |
| RpPos_PhaseMask vs RpPos_MatchedRandMask | object_only | +2.2% [+1.8, +2.6] | +2.7% [+2.3, +3.1] | +0.2% [+0.0, +0.4] |
| RpPos_PhaseMask vs RpPos_MatchedRandMask | background | +1.7% [+0.9, +2.3] | +1.7% [+1.3, +2.1] | +0.0% [-0.1, +0.1] |
| RpPos_PhaseMask vs RpPos_MatchedRandMask | gripper_only | +1.6% [+0.7, +2.3] | +1.7% [+1.3, +2.1] | +0.4% [+0.3, +0.6] |

## Decisions

```
{
 "_doc": "Amendment A1 decision contract. 'not_resolved' / NOT_ESTABLISHED never means equivalence or no effect.",
 "H1": {
  "question": "generated synthetic RGB vs SmolVLA missing-camera treatment on synthetic rows (real RGB unchanged)",
  "generated_rgb_incremental_far_grasp": 0.028172994028176412,
  "ci95": [
   -0.012014152599442662,
   0.07297452887009852
  ],
  "per_seed": [
   0.010445332565732435,
   0.032480585342603896,
   0.04101999304727755
  ],
  "replication_noise_far_grasp": 0.0,
  "exceeds_replication_noise": true,
  "sensitivity_rerun_vs_NoSynthRGB_far_grasp": [
   0.028172994028176412,
   [
    -0.012877234097927265,
    0.07111116484274009
   ]
  ],
  "verdict": "INCREMENTAL_VALUE_NOT_ESTABLISHED",
  "not_established_reasons": [
   "|effect| below the 3% threshold",
   "95% CI includes 0"
  ],
  "no_rgb_arm_vs_R_far_grasp": [
   0.07241352054596273,
   [
    0.03444087430265881,
    0.10929882174908091
   ]
  ],
  "synthetic_state_action_without_synthetic_rgb_passes_far_grasp_screen_part": true,
  "retained_gain_fraction_far_grasp": [
   0.7348164317702089,
   [
    0.38005351495922307,
    1.1431059656799232
   ]
  ]
 },
 "H2": {
  "question": "does phase-placed synthetic target supervision beat the per-row matched random control and reduce handover harm without destroying the far-grasp gain? (estimand: target placement within matched active contexts)",
  "A_handover_improvement_vs_RpPos": {
   "rel_change": -0.03213138375450275,
   "ci95": [
    -0.04921226030518807,
    -0.015569235957972216
   ],
   "per_seed": [
    -0.03401344484125768,
    -0.03718573839756867,
    -0.025197195032910846
   ],
   "status": "worsened",
   "sensitivity_vs_RpPos_rerun": [
    -0.03213138375450275,
    [
     -0.048382772390032996,
     -0.015928578228464826
    ]
   ],
   "matched_random_vs_RpPos": {
    "rel_change": -0.0308731315524291,
    "ci95": [
     -0.04798753510365887,
     -0.01365019460570256
    ],
    "status": "worsened"
   }
  },
  "B_residual_handover_degradation_vs_R": {
   "degradation": 0.07145469858953724,
   "ci95": [
    0.05761629151499381,
    0.08611439230737286
   ],
   "guardrail": 0.02,
   "within_guardrail_point_estimate": false
  },
  "C_far_grasp_preservation": {
   "vs_R": [
    0.13537646147968163,
    [
     0.08318000321980763,
     0.19536381440360212
    ]
   ],
   "P2_preserved_vs_R": true,
   "vs_RpPos": [
    0.003395542726238321,
    [
     -0.019753222924442704,
     0.019922405710833726
    ]
   ],
   "vs_RpPos_status": "not_resolved",
   "far_grasp_reduced_vs_RpPos": false
  },
  "D_phase_specific_increment_vs_matched_random": {
   "rel_change": -0.001220569402346161,
   "ci95": [
    -0.00214404775767571,
    -0.00031895728725368506
   ],
   "per_seed": [
    -0.0017759554052578785,
    -0.0009663303760083508,
    -0.0009202619910659848
   ],
   "status": "worsened"
  },
  "E_statistical_resolution": {
   "replication_noise_handover": 0.0,
   "replication_noise_far_grasp": 0.0,
   "A_exceeds_noise": true,
   "D_exceeds_noise": true,
   "A_ci_width": 0.033643024347215854,
   "D_ci_width": 0.0018250904704220248
  },
  "original_P1_reduction_and_guardrail": false,
  "original_P2_far_grasp_preserved": true,
  "original_P3_beats_control": false,
  "mechanism_label": "PHASE_MASK_INCREASES_HANDOVER_ERROR",
  "Q2_answer": "NOT_SUPPORTED_BY_THIS_SCREEN"
 }
}
```
