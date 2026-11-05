| arm | phase-balanced error (mean ± sd, 3 seeds) | vs R | skill vs persistence | gripper precision / recall (steps 0-10) | transition timing error (frames) |
|---|---|---|---|---|---|
| R | 0.3341 ± 0.0029 | +0.0% | 0.528 | 0.932 / 0.943 | 3.09 |
| R+G+ | 0.3285 ± 0.0007 | +1.7% | 0.536 | 0.935 / 0.940 | 3.03 |
| R+C | 0.3269 ± 0.0009 | +2.2% | 0.536 | 0.937 / 0.943 | 2.91 |

| arm | approach | carry | handover | retreat |
|---|---|---|---|---|
| R | 0.3533 | 0.3538 | 0.3168 | 0.3127 |
| R+G+ | 0.3406 | 0.3537 | 0.3026 | 0.3172 |
| R+C | 0.3395 | 0.3589 | 0.2995 | 0.3095 |

persistence baseline (hold current state) error: 0.7326; decision: {"RpGp": {"rel_improvement_vs_R": 0.016860704649199754, "timing_delta_frames": -0.0555555555555558, "augmentation_helps": false}, "RpC": {"rel_improvement_vs_R": 0.02181590202317163, "timing_delta_frames": -0.18333333333333357, "augmentation_helps": false}, "contact_helps": false}
