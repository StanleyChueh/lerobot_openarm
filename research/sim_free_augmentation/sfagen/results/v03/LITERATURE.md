# v03 literature check (2026-11-13, Claude)

Question: should synthetic data cover unseen object positions even when its images are imperfect? How do recent
papers handle artifacts and evaluate, and what is acceptable at a top venue?

| Paper | Finding | Use here |
|---|---|---|
| [Sim-and-Real Co-Training](https://arxiv.org/abs/2503.24361) (Maddukuri et al., RSS 2025) | Sim data with clear visual discrepancies raised real success from 45.3 % to 76.8 %; at unseen object positions, 11 → 28 % (Panda) and 43 → 100 % (GR-1). Camera misalignment hurt: 95 → 70 % (GR-1 CupPnP), 67 → 56 % (Panda). Best mixing ratio was task-dependent, with up to 99 % sim. | Imperfect pixels are tolerated, wrong geometry is not, so coverage is worth some distortion. Motivates the strict/loose ablation and a mixing-ratio knob. |
| [R2RDreamer](https://arxiv.org/abs/2606.17040) (2026) | 1 source demo + 3D-aware RGB augmentation (point-cloud control video + video diffusion) on held-out locations. DP: 21.9–40.6 % vs 15 human demos 15.6–31.3 %. 30 demos remain better. Artifacts are admitted; ablations are qualitative only. | Closest RGB competitor. It has **no quantitative visual-quality vs policy-success study**, which is the gap the v03 strict/loose design addresses. |
| [Real2Edit2Real](https://arxiv.org/abs/2512.19402) (CVPR 2026) | 1–5 source demos → generated multi-view demos, including both wrists, via a depth-controlled video model. Matches or beats 50 real demos. | Reviewers will expect wrist views handled; copied vs rendered wrist must be an ablation. |
| [DemoGen](https://arxiv.org/abs/2502.16932) (RSS 2025) | MimicGen/TAMP-style action adaptation + point-cloud editing from 1 demo. States the single-view visual mismatch as a limitation. | Accepted practice: the mismatch is disclosed, not eliminated. |
| [LIBERO-PRO](https://arxiv.org/abs/2510.03827), [LIBERO-Plus](https://arxiv.org/abs/2510.13626) (CVPR 2026) | VLAs that score high on LIBERO collapse under object-position perturbations. | A simulation benchmark for the position-coverage claim, with oracle renders available (Part B scoping). |
| [GenAug](https://arxiv.org/abs/2302.06671), [ROSIE](https://arxiv.org/abs/2302.11550) | Generated or inpainted visual augmentation, with imperfect frames, improves real generalization. | Further support that imperfect augmentation helps. |

## Implications for v03
- Cover the arena evenly and cap only heavy distortion with a measured, pre-registered score. Compare strict vs loose
  by policy success; that comparison is the novel axis.
- Discrete variation (battery slots, object types, grasps, orientations) must be in the real demos. Our edits are
  translations.
- Evaluate on a uniform position grid that includes corners, with ≥ 5 trials per position, against real-only and a
  "more real demos" baseline.
