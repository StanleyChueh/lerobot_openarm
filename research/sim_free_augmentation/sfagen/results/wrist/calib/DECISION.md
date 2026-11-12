# B2 mount refinement: decision

**Adopted.** This corrects commit `35ec7df`, whose message swapped the two held-out runs.

| Mount | Dev T1 objective ('plane' depth) | Held-out T1 render PSNR (RAFT), right / left | Held-out T1 render − copy, right / left |
|---|---|---|---|
| Initial (v3_2) | 15.99 / 18.90 dB | 15.49 / 19.02 dB | +0.59 [0.12, 1.03] / +0.43 [0.20, 0.71] dB |
| Refined (`refined_spec.yaml`) | 16.95 / 20.04 dB (+0.96 / +1.14) | 15.83 / 19.51 dB (+0.34 / +0.49) | +1.01 [0.67, 1.31] / +0.82 [0.74, 0.90] dB |

**Rule (PROTOCOL.md):** dev gain ≥ 0.3 dB **and** held-out change ≥ −0.1 dB. Both cameras pass.

**Offsets from the v3_2 mount** (lag in frames, translation in m, rotation in deg):
- left: −0.25 / (−0.0125, 0.0125, 0.0125) / (1.5, 3.0, −3.0);
- right: +0.5 / (0, −0.025, 0.025) / (1.5, −3.0, 1.5).

Several rotation offsets sit at the ±3° search bound, so the true optimum may lie outside the frozen box.

The registered gate (`final_*`) uses the refined mounts. The `supp_initial_mount_pringles_heldout` run (initial mounts)
is kept as supplementary evidence only.
