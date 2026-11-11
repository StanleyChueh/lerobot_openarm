# R1 + R2: how far can the object move, and how few source demos are enough? (Pringles)

**Date:** 2026-11-12. **Branch:** `research/sfagen-few-source`. **Status:** executed, pending independent review.
Offline only: no VLA training, no robot, no Hub upload, no physics.

## R1: shift limit of the body-camera renderer

- **Protocol:** `r1_relocation/PROTOCOL.md`, frozen at `327d38f` (sha256 `06baf504…`) before any rendering.
- **Code:** `tools/xfer_bench.py` at `72fe6c5`.
- **Run:** `r1_relocation/run/` (`frames.jsonl`, `summary.json`, `run_meta.json`, `sheets/`); 1,080 cross-episode
  frame pairs + 120 same-episode frames.
- **Data:** `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`, sources 0–9; cached reconstructions; one RTX 4090.

**Method:** re-render source A's frame with episode B's *real* joints and can pose, and compare with B's real frame. All 90
ordered pairs at 4 aligned phases × 3 frames. This measures rendering error only; the trajectory is real.

| Can shift (cm) | Ordered pairs | Median render PSNR | Δ vs [0,3) [95% CI] | Render − copy [95% CI] | Rule |
|---|---:|---:|---|---|---|
| 0–3 | 10 | 16.43 dB | reference | +3.94 [2.95, 4.49] | – |
| 3–5 | 10 | 16.28 | −0.16 [−1.90, 0.74] | +4.13 [3.92, 4.27] | pass |
| 5–7 | 10 | 16.28 | −0.16 [−1.39, 1.02] | +4.29 [3.90, 4.59] | pass |
| 7–9 | 20 | 15.97 | −0.47 [−1.91, 0.56] | +4.28 [3.67, 5.16] | pass |
| 9–11 | 16 | 16.01 | −0.42 [−1.86, 0.52] | +4.81 [4.04, 5.85] | pass |
| 11–15 | 10 | 15.61 | −0.82 [−2.39, −0.08] | +4.04 [3.54, 5.07] | fail |
| 15–21 | 14 | 15.34 | −1.09 [−2.24, −0.35] | +4.18 [3.45, 4.65] | fail |

**Registered result: r\* = 11 cm, which gives decision A** (keep wide shifts up to r\*, fix the worst sub-region).

**What the numbers say beyond the rule (secondary, not part of the decision):**
- **The can itself is the weak part, and it degrades with distance.** Can-region median PSNR is 14.9 dB at 0–3 cm,
  13.6 at 5–7, 13.3 at 7–11, 12.0 at 11–15, and 11.5 at 15–21 (−3.4 dB). The robot region drops about 1 dB as soon as
  the arm is re-posed, then stays flat (16.9–17.5 dB). The revealed table and the background do not degrade.
- **A cross-episode gap exists at every distance.** The same-episode re-render scores 18.98 dB (can region 25.1 dB). The
  closest cross-episode bin scores 16.43 dB (can 14.9 dB). Most of the error comes from re-posing the arm to a different
  trajectory and from a different can pose, not from the distance alone. Real generation changes only the grasping arm,
  smoothly, so R1 is a harsher test than generation.
- **By phase:** start −1.4 dB, approach −2.0 dB and grasp −1.3 dB at 15–21 cm; lift is flat.
- **Correlations:** PSNR with displacement −0.56; with joint change −0.61. Displacement and joint change correlate 0.70.
- **SSIM** stays 0.64 for the render against 0.45 for copy at every distance.
- **Contact sheets** (`r1_relocation/run/sheets/`): the can lands in the right place. Re-posed arms show fragmented
  surfaces, dark holes and a ghost gripper outline. Beyond 11 cm the can lid edge tears.

**Wrist cameras** (`r1_relocation/wrist_validity.json`): the wrist views are not re-rendered, so edited frames are flagged
invalid.

| Episodes | Shift | Wrist frames invalid |
|---|---|---|
| Published Pringles, object-position | ≤ 6 cm | 49–52 % |
| Published Pringles, hand-over | ≤ 6 cm | 60 % |
| F1W previews | > 6.5 cm | 80 % (2 episodes) |

**A three-camera VLA sees no valid synthetic wrist view for half or more of every synthetic episode.** This holds at any
shift size.

## R2: fewest source demos at the 11 cm cap (coverage proxy)

- **Code:** `tools_workspace_budget.py` with the new `--max-shift-cm 11`.
- **Input:** the existing F1W candidates (sources 0–4, 48 targets each).
- **Result file:** `r2_fewsource/source_budget_cap11.json`.
- **Method:** each subset is re-checked against its own speed, acceleration and table envelope. Coverage counts the 40
  other real recordings (pool `2c1bc860`, episodes 10–49) with a training can position within 2 cm (FK at the
  right-gripper close).

**Nested sources 0…N−1** (a fixed rule, not chosen on the evaluation pool):

| N | Real demos only | Old dense grid, label-checked | F1W capped at 11 cm | F1W uncapped |
|---|---:|---:|---:|---:|
| 1 | 2 | 4 | 10 | 11 |
| 2 | 3 | 10 | 16 | 18 |
| 3 | 3 | 12 | 21 | 24 |
| 4 | 6 | 22 | 28 | 32 |
| 5 | 7 | 22 | 28 | 32 |

**Spread of all subsets of sources 0–4, F1W capped** (post hoc: chosen with the evaluation pool, so it is an upper
bound and not a claim):

| N | Best subset | Worst subset |
|---|---|---|
| 1 | 19 (source 3) | 1 |
| 2 | 25 (sources 0, 3) | 3 |
| 3 | 27 | 7 |

For reference, the published 10-source sfagen dataset covers 24/40.

**Reading:**
- **The 11 cm cap costs little.** At N = 4–5 it gives 28 vs 32/40; at N = 2, 16 vs 18.
- **Which sources you pick matters more than how many.** With two well-placed sources, capped F1W reaches the published
  10-source coverage on this proxy. But the best pair was found on the evaluation pool. A pre-registered, source-only
  selection rule must be tested before claiming this.

## Limits
- Sources 0–9 and pool 10–49 are reused development data.
- Can poses are reconstructions, and the coverage positions are FK proxies.
- The reference bin has only 5 unordered pairs, so the Δ CIs are wide.
- PSNR and coverage are proxies. Whether 1–2 sources train a useful VLA needs the separately authorized SmolVLA screen.

## Next decisions
1. **Fix the can appearance under re-posing first** (the worst sub-region). Candidate causes: the lid/edge tearing and
   the view-dependent appearance of the can at a new image position.
2. **Freeze a source-only selection rule** (e.g. k-center on recorded can positions over the full 10 sources) and rerun
   R2 with F1W candidates generated for all 10 sources.
3. **Decide what to do about the wrist cameras:** re-render them, or train with masked wrist views. Today they are
   invalid for at least half of every synthetic episode.
4. Then run the authorized SmolVLA screen: N ∈ {1, 2, 4, 10} at the 11 cm cap, 3 paired seeds.

## Reproduce

```bash
cd research/sim_free_augmentation/sfagen
ln -s ~/Stanley_ws/sfa_hf_export/sfagen/configs/work configs/work; ln -s ~/Stanley_ws/sfa_hf_export/sfagen/logs logs
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 $HOME/Stanley_ws/lerobot_openarm/.venv/bin/python"
$PY -B -m unittest discover -s tests                                                  # 32 CPU tests
(cd tools && $PY -B xfer_bench.py run cuda:0 ../results/r1_relocation/run)            # R1 (GPU; rendering took 10 min after loading)
(cd tools && $PY -B xfer_bench.py summarize ../results/r1_relocation/run)             # R1 statistics only (CPU)
(cd tools && $PY -B xfer_bench.py wrist ../results/r1_relocation)                     # wrist validity (CPU)
$PY -B tools_workspace_budget.py configs/openarm_pringles_handover_workspace.yaml \
   --pool-data ~/.cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d/data \
   --exclude 0,1,2,3,4,5,6,7,8,9 --pool-revision 2c1bc860347c44fc0fbae7567469d36ee49baa8d \
   --comparison-work configs/work/openarm_pringles_handover_dense --max-shift-cm 11 \
   --out results/r2_fewsource/source_budget_cap11.json                                # R2 (CPU)
```
