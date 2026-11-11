# R1 protocol: how far can the object move before body-camera RGB stops being valid?

**Frozen** 2026-11-12, before any R1 rendering, on branch `research/sfagen-few-source`. The only data looked at before
freezing: the can table positions of sources 0–9 (to choose the bins) and the published `xbench` numbers.

## Question
F1W moves the Pringles can up to 15.8 cm from where it was recorded, and its previews show artifacts. The safe shift
radius **r\*** decides between two options:
- (A) wide shifts from 1–2 sources, after fixing the artifacts;
- (B) shifts anchored near the recorded positions, with more spread-out sources.

## Ground truth: cross-episode pose transfer
- **Data:** Pringles sources 0–9 (`ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`). All are real recordings of
  the same task in one session, with different can positions, and their reconstructions are cached in
  `configs/work/openarm_pringles_handover/cache/`.
- **Pairs:** all 90 ordered pairs (A, B), A ≠ B.
- **Render:** with the sfagen renderer of this branch (`Renderer(recA, cfg=ctx.cfg["render"])`), re-render A's video frame
  `jA` with B's joints, B's grippers and B's can pose at B's frame `tB`:
  `render(jA, q=recB.gv[·][tB], g=recB.grip_v[·][tB], objs=[recB.obj_pose(0, tB + recB.lag)])`.
- **Compare** with B's real frame `recB.F[tB]`. This is exactly what the generator does when it moves the can to a new
  place and the arm follows, except that the joints are a real recording instead of IK, so only **rendering** error is
  measured.
- **Phases, aligned by state time** (video frame = round(state time − lag)), using frame offsets −2, 0, +2 around each:

  | Phase | State time |
  |---|---|
  | start | t = 5 (so the offsets give frames 3–7) |
  | approach | t = round(0.6 · t_grasp) |
  | grasp | t = t_grasp (the can's `grasp_frame`, the right close) |
  | lift | t = t_grasp + round(0.5 · (t_left_close − t_grasp)) |

  t_left_close is the first left close after t_grasp. That gives 12 frame pairs per ordered pair.
- **Copy baseline** (floor): A's frame `jA` unchanged.
- **Same-episode reference** (ceiling): for each source e, render frame `j` with its own state at `j + 8` (the xbench
  k = 8 setting, `exclude = (j+8±5)`), at the same 12 phase frames.

## Scoring
Use the xbench region: `fg(x)` = |frame − temporal median of its episode (`F[::3]`)| > 30, opened 3×3, and
m = dilate(fg_A(jA) ∪ fg_B(tB), 11×11).

| Score | Definition |
|---|---|
| Primary | PSNR over m |
| Secondary | masked SSIM (`sfagen.bench.ssim_masked`) over m |
| Sub-regions (PSNR; skipped below 200 px) | can = B's can silhouette at tB (dilated 5); robot = B's projected arms at tB (dilated 5), minus can; revealed = A's can silhouette at jA, minus can and robot; other = the rest of m |

**Pair value:** the mean over its 12 frames.

**Displacement:** the xy distance between A's and B's can table positions.
- Bins (cm): [0,3) [3,5) [5,7) [7,9) [9,11) [11,15) [15,21]. Ordered pairs per bin: 10, 10, 10, 20, 16, 10, 14.
- Secondary axis: the mean over frames of max |q_A − q_B| (rad, arm joints).

**Uncertainty:** episode-cluster bootstrap with 2000 resamples. Each resample draws 10 episodes with replacement and
keeps every ordered pair with both ends drawn, weighted by multiplicity. Seed 0.

## Decision rule (frozen)
- Bin statistic: the **median pair PSNR** of the render. Δ_b = median_b − median_[0,3).
- A bin **passes** if both hold:
  - Δ_b ≥ −0.5 dB (point estimate);
  - the paired (render − copy) PSNR in that bin has a 95% CI lower bound > 0.
- **r\*** = the upper edge of the last bin in an unbroken run of passing bins starting from [3,5). If [3,5) fails, r\* = 3 cm.

| r\* | Next |
|---|---|
| ≥ 11 cm | Keep F1W's wide shifts (option A) and fix the sub-region with the largest drop first |
| ≤ 7 cm | Option B: anchor shifts within r\* of recorded positions and select spread-out sources |
| 9 cm (in between) | F1W planner capped at r\* |

The CI of Δ is reported but does not enter the rule. The reference bin has only 5 unordered pairs.

## Also reported (no decision attached)
- Per-phase and per-sub-region curves, render − copy, SSIM.
- Contact sheets per bin (real B | render | copy | region).
- **Wrist validity:** the fraction of frames with `wrist_view_valid = 0` against |shift|, from the published Pringles
  dataset (96 episodes, ≤ 6 cm) and the F1W preview export (3 episodes). The wrist cameras are not re-rendered.

## Limits stated in advance
- Sources 0–9 are reused development data; this validates the renderer, it is not a held-out claim.
- B's can pose is a reconstruction, not measured truth.
- One session, so the lighting confound is small.
- PSNR is a proxy; VLA usefulness needs a separately authorized training screen.
