# B: wrist-camera rendering, results (Pringles + battery)

- **Date:** 2026-11-12.
- **Protocol:** `PROTOCOL.md` (frozen `0a43e74` before any wrist render).
- **Renderer:** `sfagen/wrist.py`, the moving-camera MGW. It is task-agnostic: URDF, joints, the body-camera object
  reconstruction and the source wrist video only.
- **Bench:** `tools/wrist_bench.py`.

**Registered outcome:**

| Task | Right wrist | Left wrist |
|---|---|---|
| Pringles (held-out) | FAIL | FAIL |
| Battery (all episodes, never used for design) | **PASS** | FAIL |

**Consequence:**
- The primary v01 Pringles datasets keep copied wrist frames with `wrist_view_valid`, as in v00.
- A rendered-wrist **ablation** variant is published separately, labelled as having failed the gate. Only policy
  training can say whether the rendered views help.

## Design choices (dev, Pringles 0–4, T1 render − copy, 95 % CI)
| Background depth | Right wrist | Left wrist |
|---|---|---|
| **RAFT two-view triangulation (chosen)** | **+0.59 [0.43, 0.77] dB** | **+1.25 [0.89, 1.54] dB** |
| table plane, else far plane | +0.56 [0.36, 0.74] | +1.03 [0.73, 1.36] |
| Depth Anything V2 metric, scale-aligned | −0.11 [−0.18, −0.02] | −0.81 [−1.36, −0.27] |

**Other design steps (dev only):**
- **Pixels rigid with the camera.** Fingers and a held or just-grasped can fill the view at close range, and the
  projected geometry misses them. Before this step they were torn.
- **The fix is a two-hypothesis colour test.** In frames j ± 4, a pixel is explained either by staying put in the
  image or by the static-world motion. Pixels that stay put take the nearest own-hand or object motion. It uses no
  colour rules.
- **Mount refinement (B2):** see `calib/DECISION.md`.
  - Dev gain: +0.96 / +1.14 dB.
  - Held-out render PSNR: +0.34 / +0.49 dB. **Adopted.**
  - Several rotation offsets sit at the ±3° bound.

## Registered gate (refined mounts, RAFT depth; render − copy PSNR, 95 % episode-cluster CI)
**T1** = same-episode novel view (real camera motion). **T2** = cross-episode transfer, binned by camera translation,
camera rotation ≤ 5°.

| Task / camera | T1 | T2 [0, 3) cm | T2 [3, 6) cm | T2 [6, 11) cm | Gate |
|---|---|---|---|---|---|
| Pringles held-out / right | **+1.01 [0.67, 1.31]** (298 fr) | +0.19 [−1.12, 1.30] (82 fr) | −4.92 [−6.49, −2.33] (10 fr) | – | FAIL |
| Pringles held-out / left | **+0.82 [0.74, 0.90]** (172 fr) | **+3.40 [2.01, 4.57]** (194 fr) | +1.27 [−0.54, 2.42] (8 fr) | – | FAIL |
| Battery / right | **+1.64 [1.35, 1.93]** (1130 fr) | **+1.60 [1.22, 1.98]** (816 fr) | **+2.12 [1.91, 2.64]** (10 fr) | – | **PASS** |
| Battery / left | **+1.47 [1.09, 1.80]** (874 fr) | +0.85 [−0.28, 1.98] (304 fr) | −3.14 [−7.02, −0.95] (56 fr) | −1.58 [−3.25, −0.11] (30 fr) | FAIL |

The supplementary run with the initial (v3_2) mounts on Pringles held-out also fails: T1 +0.59 / +0.43 dB
(`supp_initial_mount_pringles_heldout/`).

## Reading
- **Within an episode the renderer is always better than copying** (4 of 4 task–camera cells, CI > 0). Those camera
  moves include up to 22° of rotation.
- **Across episodes it is not reliable once the camera moves ≥ 3 cm.** Those frames see room regions the source frame
  never saw: 15–23 % of pixels are inpainted, and the other episode's arm and objects differ.
- **T2 cannot fully test the generation regime.** Real cross-episode pairs rarely share camera orientation. At ≤ 5°,
  only 66 of 240 Pringles right-wrist frames survive, nearly all at the episode start. The bins at 3 cm and above hold
  only 8–56 frames.
- **Weakness for the paper:** the wrist views are geometrically right (the room moves correctly, and the held can stays
  fixed in view), but disoccluded borders and the window region show fragments. See `heldout_t1_refined/sheets/`.
- **Literature:**
  - WristWorld and Real2Edit2Real use learned generators exactly for these disocclusions.
  - A diffusion inpainter restricted to the provenance-flagged holes is the natural next step.

## Reproduce
```bash
cd research/sim_free_augmentation/sfagen/tools
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 $HOME/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
for d in plane da2 raft; do $PY wrist_bench.py run cuda:0 ../results/wrist/dev_t1_$d pringles dev "{\"depth\":\"$d\"}" --t1; done   # depth choice
$PY wrist_calib.py cuda:1 ../results/wrist/calib dev                                                   # B2 (about 1 h)
$PY wrist_bench.py run cuda:0 ../results/wrist/heldout_t1_refined pringles heldout '{"depth":"raft"}' --t1 --spec results/wrist/calib/refined_spec.yaml
$PY wrist_bench.py run cuda:0 ../results/wrist/final_pringles_heldout pringles heldout '{"depth":"raft"}' --spec configs/_openarm_wrist_cameras_refined.yaml
$PY wrist_bench.py run cuda:1 ../results/wrist/final_battery battery all '{"depth":"raft"}' --spec configs/_openarm_wrist_cameras_refined.yaml   # about 1 h
$PY wrist_bench.py summarize ../results/wrist/final_battery
```
