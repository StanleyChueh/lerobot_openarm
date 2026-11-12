# A: can-appearance fix, results (Pringles body camera)

- **Date:** 2026-11-12.
- **Protocol:** `PROTOCOL.md` (frozen `945141b`, amendment 1 `f4539ce`, both before any held-out render).
- **Code:** `tools/canfix_bench.py`; renderer keys in `sfagen/render.py`, all default off.

**Registered outcome: not adopted.** The new renderer keys stay off, and the datasets use the published renderer.

## What was tried (dev, Pringles episodes 0–4)
| Candidate (config key) | Idea | Dev cross: can / whole image | Dev lift: can / whole image |
|---|---|---|---|
| `object_erode 3` + `object_texture` | fill the whole target silhouette; a per-point colour atlas built from the episode | +0.04 / +0.00 dB | +0.08 / +0.06 dB |
| `object_mask_grow 60` | grow the object mask over the connected plate-difference region (removes the ghost lid) | 0.00 / −0.03 dB | +0.01 / +0.02 dB |
| all three (chosen) | | +0.04 / −0.03 dB | +0.09 / +0.08 dB |

**Diagnosis (dev):**
- 86–95 % of can pixels are forward-warped real can pixels, and it is their quality that drops with distance.
- Edge filling and the atlas reach only the remaining 5–14 %.
- The best 2-D shift of the render raises the can-region PSNR by +2.4 dB. The median shift is 9–16 px (about 1–2 cm),
  **even at 0–3 cm** and even when the can moves rigidly with the hand (lift test, 14–16 px).

**So the dominant error is the object-pose reconstruction**, with can-silhouette IoU 0.57–0.80 across the 10 episodes
and the can moving inside the gripper at the grasp. It is not the renderer.

## Held-out (Pringles episodes 5–9, rendered once)
| Test | Can region (median pair gain) | Whole image (mean gain) | Identity |
|---|---|---|---|
| **cross-episode (registered)** | **−0.16 dB [−0.36, −0.01]** | +0.01 dB [−0.08, +0.17] | −0.02 dB |
| same-can lift (amendment 1) | **+0.53 dB [+0.37, +0.89]** | +0.17 dB [+0.11, +0.23] | |

Gains are candidate − baseline on identical frames; brackets are 95 % CIs from an episode-cluster bootstrap.

**Rule:** the can-region gain must be ≥ +0.5 dB with a CI above 0 on the cross test. That fails, so the change is not
adopted.

**The secondary test disagrees:** when the same can is moved (no cross-episode label or pose confound), the change is
better by 0.5 dB on the can and 0.2 dB on the whole image. I report both. The registered decision stands.

## Consequences
- **Datasets:** the v01 datasets keep the published body renderer.
- **Next fix: object-pose reconstruction**, not rendering. Candidate fixes:
  - a SAM2 object mask instead of plate difference;
  - more pose degrees of freedom in the grasp-constrained fit;
  - re-fitting the held-object pose after the grasp.
- **Literature:**
  - RoboSplat and Real2Edit2Real avoid this by using dense object models (3DGS) or a generative video model.
  - Our proxy-cylinder pose is the limit.

## Reproduce
```bash
cd research/sim_free_augmentation/sfagen/tools
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 $HOME/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
C='{"object_mask_grow":60,"object_erode":3,"object_texture":true}'
$PY canfix_bench.py run cuda:0 ../results/canfix/dev_base dev '{}'          # about 4 min each on an RTX 4090
$PY canfix_bench.py run cuda:1 ../results/canfix/dev_g60_e3_tex dev "$C"
$PY canfix_bench.py compare ../results/canfix/dev_base ../results/canfix/dev_g60_e3_tex
$PY canfix_bench.py run cuda:0 ../results/canfix/heldout_base heldout '{}'
$PY canfix_bench.py run cuda:1 ../results/canfix/heldout_g60_e3_tex heldout "$C"
$PY canfix_bench.py compare ../results/canfix/heldout_base ../results/canfix/heldout_g60_e3_tex --rule
```
