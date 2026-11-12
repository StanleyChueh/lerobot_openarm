# C: few-source F1W generation (Pringles, object shift ≤ 11 cm)

- **Date:** 2026-11-12.
- **Selection rule:** frozen in `SELECTION.md` (`4b2e0eb`) before any coverage number.
- **Config:** `configs/openarm_pringles_fewsrc.yaml` (10 demos, `max_shift_m: 0.11`).

## Generation (10 demos, envelope of all 10 demos)
| Demo | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Passing / tried targets | 33/48 | 39/48 | 14/48 | 46/48 | 39/48 | 22/38 | 34/48 | 8/48 | 13/48 | 21/48 |

## Coverage of 40 unseen real recordings (pool `2c1bc860`, episodes 10–49; can within 2 cm; FK proxy)
Source: `source_budget10.json`. Chart: `../figs/coverage_vs_demos.png`.

| Demos | Frozen rule (subset) | Post-hoc best subset | Median subset | Real demos only | Old dense grid (label-checked) |
|---|---|---|---|---|---|
| 1 | **16** ({3}) | 16 | 4 | 3 | 10 |
| 2 | **24** ({3, 8}) | 26 | 8 | 3 | 12 |
| 4 | **27** ({1, 3, 5, 8}) | 30 | 23 | 5 | 19 |
| 10 | **31** | 31 | 31 | 9 | 27 |

**Reading:**
- The source-only rule is within 0–3 recordings of the post-hoc best.
- Two rule-chosen demos match the published 10-demo v00 dataset (24/40).
- Random pairs do much worse (median 8), so **which demos you record matters more than how many**.
- This is a coverage proxy on reused development recordings. It is not policy evidence.

## Candidate datasets (`datasets.json`)
- Each N keeps **every** variant that passes the speed, acceleration and table envelope of **its own N demos**.
- A first selection capped each demo at 16 spread variants as a compute guard (`datasets_cap16.json`). That cap cut
  N=2 dataset coverage from 24 to 19 of 40, so it was removed before any dataset was published.

| N | Demos | Episodes (per demo) | Frames | Dataset coverage (of 40) |
|---|---|---|---|---|
| 1 | {3} | 25 | 7,328 | 16 |
| 2 | {3, 8} | 51 (43 + 8) | 15,934 | 24 |
| 4 | {1, 3, 5, 8} | 111 (39 + 46 + 13 + 13) | 32,668 | 27 |
| 10 | all | 269 | 84,587 | 31 |

## Reproduce
```bash
cd research/sim_free_augmentation/sfagen
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 $HOME/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
(cd tools && $PY select_sources.py configs/openarm_pringles_fewsrc.yaml ../results/fewsrc/selection.json)   # C1 (CPU)
$PY -m sfagen generate configs/openarm_pringles_fewsrc.yaml --episodes 0,1,2,3,4      # about 20 s per demo
$PY -m sfagen generate configs/openarm_pringles_fewsrc.yaml --episodes 5,6,7,8,9
$PY tools_workspace_budget.py configs/openarm_pringles_fewsrc.yaml \
   --pool-data ~/.cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d/data \
   --exclude 0,1,2,3,4,5,6,7,8,9 --pool-revision 2c1bc860347c44fc0fbae7567469d36ee49baa8d \
   --comparison-work configs/work/openarm_pringles_handover_dense --out results/fewsrc/source_budget10.json   # C3 (CPU)
$PY tools/fewsrc_build.py select                                                          # C4 selection (CPU)
```
