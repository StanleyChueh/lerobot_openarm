# Targeted Codex re-audit request: v4B.1 Amendment A1

**Date:** 2026-11-06.

**Commits:**
* Code and protocol under review: `e4149556d77051bfb4b05a331b866e291b348d97`, branch `research/contact-aware-augmentation`. Not pushed.
* Previous Codex-audited commit: `1f80f2e`. Its report is `CODEX_SIM_FREE_V4B1_PRETRAIN_AUDIT.md` (local, untracked; not modified).

**Scope:** please check only the blockers identified in that audit. Do not restart the full project audit.

**State:** no full v4B.1 training has run: zero 1,000-update checkpoints, zero study predictions, no `results.json`. Launch waits for this re-audit.

## What changed (spec: `V4B1_EXPERIMENT_SPEC.md` → Amendment A1)

| blocker in the 1f80f2e audit | A1 change | where |
|---|---|---|
| H2 random control matched only pooled retention | `RpPos_RandMask` superseded (never run). New `RpPos_MatchedRandMask` retains, for each sampled synthetic row, exactly PhaseMask's K among the row's originally valid targets, uniformly and seeded (4208, seed, global index). K = 0 → 0 | `sfa4b1/interventions.py::synthetic_exclusion`, `matched_random_retained`, `valid_targets`; `configs/v4b1.json` |
| Matching must be tested if claimed | Exact equality over 3 seeds × 1,000 batches: retained count per row, active rows, batch denominators, source exposure, real-target weights, padding | `tests/test_v4b1_amendment.py`, `sfa4b1/schedule_masks.py`, `scripts/masks.py` → `outputs/matched_budget.json`, `outputs/masks/*.npz` |
| Narrower estimand | Placement within matched active contexts. Positions differ only in about 19 % of draws (the boundary-crossing ones). Whole-context removal is not identified | spec A1.3, A1.4 |
| Incomplete seeds or arms give verdicts | `StudyContractError` on any seed set ≠ {0,1,2}, a missing arm or a missing contrast. Smoke mode gives no decisions | `scripts/score_v4b1.py::check_names`, `check_complete`, `_need` |
| Frozen references not protected | Name whitelist; frozen directory refused; content hashes of frozen archives, models and normalisers; step = 1000 | `check_names`, `load_study`, `check_frozen_refs` |
| Evaluation contract unchecked | Item order, `eval_meta`, targets, six conditions, finite (1524, 50, 16) | `check_archive`, `check_eval_contract` |
| Update counts unchecked | Per-run step counter, consumed row indices = frozen schedule, applied masks = expected, normaliser, clean tree | `scripts/verify_runs.py`; scorer `check_new_runs` |
| Partial replication-noise guard | Applied to H1 useful, H1 harmful and every H2 component; rerun contrasts required | `decisions`, `_effect`, `config scoring.replication_noise` |
| Overconfident labels | H2 split into components A–E. Budget-explanation and no-reduction labels withdrawn. Q2 = YES only if A ∧ D ∧ B ∧ C | `decisions`; spec A1.6 |
| Launch hashes a list of hashes, not content | `content_checks()` hashes the 17 inputs, 3 base snapshots, 9 references and the mask manifests, and aborts on mismatch or a dirty v4b1 tree | `scripts/run_v4b1.py` |

## Results of my checks (to be reproduced independently)

* **Tests: 32/32 pass** (about 20 s, CPU): the 6 original tests, 7 A1 mask tests and 19 scorer tests. The 322,800 boundary entries are preserved.
* **`outputs/matched_budget.json`:**
  * matched = phase in 16,000/16,000 rows and 1,000/1,000 denominators per seed;
  * the old RandMask had equal denominators in only 5/8/5 batches;
  * the PhaseMask numbers equal those in Codex's table (124,653 / 119,265 / 122,074 kept).
* **GPU smoke** (20 steps, seed 0, both H2 arms; scratch only, not committed): the applied masks and consumed indices equal the expected values bit for bit, and the per-step valid counts are identical between the arms.

## Commands (repository root)

```bash
cd research/sim_free_augmentation/v4b1
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ../../../.venv/bin/python"
CUDA_VISIBLE_DEVICES= $PY -m unittest discover -s tests -v        # 32 tests
CUDA_VISIBLE_DEVICES= $PY -c "
import sys; sys.path.insert(0,'.'); from sfa4b1 import schedule_masks as sm
m=sm.Meta()
for s in (0,1,2):
    p,r=sm.expected(m,'RpPos_PhaseMask',s),sm.expected(m,'RpPos_MatchedRandMask',s)
    print(s,(p['valid'].sum(-1)==r['valid'].sum(-1)).all(),(p['index']==r['index']).all())"
```

`scripts/masks.py` rewrites `outputs/masks/*.npz`. The zip timestamps change the file hashes, so run it in a copy, or compare the arrays rather than the bytes.

## Questions for the re-audit

1. Is the per-row matched control correct and exactly matched as claimed, including for K = 0 and padded rows?
2. Is the narrowed H2 estimand (A1.3/A1.4) stated correctly, with nothing over-claimed?
3. Does the scorer now fail closed on every defect listed in the 1f80f2e audit? Are the replication guard and labels consistent with the frozen thresholds?
4. Is anything **critical** left that blocks the bounded 15-run launch (`scripts/run_v4b1.py --lanes 4`)?
