"""Expected per-batch supervision of a v4B.1 arm under its frozen v4B schedule, from parquet metadata only (no video).

Used by scripts/masks.py (manifest + budget accounting), scripts/verify_runs.py (training-recorded masks vs expected)
and tests/test_v4b1_amendment.py. Mask logic itself lives in interventions.py (single source of truth).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import interventions as iv

mx = iv.mx
V31 = iv.HERE.parent / "v3_1"


def tables(name):
    root = V31 / "outputs/lerobot" / name
    ep = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))]).sort_values("index")
    ranges = {int(e): (int(a), int(b)) for e, a, b in zip(ep["episode_index"], ep["dataset_from_index"], ep["dataset_to_index"])}
    return ranges, df


class Meta:
    """row-level facts for the concatenated real+synthetic view (global index i: real < nR <= synthetic)."""

    def __init__(self):
        from sfa.data import command_events  # noqa: PLC0415
        self.rr, dr = tables("sfa_v31_R")
        self.sr, ds = tables("sfa_v31_Gp_matched_evidence_complete")
        self.nR = len(dr)
        A = np.stack(ds["action"].to_numpy())
        self.grasp_end = {e: command_events(A[a:b])["right_close"] + iv.CFG["grasp_end_offset"] for e, (a, b) in self.sr.items()}
        n = self.nR + len(ds)
        self.ep, self.frame, self.length = np.empty(n, int), np.empty(n, int), np.empty(n, int)
        for off, R in ((0, self.rr), (self.nR, self.sr)):
            for e, (a, b) in R.items():
                self.ep[off + a:off + b], self.frame[off + a:off + b], self.length[off + a:off + b] = e, np.arange(b - a), b - a
        self.sel = mx.selection()
        self.syn_source = {e: E for arm in self.sel["arms"].values() for E, eps in arm.items() for e in eps}

    def source(self, i):
        return int(self.ep[i]) if i < self.nR else self.syn_source[int(self.ep[i])]

    def schedule(self, base, seed, steps=1000):
        real, syn = mx.frame_pools(base, self.sel, self.rr, self.sr, self.nR)
        tr = mx.CFG["training"]
        return mx.StratifiedBatches(base, self.sel, real, syn, steps, tr["batch_size"], seed, tr["sampler_seed"], tr["real_fraction"])


def expected(meta, arm, seed, steps=1000):
    """-> dict(index [S,B], is_syn [S,B], source [S,B], valid_orig [S,B,50], valid [S,B,50]) for the arm's schedule."""
    spec = iv.arm_spec(arm)
    idx = np.array([list(b) for b in meta.schedule(spec["base"], seed, steps)], np.int64)
    is_syn = idx >= meta.nR
    valid_orig = (meta.frame[idx][..., None] + np.arange(iv.CHUNK)) < meta.length[idx][..., None]
    valid = valid_orig.copy()
    for s, b in zip(*np.nonzero(is_syn)):
        i = int(idx[s, b])
        ex = iv.synthetic_exclusion(spec["target_mask"], seed, i, int(meta.frame[i]), int(meta.length[i]),
                                    meta.grasp_end.get(int(meta.ep[i])))
        valid[s, b] &= ~ex
    src = np.vectorize(meta.source)(idx)
    return {"index": idx, "is_syn": is_syn, "source": src, "valid_orig": valid_orig, "valid": valid}
