"""Registered target-paired statistics; model-free, no generator imports."""
import numpy as np
from scipy.stats import binomtest

SEED = 20261108
N_BOOT = 10000


def paired_effect(a, b, target_indices=None):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.ndim==1 and len(a)>0 and np.isfinite(a).all() and np.isfinite(b).all()
    n=len(a)
    ids=np.arange(100) if target_indices is None else np.asarray(target_indices)
    assert len(ids)==n and len(set(ids.tolist()))==n and ids.min()>=0 and ids.max()<100
    d = a.astype(float) - b.astype(float)
    rng = np.random.Generator(np.random.PCG64(SEED))
    ix = rng.integers(0, n, (N_BOOT, n))
    ordinary = d[ix].mean(1)
    # All100: 25 adjacent2x2 blocks. For a common RGB subset retain its nonempty original blocks.
    block_id=5*((ids//10)//2)+(ids%10)//2
    groups=[np.flatnonzero(block_id==k) for k in sorted(set(block_id))]
    sums=np.array([d[g].sum() for g in groups]);counts=np.array([len(g) for g in groups])
    bi = rng.integers(0,len(groups),(N_BOOT,len(groups)))
    spatial = sums[bi].sum(1)/counts[bi].sum(1)
    out = {"difference": float(d.mean()), "paired_ci95": np.quantile(ordinary,[.025,.975]).tolist(),
           "spatial_block_ci95": np.quantile(spatial,[.025,.975]).tolist()}
    if np.isin(a,[0,1]).all() and np.isin(b,[0,1]).all():
        win, loss = int(((a==1)&(b==0)).sum()), int(((a==0)&(b==1)).sum())
        out.update(sd_only=win, control_only=loss,
                   exact_mcnemar_p_two_sided=float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.0)
    return out


def meaningful_yield(a,b):
    assert np.shape(a)==np.shape(b)==(100,)
    s=paired_effect(a,b)
    return {**s,"PASS":bool(s["difference"] >= .10-1e-12 and s["paired_ci95"][0]>0
               and s["spatial_block_ci95"][0]>0 and s["exact_mcnemar_p_two_sided"]<=.05)}
