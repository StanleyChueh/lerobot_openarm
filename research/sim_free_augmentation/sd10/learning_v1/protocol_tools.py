"""Model-free SD10-L1 reference definitions. No training, policy loading, or generated trajectories."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

HEAD = "59b2bfe697c7400e201c988d3f05123b9018dcac"
LEROBOT_HEAD = "530b253f262d6af92ea3b91d62abcae0295af59f"
ARMS = {
    "Original10": list(range(10)),
    "SD10": [1, 6, 7, 11, 13, 14, 27, 34, 42, 47],
    "Random10": [5, 6, 7, 8, 18, 25, 29, 32, 43, 44],
}
# Shared captures occupy identical slots in every arm that contains them.
SLOTS = {
    "Original10": [6, 7, 1, 5, 8, 0, 2, 3, 4, 9],
    "SD10": [6, 7, 1, 11, 13, 14, 27, 34, 42, 47],
    "Random10": [6, 7, 18, 5, 8, 25, 29, 32, 43, 44],
}
EVAL = [12, 15, 16, 19, 20, 21, 22, 23, 26, 28, 30, 33, 35, 36, 37, 39, 40, 41, 46, 48, 49]
V4A = [10, 17, 24, 31, 38, 45]
SEEDS = [0, 1, 2]
PHASES = ["approach", "grasp", "carry", "handover", "release"]
HORIZON, UPDATES, BATCH, SAMPLER_SEED = 50, 1000, 16, 20261107
TASK = "Pick up the Pringles can with the right arm, hand it to the left arm"


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def events(action):
    # Interleaved 16-D raw command layout; open values from the recorded dataset convention.
    r = np.clip(action[:, 14] / -1.298, 0, 1) < .5
    l = np.clip(action[:, 15] / 1.2121, 0, 1) < .5
    rr = np.flatnonzero(r[1:] != r[:-1]) + 1
    ll = np.flatnonzero(l[1:] != l[:-1]) + 1
    assert not r[0] and not l[0] and len(rr) == 2 and len(ll) == 1
    rc, lc, ro = int(rr[0]), int(ll[0]), int(rr[1])
    assert 5 <= rc and rc + 15 <= lc - 5 <= ro + 5 < len(action)
    return rc, lc, ro


def phases(action):
    rc, lc, ro = events(action)
    t = np.arange(len(action))
    return np.select([t < rc - 5, t < rc + 15, t < lc - 5, t < ro + 5], [0, 1, 2, 3], default=4).astype(np.int8)


def eval_anchors(length):
    assert length >= HORIZON
    return sorted(set(range(0, length - HORIZON + 1, 5)) | {length - HORIZON})


def schedule(seed):
    """Exact 1,600 draws per slot; paired stratified time quantiles, independent of outcomes."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SAMPLER_SEED, seed])))
    n = UPDATES * BATCH // 10
    slot = np.repeat(np.arange(10, dtype=np.int16), n)
    rng.shuffle(slot)
    q = np.empty(len(slot), dtype=np.float64)
    for j in range(10):
        qq = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(qq)
        q[slot == j] = qq
    return slot.reshape(UPDATES, BATCH), q.reshape(UPDATES, BATCH)


def learning_rates():
    # Current LeRobot CosineDecayWithWarmup: configured 1000/30000 scales to 33/1000.
    out = []
    for step in range(UPDATES):
        if step < 33:
            factor = 1 / 34 if step == 0 else (1 / 34 - 1) * (1 - step / 33) + 1
        else:
            factor = (1 - .025) * .5 * (1 + math.cos(math.pi * min(step, 1000) / 1000)) + .025
        out.append(1e-4 * factor)
    return np.array(out, dtype=np.float64)


def recording_scores(pred, items, targets, phase_ids, lengths):
    """First average absolute errors of forecasts of each native target time, then recordings/phases.

    pred/targets: [anchors,50,16], float32 physical raw coordinates. Each repeated target is one
    physical frame for phase averaging; forecasts are NOT averaged before computing absolute error.
    """
    assert pred.dtype == targets.dtype == np.float32
    assert pred.shape == targets.shape == (len(items), HORIZON, 16)
    assert np.isfinite(pred).all() and np.isfinite(targets).all()
    arm = np.abs(pred[..., :14].astype(np.float64) - targets[..., :14]).mean(-1)
    # Do not clip: a saturated-looking but badly scaled command must still incur an error.
    grip = np.abs((pred[..., 14:].astype(np.float64) - targets[..., 14:]) / np.array([-1.298, 1.2121]))
    out = {}
    for ep in EVAL:
        n = lengths[ep]
        sums, gs, counts = np.zeros(n), np.zeros((n, 2)), np.zeros(n, dtype=np.int64)
        ph = np.full(n, -1, dtype=np.int8)
        for i in np.flatnonzero(items[:, 0] == ep):
            t = int(items[i, 1]); rows = np.arange(t, t + HORIZON)
            sums[rows] += arm[i]; gs[rows] += grip[i]; counts[rows] += 1
            old = ph[rows]; assert np.all((old == -1) | (old == phase_ids[i]))
            ph[rows] = phase_ids[i]
        assert np.all(counts > 0) and np.all(ph >= 0)
        vals, gv = sums / counts, gs / counts[:, None]
        out[ep] = {name: float(vals[ph == j].mean()) for j, name in enumerate(PHASES)}
        out[ep]["overall_time_weighted"] = float(vals.mean())
        out[ep]["phase_balanced"] = float(np.mean([out[ep][p] for p in PHASES]))
        out[ep]["handover_gripper_openness_mae"] = gv[ph == 3].mean(0).tolist()
    return out


def paired_relative(x, ref, recording_indices, seed_indices=None):
    """[3,recordings], positive = lower error. CIs resample paired recording IDs, never chunks."""
    assert x.shape == ref.shape and x.shape[0] == 3 and np.isfinite(x).all() and np.isfinite(ref).all()
    assert np.all(ref.mean(1) > 0)
    point = float(1 - x.mean() / ref.mean())
    if seed_indices is None:
        xx, rr = x.mean(0), ref.mean(0)
        bs = 1 - xx[recording_indices].mean(1) / rr[recording_indices].mean(1)
    else:
        bs = np.array([1 - x[np.ix_(si, ei)].mean() / ref[np.ix_(si, ei)].mean()
                       for si, ei in zip(seed_indices, recording_indices)])
    assert np.isfinite(bs).all()
    return {"relative_improvement": point, "ci95": np.quantile(bs, [.025, .975]).tolist(),
            "per_seed": (1 - x.mean(1) / ref.mean(1)).tolist()}
