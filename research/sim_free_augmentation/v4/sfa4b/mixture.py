"""v4B training mixtures: frozen candidate selection, a two-part read-only dataset view, and a common-random-number
stratified batch sampler (exactly 50 % real rows per batch in augmented arms).

Everything here is a pure function of configs/v4b.json and outputs/v4b/family_audit.json, so the selection can be
re-derived and tested. The frozen v3.1 LeRobot datasets are only READ.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SFA = HERE.parent
CFG = json.loads((HERE / "configs/v4b.json").read_text())
AUDIT = HERE / "outputs/v4b/family_audit.json"


def selection(cfg=CFG, audit_path=AUDIT):
    """-> {"sources": [...], "k": {E: k}, "arms": {arm: {E: [ec_dataset_episode_index, ...]}}, "classes": {...}}"""
    tr = cfg["training"]
    rows = json.loads(Path(audit_path).read_text())["candidates"]
    rows = [r for r in rows if r["in_frozen_Gp_matched_evidence_complete_dataset"]]
    by = {c: {} for c in ("position_only", "handover_only")}
    for r in rows:
        if r["class"] in by:
            by[r["class"]].setdefault(r["E"], []).append(r)
    sources = sorted(set(by["position_only"]) & set(by["handover_only"]))
    order = {}
    for c, d in by.items():
        for E, L in d.items():
            L = sorted(L, key=lambda r: r["key"])
            perm = np.random.default_rng(tr["selection_seed"] + 100 * E + (0 if c == "position_only" else 1)).permutation(len(L))
            order[(c, E)] = [L[i] for i in perm]
    k = {E: min(len(order[("position_only", E)]), len(order[("handover_only", E)])) for E in sources}
    arms = {"R": {}, "RpPos": {}, "RpHO": {}, "RpBoth": {}, "RpPosAll": {}}
    for j, E in enumerate(sources):
        P = [r["ec_dataset_episode_index"] for r in order[("position_only", E)]]
        Hh = [r["ec_dataset_episode_index"] for r in order[("handover_only", E)]]
        arms["RpPos"][E] = P[:k[E]]
        arms["RpHO"][E] = Hh[:k[E]]
        kp = (k[E] + 1) // 2 if j % 2 == 0 else k[E] // 2              # alternate which family gets the odd one
        arms["RpBoth"][E] = P[:kp] + Hh[:k[E] - kp]
        arms["RpPosAll"][E] = P
    keys = {(r["E"], r["ec_dataset_episode_index"]): r["key"] for r in rows}
    return {"sources": sources, "k": k, "arms": arms,
            "keys": {a: {E: [keys[(E, i)] for i in v] for E, v in d.items()} for a, d in arms.items()}}


class TwoPart:
    """Read-only concatenation of the real dataset (indices [0, nR)) and the frozen synthetic dataset ([nR, nR+nG)).
    Exposes the REAL dataset's metadata, so normalisation statistics are real-only for every arm."""

    def __init__(self, ds_real, ds_syn):
        self.parts = (ds_real, ds_syn)
        self.nR = len(ds_real)
        self.meta = ds_real.meta
        self.episodes = None
        self.num_frames = len(ds_real) + len(ds_syn)
        self.num_episodes = ds_real.num_episodes + ds_syn.num_episodes

    def __len__(self):
        return self.num_frames

    def __getitem__(self, i):
        i = int(i)
        return self.parts[0][i] if i < self.nR else self.parts[1][i - self.nR]


def episode_ranges(meta):
    ep = meta.episodes
    return {int(e): (int(a), int(b)) for e, a, b in zip(ep["episode_index"], ep["dataset_from_index"], ep["dataset_to_index"])}


def frame_pools(arm, sel, real_ranges, syn_ranges, nR):
    """real: {real episode index: frame indices}; syn: {source E: global indices of this arm's synthetic frames of E}.
    The real dataset's episode index i IS source episode i (sfa_v31_R holds real_10 episodes 0-9 in order)."""
    real = {int(e): np.arange(a, b) for e, (a, b) in sorted(real_ranges.items())}
    syn = {}
    for E, eps in sel["arms"].get(arm, {}).items():
        syn[E] = np.concatenate([np.arange(*syn_ranges[i]) for i in eps]) + nR if eps else np.zeros(0, int)
    return real, syn


def real_source_weights(arm, sources_real, sources_syn, real_fraction):
    """Source-balanced exposure: every source episode receives exactly 1/len(sources_real) of all training rows in
    EVERY arm. R: uniform over sources. Augmented: synthetic rows are uniform over sources_syn, so the real-row
    weights compensate: w_E = (1/n - (1 - rf) * s_E) / rf."""
    n = len(sources_real)
    if arm == "R":
        return np.full(n, 1.0 / n)
    s = np.array([1.0 / len(sources_syn) if E in sources_syn else 0.0 for E in sources_real])
    w = (1.0 / n - (1 - real_fraction) * s) / real_fraction
    assert (w >= 0).all() and abs(w.sum() - 1) < 1e-12, w
    return w


class StratifiedBatches:
    """Yields `steps` batches of indices with common random numbers across arms: for (seed, step) every arm uses the
    same uniforms. Real slot j: source = inverse-CDF(u_src[j]; arm real weights), frame = uniform within that source.
    R uses 16 real slots; augmented arms use the first round(16 * real_fraction) real slots plus synthetic slots:
    source ~ uniform over the shared synthetic sources, frame = pool_E[floor(u * |pool_E|)]."""

    def __init__(self, arm, sel, real, syn, steps, batch_size, seed, base_seed, real_fraction=0.5):
        self.arm, self.real, self.syn, self.steps, self.bs = arm, real, syn, steps, batch_size
        self.seed, self.base = seed, base_seed
        self.n_real = batch_size if arm == "R" else int(round(batch_size * real_fraction))
        self.sources = sel["sources"]
        self.real_sources = sorted(real)
        self.w = real_source_weights(arm, self.real_sources, set(self.sources), real_fraction)
        self.cdf = np.cumsum(self.w)

    def __len__(self):
        return self.steps

    def batch(self, step):
        rng = np.random.default_rng([self.base, self.seed, step])
        ur, vr = rng.random(self.bs), rng.random(self.bs)
        us, vs = rng.integers(0, len(self.sources), self.bs), rng.random(self.bs)
        out = []
        for j in range(self.n_real):
            E = self.real_sources[min(int(np.searchsorted(self.cdf, ur[j], side="right")), len(self.cdf) - 1)]
            out.append(int(self.real[E][int(vr[j] * len(self.real[E]))]))
        for j in range(self.bs - self.n_real):
            pool = self.syn[self.sources[us[j]]]
            out.append(int(pool[int(vs[j] * len(pool))]))
        return out

    def __iter__(self):
        for step in range(self.steps):
            yield self.batch(step)


def schedule_digest(sampler, n=None):
    h = hashlib.sha256()
    for step in range(n or sampler.steps):
        h.update(np.asarray(sampler.batch(step), np.int64).tobytes())
    return h.hexdigest()
