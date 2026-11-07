"""STEP 1b - are the original-ten videos the same captures as pool episodes 0-9, at the same frame alignment?

For each matched pair (ten i, pool i) and each camera, compare ten frame t with pool frame t+s for s in -2..2.
Same capture + same alignment => s=0 is far smaller than s=+-1. Appends 'alignment' to outputs/duplication.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sd10 import common as C  # noqa: E402

SHIFTS = (-2, -1, 0, 1, 2)


def job(args):
    cam, i, j = args
    a = C.ten().decode(cam, i)[:, ::2, ::2].astype(np.int16)
    b = C.pool().decode(cam, j)[:, ::2, ::2].astype(np.int16)
    n = len(a)
    r = {}
    for s in SHIFTS:
        lo, hi = max(0, -s), min(n, n - s)
        r[s] = float(np.abs(a[lo:hi] - b[lo + s:hi + s]).mean())
    return cam, i, r


def main():
    dup = json.loads((C.OUT / "duplication.json").read_text())
    pairs = {int(k): v for k, v in dup["mapping_ten_to_pool"].items() if v is not None}
    jobs = [(cam, i, j) for i, j in pairs.items() for cam in C.CAMS]
    out = {}
    with ProcessPoolExecutor(12) as ex:
        for cam, i, r in ex.map(job, jobs):
            out.setdefault(str(i), {})[cam] = r
    best0 = all(min(r, key=r.get) == 0 for v in out.values() for r in v.values())
    ratio = min(min(r[1], r[-1]) / r[0] for v in out.values() for r in v.values())
    dup["alignment"] = {"mean_abs_px_by_shift": out, "zero_shift_best_everywhere": best0,
                        "min_ratio_shift1_over_shift0": ratio}
    C.dump(dup, C.OUT / "duplication.json")
    print("zero shift best everywhere:", best0, "min ratio(+-1 / 0):", round(ratio, 2))


if __name__ == "__main__":
    main()
