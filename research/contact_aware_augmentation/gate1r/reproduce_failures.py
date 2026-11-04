"""Gate 1R step 1: reproduce Codex's three critical failures of the Gate 1 verifier on the real pilot data, and
record WHY each passes (which code path, which frames, which values). Uses the frozen Gate 1 code unchanged.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python \
        research/contact_aware_augmentation/gate1r/reproduce_failures.py --out research/contact_aware_augmentation/outputs/gate1r/reproduction.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G1 = HERE.parent / "gate1"
sys.path[:0] = [str(G1)]
import gate1_core as gc  # noqa: E402
import run_gate1 as rg  # noqa: E402

METHODS = ("C", "C_rel", "B_rel", "B")


def measure(s, tail, th, refC, refB):
    row = gc.score_all(s, gc.g0.BODY_CAM, tail, refC, refB)
    return row, {m: dict(zip(("score", "flag", "why"), gc.method_score(row, th, m))) for m in METHODS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", default=str(HERE.parent / "outputs/gate1"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    m = json.loads((Path(a.gate1) / "metrics_full.json").read_text())
    folds = {f["test"]: f for f in m["folds"]}
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=rg.PRINGLES, label="p", episodes=list(rg.PILOT))
    Q, A = P.vectors("observation.state"), P.vectors("action")
    data = {}
    for e in rg.PILOT:
        d = np.load(Path(a.gate1) / "tracks" / f"lid_track_ep{e}.npz")
        ev = m["tracker"]["per_episode"][str(e)]["events_action"]
        data[e] = {"q": Q[e], "a": A[e], "names": P.names("action"), "events": ev, "uv": d["uv"], "ok": d["status"] == "ok"}
    rg._init_worker(data)
    out = {"sample_fields_of_gate1_verifier": [f.name for f in dataclasses.fields(gc.Sample)], "episodes": {}}
    for e in rg.PILOT:
        f = folds[e]
        lag, tail, th = f["lag"], rg.tail_end(e), f["thresholds"]
        refC = {k: np.array(v) / 1e3 for k, v in f["refC_mm"].items()}
        refB = {k: np.array(v) / 1e3 for k, v in f["refB_mm"].items()}
        E = rg.episode(e, lag)
        base = gc.sample_from_episode(E)
        br, bs = measure(base, tail, th, refC, refB)
        w = gc.windows(base, tail)
        rec = {"lag": lag, "tail_end": tail, "events_action": E.events,
               "windows": {k: [int(v[0]), int(v[-1])] if len(v) else None for k, v in w.items() if k != "phase"},
               "original": bs}
        # F1: +1 rad on all 14 arm action channels. The Sample built from the episode has no action field at all.
        bad = copy.deepcopy(E)
        bad.a[:, :14] += 1.0
        s1 = gc.sample_from_episode(bad)
        r1, f1 = measure(s1, tail, th, refC, refB)
        rec["F1_action_plus_1rad"] = {"scores": f1, "identical_to_original": f1 == bs,
                                      "sample_has_action_field": hasattr(s1, "a"),
                                      "max_abs_action_change_rad": float(np.abs(bad.a - E.a).max())}
        # F2: smooth ~60 px horizontal excursion only between the end of the giver window and the start of the
        # receiver window (Codex's construction).
        s2 = copy.deepcopy(base)
        start, stop = int(w["giver"][-1]) + 1, int(w["recv"][0])
        shape = np.sin(np.linspace(0, np.pi, stop - start + 2))[1:-1] ** 2
        s2.uv[start:stop, 0] += 60 * shape
        r2, f2 = measure(s2, tail, th, refC, refB)
        scored = set(w["giver"]) | set(w["recv"]) | set(w["release"])
        gap = list(range(start, stop))
        rec["F2_transition_excursion_60px"] = {
            "scores": f2, "C_rel_unchanged": abs(f2["C_rel"]["score"] - bs["C_rel"]["score"]) < 1e-10,
            "excursion_frames": [start, stop - 1], "peak_px": float(60 * shape.max()),
            "excursion_frames_in_any_C_rel_window": sorted(set(gap) & scored),
            "takeover_window": rec["windows"]["takeover"],
            "takeover_component_uses_observed_track": False,  # takeover_px compares two fitted predictions only
            "takeover_px_unchanged": r2["C.takeover_px"] == br["C.takeover_px"]}
        # F3: no receiver-hold observation at all.
        s3 = copy.deepcopy(base)
        s3.ok[w["recv"]] = False
        r3, f3 = measure(s3, tail, th, refC, refB)
        rec["F3_missing_receiver"] = {"scores": f3, "res_recv": r3["C.res_recv"], "n_recv": r3["C.n_recv"],
                                      "release_px": r3["C.release_px"],
                                      "C_rel_status": "PASS" if not f3["C_rel"]["flag"] else "FAIL",
                                      "unevaluable_noted_only_in_why": "unevaluable_component" in f3["C_rel"]["why"]}
        out["episodes"][e] = rec
    out["summary"] = {
        "F1_passes_C_rel": sum(not r["F1_action_plus_1rad"]["scores"]["C_rel"]["flag"] for r in out["episodes"].values()),
        "F1_identical_all_methods": sum(r["F1_action_plus_1rad"]["identical_to_original"] for r in out["episodes"].values()),
        "F2_passes_C_rel": sum(not r["F2_transition_excursion_60px"]["scores"]["C_rel"]["flag"] for r in out["episodes"].values()),
        "F3_passes_C_rel": sum(r["F3_missing_receiver"]["C_rel_status"] == "PASS" for r in out["episodes"].values()),
        "n": len(rg.PILOT)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else (None if x != x else x)))
    print(json.dumps(out["summary"]), json.dumps({e: {k: v for k, v in r.items() if k in ("windows",)} for e, r in out["episodes"].items()}))
    for e, r in out["episodes"].items():
        print(e, "F2 frames", r["F2_transition_excursion_60px"]["excursion_frames"], "in windows:",
              r["F2_transition_excursion_60px"]["excursion_frames_in_any_C_rel_window"], "F3 res_recv", r["F3_missing_receiver"]["res_recv"])


if __name__ == "__main__":
    main()
