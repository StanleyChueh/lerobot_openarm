"""Training-source pace and realized phase exposure for SD10-L1 (descriptive; not used by any decision).

Realized exposure is recomputed from the CONSUMED (episode,start) draws asserted in each run and compared with the
frozen training_exposure.json. Writes outputs/exposure_pace.json.
"""
from __future__ import annotations

import json

import numpy as np

import l1lib as L

PT = L.PT


def main():
    frozen = json.loads((L.L1 / "training_exposure.json").read_text())
    caps = L.RT["captures"]
    A = {e: L.C.pool().vec(e, "action") for e in L.TRAIN_EPS}
    ph = {e: PT.phases(A[e]) for e in A}
    out = {"pace": {}, "exposure": {}}
    for arm, ids in L.ARMS.items():
        ev = np.array([caps[str(e)]["command_events_rc_lc_ro"] for e in ids])
        T = np.array([caps[str(e)]["length"] for e in ids])
        rc, lc, ro = ev.T
        dur = {"length_frames": T, "approach_to_rc_s": rc / 30, "rc_to_lc_s": (lc - rc) / 30, "lc_to_ro_s": (ro - lc) / 30,
               "ro_to_end_s": (T - ro) / 30}
        phase_frames = np.stack([np.bincount(ph[e], minlength=5) for e in ids])
        out["pace"][arm] = {k: {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()),
                                "values": v.tolist()} for k, v in dur.items()}
        out["pace"][arm]["native_phase_frames_total"] = dict(zip(PT.PHASES, phase_frames.sum(0).tolist()))
        out["pace"][arm]["native_rows"] = int(T.sum())
        out["exposure"][arm] = {}
        for s in L.SEEDS:
            z = np.load(L.L1 / f"sampling_seed{s}.npz", allow_pickle=False)
            term = json.loads((L.TRAIN / f"{arm}_s{s}" / "terminal.json").read_text())
            assert term["schedule_state"]["consumed_equals_frozen"]
            ep, fr = z[arm + "_episode"].ravel(), z[arm + "_frame"].ravel()
            tgt = np.stack([ph[int(e)][int(t):int(t) + 50] for e, t in zip(ep, fr)])
            inp = np.array([ph[int(e)][int(t)] for e, t in zip(ep, fr)])
            tc, ic = np.bincount(tgt.ravel(), minlength=5), np.bincount(inp, minlength=5)
            assert np.array_equal(tc, frozen["seeds"][str(s)][arm]["target_phase_counts"])
            uniq = sum(len(np.unique(fr[ep == e])) for e in ids)
            reuse = [np.bincount((fr[ep == e][:, None] + np.arange(50)).ravel(), minlength=caps[str(e)]["length"]) for e in ids]
            out["exposure"][arm][str(s)] = {"target_phase_counts": dict(zip(PT.PHASES, tc.tolist())),
                                            "input_phase_counts": dict(zip(PT.PHASES, ic.tolist())),
                                            "valid_target_timesteps": int(tc.sum()), "unique_input_starts": int(uniq),
                                            "eligible_input_starts": int(sum(caps[str(e)]["length"] - 49 for e in ids)),
                                            "mean_reuse_per_native_target_row": float(np.mean(np.concatenate(reuse))),
                                            "source_draws": term["source_draws_consumed"],
                                            "optimizer_consumed_batches": term["consumed_optimizer_updates"],
                                            "discarded_prefetched_batches": term["schedule_state"]["discarded_prefetched_batches"],
                                            "matches_frozen_plan": True}
        m = np.mean([[out["exposure"][arm][str(s)]["target_phase_counts"][p] for p in PT.PHASES] for s in L.SEEDS], 0)
        out["exposure"][arm]["seed_mean_target_phase_counts"] = dict(zip(PT.PHASES, m.tolist()))
    base = out["exposure"]["Original10"]["seed_mean_target_phase_counts"]
    for arm in L.ARMS:
        out["exposure"][arm]["relative_to_Original10"] = {p: out["exposure"][arm]["seed_mean_target_phase_counts"][p] / base[p] - 1 for p in PT.PHASES}
    (L.OUT / "exposure_pace.json").write_text(json.dumps(out, indent=1))
    for arm in L.ARMS:
        print(arm, {k: round(v["mean"], 3) for k, v in out["pace"][arm].items() if isinstance(v, dict) and "mean" in v},
              {p: f"{100 * v:+.1f}%" for p, v in out["exposure"][arm]["relative_to_Original10"].items()})


if __name__ == "__main__":
    main()
