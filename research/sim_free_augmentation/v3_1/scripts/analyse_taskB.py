"""Task B analysis: v3 (frozen) vs v3.1 regeneration - boundary states, label-rule sensitivity, the 66 v3 lead-only
rejections, receiver evidence status, matched-set changes.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/analyse_taskB.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
V3 = HERE.parent / "v3"
EPS = list(range(10))


def R(d, E):
    return json.loads((d / f"outputs/ep{E}/results.json").read_text())


def main():
    out = {}
    # 1. boundary states of matched-scale plans (own scale) G+ vs C: frame 0 and first exported row (2)
    d0, d2, d0v3 = [], [], []
    for E in EPS:
        r1 = R(HERE, E)
        for vid, p in r1["plans"].items():
            s = p["common_scale"]
            if s is None or vid == "identity":
                continue
            fg, fc = HERE / f"outputs/ep{E}/candidates/{vid}__G+__s{s}.npz", HERE / f"outputs/ep{E}/candidates/{vid}__C__s{s}.npz"
            if fg.exists() and fc.exists():
                g, c = np.load(fg)["state"], np.load(fc)["state"]
                d0.append(float(np.abs(g[0] - c[0]).max()))
                d2.append(float(np.abs(g[2] - c[2]).max()))
        r3 = R(V3, E)
        for vid, p in r3["plans"].items():
            s = p["common_scale"]
            if s is None or vid == "identity":
                continue
            fg, fc = V3 / f"outputs/ep{E}/candidates/{vid}__G+__s{s}.npz", V3 / f"outputs/ep{E}/candidates/{vid}__C__s{s}.npz"
            if fg.exists() and fc.exists():
                d0v3.append(float(np.abs(np.load(fg)["state"][0] - np.load(fc)["state"][0]).max()))
    out["boundary_states"] = {"v31_frame0_max_rad": max(d0), "v31_first_export_row_max_rad": max(d2), "n_pairs": len(d0),
                              "v3_frame0_max_rad": max(d0v3), "n_pairs_v3": len(d0v3)}
    # 2. the 66 v3 lead-only rejections, re-evaluated in v3.1
    lead_only = []
    for E in EPS:
        r3 = R(V3, E)
        for k, c in r3["candidates"].items():
            if (not c["labels_ok"]) and not c["verify_failed"] and c["ik"]["ok"] and c["targets_ok"]:
                lead_only.append((E, k))
    r1s = {E: R(HERE, E) for E in EPS}
    cls = collections.Counter()
    rows = []
    for E, k in lead_only:
        c1 = r1s[E]["candidates"].get(k)
        if c1 is None:
            cls["not re-proposed in v3.1 (earlier scale accepted)"] += 1
            rows.append({"E": E, "key": k, "v31": "not re-proposed"})
            continue
        if not c1["accepted"]:
            why = c1["verify_failed"] + ([] if c1["labels_ok"] else ["hard-label"]) + ([] if c1["targets_ok"] else ["targets"])
            cls["rejected in v3.1: " + ",".join(why)] += 1
        elif c1["residual_uncertain"]:
            cls["accepted, RESIDUAL_UNCERTAIN"] += 1
        else:
            cls["accepted, residual-consistent (lead diagnostic flag only)"] += 1
        rows.append({"E": E, "key": k, "accepted": c1["accepted"], "residual_uncertain": c1["residual_uncertain"],
                     "lead_flag": c1["lead_diagnostic_flag"], "residual_max_ratio": c1["labels"]["residual"]["max_ratio"]})
    out["v3_lead_only_rejections"] = {"n": len(lead_only), "by_condition": dict(collections.Counter(k.split("__")[1] for _, k in lead_only)),
                                      "v31_classification": dict(cls), "rows": rows}
    # 3. paired sensitivity: acceptance / matched pairs, v3 vs v3.1
    def counts(d):
        acc = {c: 0 for c in ("G+", "C")}
        full = {c: 0 for c in ("G+", "C")}
        common = 0
        unc = {c: 0 for c in ("G+", "C")}
        for E in EPS:
            r = R(d, E)
            for vid, p in r["plans"].items():
                if vid == "identity":
                    continue
                for c in ("G+", "C"):
                    if p["own_scale"][c] is not None:
                        acc[c] += 1
                        full[c] += p["own_scale"][c] == 1.0
                        cc = r["candidates"][f"{vid}__{c}__s{p['own_scale'][c]}"]
                        unc[c] += bool(cc.get("residual_uncertain", False))
                common += p["common_scale"] is not None
        return {"accepted_own_scale": acc, "full_scale": full, "common_scale_pairs": common, "residual_uncertain_among_accepted": unc}
    out["sensitivity"] = {"v3": counts(V3), "v31": counts(HERE)}
    # receiver evidence / uncertainty in the v3.1 export checks
    ec = HERE / "outputs/dataset_export/export_checks.json"
    if ec.exists():
        x = json.loads(ec.read_text())
        chk = {(c["episode"], c["key"]): c for c in x["checks"]}
        st = collections.Counter((k[1].split("__")[1], c["receiver_lag_status"]) for k, c in chk.items() if c["ok"])
        out["export"] = {"matched_pairs": len(x["matched"]), "c_all": len(x["c_all"]),
                         "receiver_status_among_export_ok": {f"{a}:{b}": n for (a, b), n in st.items()},
                         "evidence_complete_matched_pairs": sum(1 for E, v, s, f in x["matched"]
                                                                if chk[(E, f"{v}__G+__s{s}")]["evidence_complete"] and chk[(E, f"{v}__C__s{s}")]["evidence_complete"])}
    (HERE / "outputs/taskB_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "v3_lead_only_rejections"} | {"lead_only": {kk: vv for kk, vv in out["v3_lead_only_rejections"].items() if kk != "rows"}}, indent=1, default=float))


if __name__ == "__main__":
    main()
