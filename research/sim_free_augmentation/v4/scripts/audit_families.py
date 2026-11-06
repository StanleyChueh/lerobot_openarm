"""v4B Task 1: audit which corrected v3.1 G+ candidates change (A) only the initial object position, (B) only the
handover interaction, or (C) both - measured from the frozen candidate arrays, not from family names.

    cd research/sim_free_augmentation/v4
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/audit_families.py

Read-only on v3/v3.1. Writes outputs/v4b/family_audit.json and family_audit.md.

Per candidate, against the identity candidate of the same source episode (== the recorded source, checked here):
  init_shift_mm   |p_can(0) - p_can_src(0)|                    (standing can before grasp)
  approach_dq     max |q - q_src| over arm joints, frames [0, rc-5)      (pre-grasp robot motion)
  grasp_dq        same over [rc-5, rc+15)
  handover_dq     same over [lc-5, ro+5)                                   (both arms, transfer)
  handover_can_mm max |p_can - p_can_src| over [lc-5, ro+5)
  grip_shift      gripper-command event times differ from the source (frames)
Classes (thresholds frozen in configs/v4b.json):
  position_only : init_shift >= 5 mm, handover window unchanged (dq <= 1 mrad, can <= 1 mm, no gripper-event shift)
  handover_only : init_shift <= 1 mm, approach unchanged (dq <= 1 mrad), handover window changed
  both          : init_shift >= 5 mm and handover window changed
  other         : anything else (reported, never trained on)
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SFA = HERE.parent
V3, V31 = SFA / "v3", SFA / "v3_1"
sys.path.insert(0, str(V3))
from sfa3.recon3 import data, rb  # noqa: E402

CFG = json.loads((HERE / "configs/v4b.json").read_text())
TH = CFG["family_audit_thresholds"]
ARMJ = [rb.NAMES.index(f"{s}J{i}.pos") for s in "RL" for i in range(1, 8)]
OUT = HERE / "outputs/v4b"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def family_of(vid, fams):
    if vid == "identity":
        return "identity"
    return next(f for f, L in fams.items() if any(d["id"] == vid for d in L))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg31 = json.loads((V31 / "configs/v3_1.json").read_text())
    fams = cfg31["families"]
    checks = json.loads((V31 / "outputs/dataset_export/export_checks.json").read_text())
    chk = {(c["episode"], c["key"]): c for c in checks["checks"]}
    ec_manifest = json.loads((V31 / "outputs/dataset_export/Gp_matched_evidence_complete_manifest.json").read_text())
    in_ec_dataset = {(e["E"], e["key"]): e["episode_index"] for e in ec_manifest if e["kind"] == "synthetic"}
    rows = []
    src_check = {}
    for E in range(10):
        res = json.loads((V31 / f"outputs/ep{E}/results.json").read_text())
        ev = res["recon"]["events"]
        rc, lc, ro = ev["right_close"], ev["left_close"], ev["right_open"]
        ident = np.load(V31 / f"outputs/ep{E}/candidates/identity__G+__s1.0.npz")
        src = data.load(E, cams=())
        src_check[E] = {"identity_state_equals_recorded_state_max_abs": float(np.abs(ident["state"] - src.state).max()),
                        "T": int(len(src.state)), "events": ev}
        qs, cs = ident["state"][:, ARMJ], ident["can"][:, :3, 3]
        g_src = data.command_events(src.action)
        for key, c in res["candidates"].items():
            if c["cond"] != "G+" or not c["accepted"] or c["var"] == "identity":
                continue
            z = np.load(V31 / f"outputs/ep{E}/candidates/{key}.npz")
            q, can = z["state"][:, ARMJ], z["can"][:, :3, 3]
            T = len(q)
            w = {"approach": (0, rc - 5), "grasp": (rc - 5, rc + 15), "handover": (lc - 5, min(T, ro + 5))}
            dq = {k: float(np.abs(q[a:b] - qs[a:b]).max()) if b > a else 0.0 for k, (a, b) in w.items()}
            a, b = w["handover"]
            dcan_ho = float(np.linalg.norm(can[a:b] - cs[a:b], axis=1).max() * 1000)
            init_mm = float(np.linalg.norm(can[0] - cs[0]) * 1000)
            g_c = data.command_events(z["action"])
            grip_shift = int(max(abs(g_c[k] - g_src[k]) for k in g_src))
            ho_changed = dq["handover"] > TH["dq_rad"] or dcan_ho > TH["can_mm"] or grip_shift > 0
            if init_mm >= TH["init_moved_mm"] and not ho_changed:
                cls = "position_only"
            elif init_mm <= TH["init_still_mm"] and dq["approach"] <= TH["dq_rad"] and ho_changed:
                cls = "handover_only"
            elif init_mm >= TH["init_moved_mm"] and ho_changed:
                cls = "both"
            else:
                cls = "other"
            ck = chk.get((E, key), {})
            rows.append({"E": E, "key": key, "variation": c["var"], "family": family_of(c["var"], fams), "scale": c["scale"],
                         "class": cls, "init_shift_mm": init_mm, "approach_dq_rad": dq["approach"], "grasp_dq_rad": dq["grasp"],
                         "handover_dq_rad": dq["handover"], "handover_can_mm": dcan_ho, "gripper_event_shift_frames": grip_shift,
                         "export_ok": bool(ck.get("ok", False)), "evidence_complete": bool(ck.get("evidence_complete", False)),
                         "in_frozen_Gp_matched_evidence_complete_dataset": (E, key) in in_ec_dataset,
                         "ec_dataset_episode_index": in_ec_dataset.get((E, key)), "candidate_hash": c["hash"]})
    # summaries
    tab = collections.defaultdict(collections.Counter)
    tab_ds = collections.defaultdict(collections.Counter)
    fam_vs_cls = collections.defaultdict(collections.Counter)
    for r in rows:
        tab[r["class"]][r["E"]] += 1
        fam_vs_cls[r["family"]][r["class"]] += 1
        if r["in_frozen_Gp_matched_evidence_complete_dataset"]:
            tab_ds[r["class"]][r["E"]] += 1
    out = {"sources": src_check, "thresholds": TH, "candidates": rows,
           "accepted_Gp_by_class_and_source": {k: dict(sorted(v.items())) for k, v in tab.items()},
           "in_frozen_ec_dataset_by_class_and_source": {k: dict(sorted(v.items())) for k, v in tab_ds.items()},
           "family_vs_measured_class": {k: dict(v) for k, v in fam_vs_cls.items()},
           "inputs_sha256": {"v3_1_config": sha(V31 / "configs/v3_1.json"), "export_checks": sha(V31 / "outputs/dataset_export/export_checks.json"),
                             "Gp_ec_manifest": sha(V31 / "outputs/dataset_export/Gp_matched_evidence_complete_manifest.json")}}
    (OUT / "family_audit.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# v4B Task 1 - measured augmentation classes of accepted v3.1 G+ candidates", "",
         "| v3 family | position_only | handover_only | both | other |", "|---|---|---|---|---|"]
    for f, v in sorted(fam_vs_cls.items()):
        L.append(f"| {f} | {v['position_only']} | {v['handover_only']} | {v['both']} | {v['other']} |")
    L += ["", "In the frozen `sfa_v31_Gp_matched_evidence_complete` dataset, by source episode:", "",
          "| class | " + " | ".join(f"ep{E}" for E in range(10)) + " | total |", "|---|" + "---|" * 11]
    for k in ("position_only", "handover_only", "both", "other"):
        v = tab_ds.get(k, {})
        L.append(f"| {k} | " + " | ".join(str(v.get(E, 0)) for E in range(10)) + f" | {sum(v.values())} |")
    oth = [r for r in rows if r["class"] == "other"]
    if oth:
        L += ["", "`other` candidates (not used):", ""] + [f"- ep{r['E']} {r['key']}: init {r['init_shift_mm']:.1f} mm, approach dq {r['approach_dq_rad']:.4f}, "
                                                          f"handover dq {r['handover_dq_rad']:.4f}, can {r['handover_can_mm']:.1f} mm, grip shift {r['gripper_event_shift_frames']}" for r in oth]
    L += ["", f"Identity candidate == recorded source state: max |diff| {max(v['identity_state_equals_recorded_state_max_abs'] for v in src_check.values()):.2e}"]
    (OUT / "family_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
