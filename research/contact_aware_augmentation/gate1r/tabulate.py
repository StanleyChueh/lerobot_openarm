"""Tabulate the frozen Gate 1R benchmark from outputs/gate1r/samples.jsonl (no re-scoring).

    python3 research/contact_aware_augmentation/gate1r/tabulate.py
"""
import json
from pathlib import Path

O = Path(__file__).resolve().parents[1] / "outputs" / "gate1r"
rows = [json.loads(l) for l in open(O / "samples.jsonl")]
METHODS = ["A", "B", "C", "D", "combined", "combined_minus_A", "combined_minus_B", "combined_minus_C", "combined_minus_D", "gate1_C", "gate1_C_rel"]


def cell(sel, m):
    p = sum(r["st"][m] == "PASS" for r in sel); f = sum(r["st"][m] == "FAIL" for r in sel); u = sum(r["st"][m] == "UNKNOWN" for r in sel)
    return f"{f}/{u}/{p}"


names = []
for r in rows:
    if (r["label"], r["family"], r["name"]) not in names:
        names.append((r["label"], r["family"], r["name"]))
table = {f"{l}|{fam}|{n}": {m: cell([r for r in rows if r["name"] == n], m) for m in METHODS} for l, fam, n in names}
fam_table = {}
for l, fam, _ in names:
    k = f"{l}|{fam}"
    if k not in fam_table:
        sel = [r for r in rows if r["label"] == l and r["family"] == fam]
        fam_table[k] = {"n": len(sel)} | {m: cell(sel, m) for m in METHODS}
calib = {r["name"]: [] for r in rows}
for r in rows:
    calib[r["name"]].append(round(r["B"].get("giver_calib", {}).get("fit_rms_px", float("nan")), 1))
reasons = {}
for r in rows:
    if r["label"] == "invalid" and r["st"]["combined"] == "UNKNOWN":
        why = sorted(set(([r["B"]["giver"].get("reason")] if r["B"].get("giver") else []) + ([r["B"]["receiver"].get("reason")] if r["B"].get("receiver") else []) + r["C"].get("unknown", [])) - {None})
        reasons.setdefault(r["family"], {}).setdefault(r["name"], []).append({"fold": r["fold"], "why": why})
out = {"cell_format": "FAIL/UNKNOWN/PASS counts over the 3 held-out folds (or n samples for family rows)",
       "per_variant": table, "per_family": fam_table, "giver_calibration_rms_px": calib, "invalid_unknown_reasons": reasons}
(O / "summary_tables.json").write_text(json.dumps(out, indent=1))
for k, v in fam_table.items():
    print(f"{k:42s} n={v['n']:3d} " + " ".join(f"{m}={v[m]}" for m in ("A", "B", "C", "D", "combined", "gate1_C", "gate1_C_rel")))
