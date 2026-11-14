"""Pipeline-v0 selection for one task (SIM_BENCHMARK_PLAN §6; same rule as v03 rev 2 finalize without a distortion cap):
every passing + rendered non-grid candidate (interaction location / timing) + at most 2 lowest-distortion candidates per
grid target, pooled over the task's episode groups. Then, per group, an export work dir `<group>_export` with copies of
the rig files, symlinks to the group's cache/ and candidates/, and hard links of only the selected renders, so
`sfagen build` (which exports every passing candidate with a current render) exports exactly the selection.
Usage: select_export.py TASK_RUN_DIR   -> writes selection.json and <group>_export/ dirs (refuses to overwrite)"""
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.distortion import score_file

run = Path(sys.argv[1]).resolve(); rows = []
groups = sorted(p for p in run.iterdir() if p.is_dir() and (p / "candidates").exists() and not p.name.endswith("_export"))
for g in groups:
    for cj in sorted((g / "candidates").glob("ep*/checks.json")):
        e = int(cj.parent.name[2:])
        for k, v in json.loads(cj.read_text()).items():
            f = g / "body" / f"{e}_{k}.json"
            if v["pass"] and f.exists():
                rows.append({"group": g.name, "source": e, "key": k, "family": v["family"], "score": score_file(f),
                             "shift_cm": 100 * float(sum(x * x for x in v["shift_m"][:2]) ** .5)})
keep = [r for r in rows if r["family"] != "object_workspace"]; per = {}
for r in sorted((r for r in rows if r["family"] == "object_workspace"), key=lambda r: (r["score"], r["shift_cm"])):
    if len(per.setdefault(r["key"], [])) < 2:
        per[r["key"]].append(r)
keep += [r for v in per.values() for r in v]
assert not (run / "selection.json").exists(), "selection exists"
(run / "selection.json").write_text(json.dumps({"rule": "all non-grid + <= 2 lowest-distortion per grid target (no cap)",
                                                "n": len(keep), "grid_targets": len(per), "rendered": len(rows), "selection": keep}, indent=1))
for g in groups:
    d = run / f"{g.name}_export"; d.mkdir(); (d / "body").mkdir()
    for f in ("calibration.json", "table_height.json", "lowest_arm.json"):
        shutil.copy2(g / f, d / f)
    for s in ("cache", "candidates"):
        os.symlink(g / s, d / s)
    for r in (r for r in keep if r["group"] == g.name):
        for ext in (".npy", ".json", ".fingerprint"):
            os.link(g / "body" / f"{r['source']}_{r['key']}{ext}", d / "body" / f"{r['source']}_{r['key']}{ext}")
print(json.dumps({"rendered": len(rows), "selected": len(keep), "grid_targets": len(per),
                  "by_group": {g.name: sum(r["group"] == g.name for r in keep) for g in groups}}))
