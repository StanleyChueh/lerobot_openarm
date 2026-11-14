"""Yield table with the first failed gate per candidate (gate order = sfagen checks.py pass formula; fast rejections keep
their recorded order). Usage: s1_yield.py RUN_DIR CONFIG_DIR OUT.json   -> prints a Markdown table, writes JSON."""
import collections, json, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.checks import DEFAULTS


def first_gate(v, cfg, env):
    if "error" in v and "fast_rejection_reasons" not in v:
        return "no_schedule" if "schedule" in v["error"] else "error"
    if v.get("fast_rejection_reasons"):
        return v["fast_rejection_reasons"][0]
    g = [("nonfinite", v["finite_arrays"]), ("action_alignment", v["action_reference_error_max"] <= 2e-6),
         ("ik", v["ik_pos_err_mm_max"] <= cfg["ik_pos_mm"] and v["ik_rot_err_deg_max"] <= cfg["ik_rot_deg"]),
         ("ik_blend", v["ik_pos_err_mm_max_blend"] <= cfg["ik_blend_pos_mm"] and v["ik_rot_err_deg_max_blend"] <= cfg["ik_blend_rot_deg"]),
         ("limits", v["limit_violations"] == 0), ("speed", v["speed_ratio_to_real_max"] <= cfg["speed_ratio"]),
         ("accel", v["accel_ratio_to_real_max"] <= cfg["accel_ratio"]),
         ("mesh_clearance", v["min_mesh_distance_mm"] >= min(v["source_min_mesh_distance_same_frames_mm"], 1000 * cfg["clearance_floor_m"]) - 1.0),
         ("hands", v.get("min_hand_distance_mm", 1e9) >= v.get("source_min_hand_distance_same_frames_mm", 0) - 1000 * cfg["hand_tol_m"]),
         ("table", v["lowest_arm_z_m"] >= env["min_z_m"] - cfg["table_tol_m"]), ("start_state", v["start_state_err_rad"] <= cfg["start_tol_rad"]),
         ("untouched", v["untouched_bit_identical"]), ("rejoin", v["identical_after_rejoin"]),
         ("object", v.get("placement", {}).get("ok", True) if isinstance(v.get("placement"), dict) else True),
         ("duration/speed_similarity", v["timing_ok"])]
    return next((n for n, ok in g if not ok), "object_other")


run, cdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); rows = []
for cj in sorted(run.glob("*/*/candidates/ep*/checks.json")):
    g = cj.parents[2]; task = g.parent.name
    if g.name.endswith("_export"):
        continue
    cfg = {**DEFAULTS, **yaml.safe_load((cdir / f"{task}_{g.name}.yaml").read_text()).get("checks", {})}
    env = json.loads((cj.parent / "generation_manifest.json").read_text())["replay_envelope"]
    for k, v in json.loads(cj.read_text()).items():
        rows.append({"task": task, "group": g.name, "episode": int(cj.parent.name[2:]), "key": k, "family": v.get("family"),
                     "pass": v["pass"], "first_failed_gate": None if v["pass"] else first_gate(v, cfg, env),
                     "rendered": (g / "body" / f"{cj.parent.name[2:]}_{k}.npy").exists()})
out.write_text(json.dumps(rows, indent=1))
print("| task | family | candidates | pass | yield | first failed gate (count) |\n|---|---|---|---|---|---|")
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["task"], r["family"])].append(r); agg[(r["task"], "all")].append(r)
for (t, f), rs in sorted(agg.items()):
    n, p = len(rs), sum(r["pass"] for r in rs); c = collections.Counter(r["first_failed_gate"] for r in rs if not r["pass"])
    print(f"| {t} | {f} | {n} | {p} | {100 * p / max(n, 1):.0f} % | " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + " |")
