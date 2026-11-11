"""Set up one sfagen work dir + config per source budget N (sources 0..N-1) from the dense-grid candidates/renders, with
checks.json restricted to the selected episodes and the speed/accel/table fields recomputed against the N-source limits.
usage: tools_budget_build.py N"""
import json, os, shutil, sys
from pathlib import Path
import yaml
S = Path(__file__).resolve().parent; D = S / "configs/work/openarm_pringles_handover_dense"
n = int(sys.argv[1]); sel = json.loads((S / "logs/budget_selection.json").read_text())[str(n)]
W = S / f"configs/work/pringles_src{n}"
if W.exists():
    shutil.rmtree(W)
W.mkdir(parents=True)
for f in ("calibration.json", "table_height.json", "lowest_arm.json"):
    shutil.copy(D / f, W / f)
os.symlink(os.path.realpath(D / "cache"), W / "cache"); os.symlink(D / "body", W / "body")
for e in sel["sources"]:
    (W / f"candidates/ep{e}").mkdir(parents=True)
    full = json.loads((D / f"candidates/ep{e}/checks.json").read_text()); out = {}
    for ee, key, env in sel["episodes"]:
        if ee != e:
            continue
        c = dict(full[key]); c.update({"speed_ratio_to_real_max": env["speed_ratio_to_subset_max"], "accel_ratio_to_real_max": env["accel_ratio_to_subset_max"],
                                       "real_lowest_arm_z_m": env["subset_lowest_arm_z_m"], "envelope_sources": env["envelope_sources"], "pass": True})
        out[key] = c; os.symlink(D / f"candidates/ep{e}/{key}.npz", W / f"candidates/ep{e}/{key}.npz")
    (W / f"candidates/ep{e}/checks.json").write_text(json.dumps(out, indent=1, default=float))
t = (S / "configs/openarm_pringles_handover_dense.yaml").read_text(); c = yaml.safe_load(t)
c["source"]["episodes"] = sel["sources"]; c["work_dir"] = str(W)
(S / f"configs/pringles_src{n}.yaml").write_text(f"# Source-budget dataset: {n} real source demo(s) {sel['sources']}; dense grid, interaction shifts <= 4.3 cm, "
                                                  f"replay limits from these {n} demos only.\n" + yaml.safe_dump(c, sort_keys=False, width=160))
print("ready", n, len(sel["episodes"]), "episodes from sources", sel["sources"])
