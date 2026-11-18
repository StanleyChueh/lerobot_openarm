"""R05 driver: write config variants (B = prompt cut by depth + mask repair; C = B + fit_focal false) and prepare
fresh work dirs. B reuses the existing reconstruction (symlinked recon caches, calibration, table, candidates) so that
only the masks change; C starts from the camera guess. Usage: r05_setup.py RUN_DIR"""
import sys, yaml, shutil
from pathlib import Path
S = Path(__file__).resolve().parents[2]; RUN = Path(sys.argv[1]).resolve(); out = S / "simbench/configs/r05"
JOBS = {"openarm": S / "simbench/configs/r03/openarm_v03r2_loose_v1.yaml"}
JOBS.update({p.stem: p for p in sorted((S / "simbench/configs/robotwin").glob("*_*.yaml")) if not p.stem.endswith("_base") and not p.stem.startswith("_")})
for name, p in JOBS.items():
    c = yaml.safe_load(p.read_text()); old = Path(c["work_dir"])
    c.setdefault("reconstruction", {}).update(prompt_object_cut="depth", mask_repair=0.6)
    b = dict(c); b["work_dir"] = str(RUN / "B" / name); (out / f"{name}_B.yaml").write_text(yaml.safe_dump(b, sort_keys=False))
    w = RUN / "B" / name; (w / "cache").mkdir(parents=True, exist_ok=True)
    for f in ("calibration.json", "table_height.json", "lowest_arm.json"):
        if (old / f).exists() and not (w / f).exists():
            shutil.copy(old / f, w / f)
    for f in sorted((old / "cache").glob("recon_ep*.npz")):
        if not (w / "cache" / f.name).exists():
            (w / "cache" / f.name).symlink_to(f)
    if (old / "candidates").exists() and not (w / "candidates").exists():
        (w / "candidates").symlink_to(old / "candidates")
    if name != "openarm":
        cc = dict(c); cc["calibration"] = {**c.get("calibration", {}), "fit_focal": False}; cc["work_dir"] = str(RUN / "C" / name)
        (out / f"{name}_C.yaml").write_text(yaml.safe_dump(cc, sort_keys=False))
    print(name, "episodes", c["source"].get("episodes"), "old work", old)
for p in sorted((S / "simbench/configs/robotwin").glob("*_base.yaml")):       # C rigs: camera fit with trusted intrinsics
    c = yaml.safe_load(p.read_text()); c["calibration"] = {**c.get("calibration", {}), "fit_focal": False}
    c.setdefault("reconstruction", {}).update(prompt_object_cut="depth", mask_repair=0.6)
    c["work_dir"] = str(RUN / "C" / (p.stem.replace("_base", "") + "_rig")); (out / f"{p.stem}_C.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
