"""R06 driver: write a config variant and a fresh work dir that reuses an existing reconstruction (symlinked recon
caches, calibration, table, candidates), so that only the masks change.
Usage: r06_setup.py RUN_DIR TAG key.sub=value[,key.sub=value...] NAME=SOURCE_CONFIG ...   -> simbench/configs/r06/NAME_TAG.yaml"""
import re, sys, yaml, shutil
from pathlib import Path
S = Path(__file__).resolve().parents[2]; RUN = Path(sys.argv[1]).resolve(); tag = sys.argv[2]; out = S / "simbench/configs/r06"; out.mkdir(exist_ok=True)
for job in sys.argv[4:]:
    name, src = job.split("=", 1); c = yaml.safe_load(Path(src).read_text()); old = Path(c["work_dir"])
    for kv in re.split(r",(?![^\[]*\])", sys.argv[3]):              # commas inside [..] belong to the value
        k, v = kv.split("="); a, b = k.split("."); c.setdefault(a, {})[b] = yaml.safe_load(v)
    w = RUN / tag / name; c["work_dir"] = str(w); (out / f"{name}_{tag}.yaml").write_text(yaml.safe_dump(c, sort_keys=False)); (w / "cache").mkdir(parents=True, exist_ok=True)
    for f in ("calibration.json", "table_height.json", "lowest_arm.json"):
        if (old / f).exists() and not (w / f).exists():
            shutil.copy(old / f, w / f)
    for f in sorted((old / "cache").glob("recon_ep*.npz")):
        if not (w / "cache" / f.name).exists():
            (w / "cache" / f.name).symlink_to(f.resolve())
    if (old / "candidates").exists() and not (w / "candidates").exists():
        (w / "candidates").symlink_to((old / "candidates").resolve())
    print(name, tag, "->", w)
