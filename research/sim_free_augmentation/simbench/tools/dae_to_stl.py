"""Convert the COLLADA visual meshes referenced by a URDF to binary STL (sfagen reads STL without trimesh) and rewrite
the URDF to use them. Runs in the separate mesh venv (artifacts/venvs/meshtools: trimesh 4.5.3, pycollada 0.8).
Usage: dae_to_stl.py in.urdf out.urdf    (meshes written to <urdf dir>/sfagen_stl/)"""
import sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh

src, out = Path(sys.argv[1]), Path(sys.argv[2]); base = src.parent; dst = base / "sfagen_stl"; dst.mkdir(exist_ok=True)
tree = ET.parse(src); report = {}
for m in tree.getroot().iter("mesh"):
    f = m.get("filename")
    if not f.lower().endswith(".dae"):
        continue
    o = dst / (Path(f).stem + ".stl")
    if f not in report:
        mesh = trimesh.load(base / f, force="mesh")  # applies the COLLADA unit and scene transforms
        mesh.export(o); stl = (base / f).with_suffix(".STL")
        ref = trimesh.load(stl, force="mesh").bounds.tolist() if stl.exists() else None
        report[f] = {"faces": len(mesh.faces), "bounds": np.round(mesh.bounds, 4).tolist(), "collision_stl_bounds": ref and np.round(ref, 4).tolist()}
    m.set("filename", str(o.relative_to(base)))
tree.write(out)
for k, v in report.items():
    print(k, v)
