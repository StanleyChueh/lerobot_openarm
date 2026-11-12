"""Read-only Hub check of published v01 repos: v3.0 tag = main, every local export file present with the same size.
usage: hubcheck_v01.py N [N ...]   (N or N_wristrender) -> results/datasets_v01/hubcheck.json (merged)"""
import json, sys
from pathlib import Path
from huggingface_hub import HfApi
S = Path(__file__).resolve().parents[1]; api = HfApi(); out = S / "results/datasets_v01/hubcheck.json"
res = json.loads(out.read_text()) if out.exists() else {}
for tok in sys.argv[1:]:
    n, _, v = tok.partition("_"); v = f"_{v}" if v else ""; R = f"ethanCSL/openarm_pringles_lerobot_real_to_sim_{n}_3cam{v}_v01"
    root = S / f"configs/work/pringles_fewsrc{n}{v}/dataset"; local = {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}
    info = api.dataset_info(R, files_metadata=True); remote = {f.rfilename: f.size for f in info.siblings}
    tags = {t.name: t.target_commit for t in api.list_repo_refs(R, repo_type="dataset").tags}
    bad = [k for k, s in local.items() if remote.get(k) != s]
    res[R] = {"main": info.sha, "v3.0": tags.get("v3.0"), "tag_ok": tags.get("v3.0") == info.sha, "export_files": len(local), "missing_or_size_mismatch": bad,
              "private": info.private, "pass": tags.get("v3.0") == info.sha and not bad and not info.private}
    print(R, res[R]["pass"], len(local), "files", bad[:3])
out.write_text(json.dumps(res, indent=1))
