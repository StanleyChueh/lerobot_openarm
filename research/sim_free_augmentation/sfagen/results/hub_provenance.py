"""Map each published sfagen dataset (tag v3.0) to the git commit whose sfagen/ files equal the uploaded code snapshot."""
import hashlib, json, subprocess, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); T = dict(repo_type="dataset"); WT = str(Path(__file__).resolve().parents[4])  # repository root
def git_sha(commit, rel):
    p = subprocess.run(["git", "-C", WT, "show", f"{commit}:research/sim_free_augmentation/sfagen/{rel}"], capture_output=True)
    return hashlib.sha256(p.stdout).hexdigest() if p.returncode == 0 else None
main4 = [f"ethanCSL/openarm_{t}_lerobot_{k}_10_3cam_v00" for t in ("pringles", "insertbattery") for k in ("real_to_sim", "real_and_sim")]
budget = [f"ethanCSL/openarm_pringles_lerobot_real_to_sim_{n}_3cam_v00" for n in range(1, 10)]
out = {}
for R in main4 + budget:
    commit = "50832b3" if R in main4 else "5d81f2f"
    refs = api.list_repo_refs(R, **T); v3 = {t.name: t.target_commit for t in refs.tags}["v3.0"]
    files = [f for f in api.list_repo_files(R, revision="v3.0", **T) if f.startswith("sfagen/")]
    rows = {}
    for f in files:
        i = api.get_paths_info(R, [f], revision="v3.0", **T)[0]
        h = i.lfs.sha256 if i.lfs else hashlib.sha256(open(hf_hub_download(R, f, revision="v3.0", **T), "rb").read()).hexdigest()
        rows[f[len("sfagen/"):]] = h == git_sha(commit, f[len("sfagen/"):])
    full = subprocess.run(["git", "-C", WT, "rev-parse", commit], capture_output=True, text=True).stdout.strip()
    out[R] = {"v3.0": v3, "private": api.dataset_info(R).private, "code_commit": full, "code_files": len(rows), "files_equal_to_commit": sum(rows.values()),
              "not_equal": sorted(k for k, v in rows.items() if not v)}
    print(R.split("/")[1], v3[:8], commit, f"{out[R]['files_equal_to_commit']}/{len(rows)}", out[R]["not_equal"], flush=True)
note = {"generated": "2026-11-11, read-only Hub query (huggingface_hub list_repo_refs/list_repo_files/get_paths_info)",
        "rule": "a dataset's code_commit is exact when files_equal_to_commit == code_files (every uploaded sfagen/ file is byte-identical)",
        "visibility": "the nine source-budget repos are public by the owner's decision (confirmed 2026-11-11), although tools_budget_publish.py creates them private",
        "known_data_issues": "see results/published_check_2026-11-11/published_check.json: clipped action labels and one missing battery wrist frame (episode 91; mixed 101)"}
json.dump({"_about": note, "datasets": out}, open(sys.argv[1], "w"), indent=1)
