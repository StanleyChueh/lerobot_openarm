"""E: publish the few-source candidate datasets to NEW Hub repos (never overwrites: refuses if the repo exists).
One commit per repo: LeRobot export + README card + quality_report.json + provenance.json (git SHA, code hashes) + the
sfagen code/configs used; then tag v3.0 (LeRobot loads it) and verify the tag. Every export and its quality gate are
checked before the first write. DRY_RUN=1 prints the plan only.
usage: push_fewsrc.py N [N ...]"""
import json, os, sys, tempfile
from pathlib import Path
from huggingface_hub import CommitOperationAdd
from _provenance import HERE as S, upload_provenance
from push_unified import export_files
T = dict(repo_type="dataset")
REPO = "ethanCSL/openarm_pringles_lerobot_real_to_sim_{n}_3cam_v01"


def job(n):
    W = S / f"configs/work/pringles_fewsrc{n}"; q = json.loads((S / f"results/datasets_v01/fewsrc{n}/quality_report.json").read_text())
    if not q["pass"]:
        raise RuntimeError(f"N={n}: quality gate failed; not publishing")
    card = S / f"results/datasets_v01/fewsrc{n}/README.md"
    if not card.exists():
        raise RuntimeError(f"missing dataset card {card}")
    code = sorted(S.glob("sfagen/*.py")) + sorted(S.glob("configs/*.yaml")) + [S / "README.md"]
    prov = Path(tempfile.mkdtemp()) / "provenance.json"; prov.write_text(json.dumps(upload_provenance(code), indent=1))
    extra = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card)),
             CommitOperationAdd(path_in_repo="quality_report.json", path_or_fileobj=str(S / f"results/datasets_v01/fewsrc{n}/quality_report.json")),
             CommitOperationAdd(path_in_repo="provenance.json", path_or_fileobj=str(prov))]
    extra += [CommitOperationAdd(path_in_repo=f"sfagen/{p.relative_to(S)}", path_or_fileobj=str(p)) for p in code]
    extra += [CommitOperationAdd(path_in_repo=f"assets/{p.name}", path_or_fileobj=str(p)) for p in sorted((S / f"results/datasets_v01/fewsrc{n}/sheets").glob("*.jpg"))[:6]]
    return REPO.format(n=n), export_files(W / "dataset"), extra


def publish(api, R, new, extra, dry):
    if api.repo_exists(R, **T):
        raise RuntimeError(f"{R} already exists; this tool never overwrites (tag + use push_unified.publish instead)")
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in new.items()] + extra
    print(R, "| new public repo | upload", len(ops), "files", flush=True)
    if dry:
        return None
    api.create_repo(R, private=False, exist_ok=False, **T)
    c = api.create_commit(R, operations=ops, commit_message="sfagen v01 few-source candidate (F1W <= 11 cm, rendered wrist views where the wrist gate passed); quality-gated", **T)
    api.create_tag(R, tag="v3.0", revision=c.oid, **T)
    if {t.name: t.target_commit for t in api.list_repo_refs(R, **T).tags}.get("v3.0") != c.oid:
        raise RuntimeError(f"{R}: v3.0 tag does not point at {c.oid}")
    print("   main", c.oid, flush=True); return c.oid


if __name__ == "__main__":
    from huggingface_hub import HfApi
    api = HfApi(); J = [job(int(n)) for n in sys.argv[1:]]          # every export + gate checked before any write
    res = {R: publish(api, R, new, extra, os.environ.get("DRY_RUN") == "1") for R, new, extra in J}
    print(json.dumps(res, indent=1))
