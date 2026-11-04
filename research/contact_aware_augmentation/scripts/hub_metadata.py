"""Read-only Hub metadata snapshot (GET requests only: dataset_info + list_repo_refs). Never uploads or writes to the Hub.

    env -u PYTHONPATH .venv-v2/bin/python -I research/contact_aware_augmentation/scripts/hub_metadata.py <out.json>
"""

import datetime as dt
import json
import sys

from huggingface_hub import HfApi

REPOS = ["ethanCSL/openarm_pringles_lerobot_real_10_v00", "ethanCSL/openarm_insertbattery_lerobot_real_v02",
         "ethanCSL/openarm_pringles_lerobot_real_v00", "ethanCSL/openarm_insertbattery_lerobot_real_v01",
         "ethanCSL/openarm_insertbattery_lerobot_real_v00"]


def main(out):
    api, res = HfApi(), {"queried_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    for r in REPOS:
        try:
            i = api.dataset_info(r, files_metadata=True)
            refs = api.list_repo_refs(r, repo_type="dataset")
            res[r] = {"sha": i.sha, "last_modified": str(i.last_modified),
                      "branches": {b.name: b.target_commit for b in refs.branches},
                      "tags": {t.name: t.target_commit for t in refs.tags},
                      "files": {s.rfilename: {"size": s.size, "lfs_sha256": (s.lfs.sha256 if s.lfs else None),
                                              "blob_id": s.blob_id} for s in i.siblings}}
        except Exception as e:  # noqa: BLE001
            res[r] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print({k: (v.get("sha") if isinstance(v, dict) else v) for k, v in res.items()})


if __name__ == "__main__":
    main(sys.argv[1])
