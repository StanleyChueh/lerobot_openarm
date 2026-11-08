"""Content-hash manifest of large SD10-G1 artifacts kept outside Git (STORE) -> outputs/store_manifest.json."""
from __future__ import annotations

import json

import g1lib as G


def main():
    files = sorted(p for p in G.STORE.rglob("*") if p.is_file())
    man = {"store": str(G.STORE), "n_files": len(files), "total_bytes": sum(p.stat().st_size for p in files),
           "files": {str(p.relative_to(G.STORE)): {"bytes": p.stat().st_size, "sha256": G.sha(p)} for p in files},
           "rgb_videos": "none - G1-B RGB synthesis not launched (G1-A decision not CONTINUE)"}
    (G.OUT / "store_manifest.json").write_text(json.dumps(man, indent=1))
    print(man["n_files"], round(man["total_bytes"] / 2**30, 2), "GiB")


if __name__ == "__main__":
    main()
