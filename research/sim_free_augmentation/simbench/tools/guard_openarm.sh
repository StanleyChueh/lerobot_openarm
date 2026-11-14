#!/usr/bin/env bash
# OpenArm bit-identity guard (SIM_BENCHMARK_PLAN §5.2): every enabling change E1-E8 must leave OpenArm output unchanged.
#   1. unittest discover -s tests passes;
#   2. Pringles v03r2 source episode 5 regenerates bit-identical candidate files (candidate_sha256) and
#   3. bit-identical rendered body frames, compared with artifacts/sfagen_v03r2_run (code 49957aa = c737e8c run).
# Usage: guard_openarm.sh <label>   (writes artifacts/simbench_runs/guard_<label>/ in this worktree; never reuses a dir)
set -euo pipefail
LABEL=${1:?label}; EP=${GUARD_EP:-5}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
SF=$ROOT/research/sim_free_augmentation/sfagen
REF=${GUARD_REF:-$HOME/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03r2_run}
W=$ROOT/artifacts/simbench_runs/guard_$LABEL
[ -e "$W" ] && { echo "refusing to reuse $W"; exit 2; }
mkdir -p "$W"; cp "$REF"/gen/{calibration.json,table_height.json,lowest_arm.json} "$W"/
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 $HOME/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
cd "$SF"
{ git -C "$ROOT" rev-parse HEAD; git -C "$ROOT" status --short -- research/sim_free_augmentation; } > "$W/code_state.txt"
sha256sum sfagen/*.py >> "$W/code_state.txt"
$PY -m unittest discover -s tests > "$W/tests.log" 2>&1 || { tail -20 "$W/tests.log"; echo "GUARD FAIL: tests"; exit 1; }
tail -3 "$W/tests.log"
$PY -m sfagen generate configs/openarm_pringles_v03r2_gen.yaml --episodes "$EP" --work-dir "$W" > "$W/generate.log" 2>&1
$PY -m sfagen render configs/openarm_pringles_v03r2_gen.yaml --episodes "$EP" --work-dir "$W" > "$W/render.log" 2>&1
$PY - "$REF" "$W" "$EP" <<'EOF' | tee "$W/guard_result.json"
import hashlib, json, sys, pathlib
import numpy as np
ref, w, ep = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
a = json.load(open(ref / f"gen/candidates/ep{ep}/generation_manifest.json"))["candidate_sha256"]
b = json.load(open(w / f"candidates/ep{ep}/generation_manifest.json"))["candidate_sha256"]
cand_bad = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
frames = {}
for f in sorted((w / "body").glob(f"{ep}_*.npy")):
    r = ref / "all" / "body" / f.name  # renders the v03r2 exports used; body_base is an older pre-Amendment-5 render
    frames[f.name] = r.exists() and hashlib.sha256(np.load(f).tobytes()).hexdigest() == hashlib.sha256(np.load(r).tobytes()).hexdigest()
ok = not cand_bad and len(a) > 0 and len(frames) > 0 and all(frames.values())
print(json.dumps({"episode": int(ep), "candidates": len(a), "candidate_mismatch": cand_bad, "body_files": len(frames),
                  "body_identical": sum(frames.values()), "body_mismatch": [k for k, v in frames.items() if not v], "pass": ok}, indent=1))
sys.exit(0 if ok else 1)
EOF
