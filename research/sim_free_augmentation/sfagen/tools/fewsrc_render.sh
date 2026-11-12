#!/usr/bin/env bash
# Render the selected few-source variants (results/fewsrc/render_keys.json) for the given demos on one GPU.
# usage: [CFG=configs/openarm_pringles_fewsrc_wristrender.yaml] tools/fewsrc_render.sh DEVICE DEMO [DEMO ...]   (run from research/sim_free_augmentation/sfagen)
set -euo pipefail
DEV=$1; shift
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ${PY_BIN:-$HOME/Stanley_ws/lerobot_openarm/.venv/bin/python} -B"
for e in "$@"; do
  KEYS=$($PY -c "import json;print(','.join(json.load(open('results/fewsrc/render_keys.json'))['$e']))")
  $PY -m sfagen render "${CFG:-configs/openarm_pringles_fewsrc.yaml}" --episodes "$e" --keys "$KEYS" --device "$DEV"
done
