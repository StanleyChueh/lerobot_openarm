#!/usr/bin/env bash
# D: verify + quality gate for one few-source dataset. usage: tools/fewsrc_qa.sh N [_wristrender]   (from research/sim_free_augmentation/sfagen)
set -euo pipefail
N=$1; V=${2:-}; PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ${PY_BIN:-$HOME/Stanley_ws/lerobot_openarm/.venv/bin/python} -B"
POOL=$HOME/.cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d/data
$PY -m sfagen verify configs/pringles_fewsrc$N$V.yaml > /dev/null
(cd tools && $PY quality_report.py configs/pringles_fewsrc$N$V.yaml ../results/datasets_v01/fewsrc$N$V --pool-data "$POOL" --exclude 0,1,2,3,4,5,6,7,8,9 > /dev/null)
cp configs/work/pringles_fewsrc$N$V/verification.json results/datasets_v01/fewsrc$N$V/verification.json
$PY -c "import json;d=json.load(open('results/datasets_v01/fewsrc$N$V/quality_report.json'));print('N=$N$V pass',d['pass'],d['episodes'],'eps',d['frames'],'frames cov',d['coverage']['within_2cm'],'wrist valid',{k[-15:]:round(v,2) for k,v in d['images']['wrist_view_valid_frac'].items()})"
