#!/usr/bin/env bash
# setup + build + verify/quality gate for the given few-source datasets. usage: tools/fewsrc_build_all.sh "1 4" [--wrist]
set -euo pipefail
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ${PY_BIN:-$HOME/Stanley_ws/lerobot_openarm/.venv/bin/python} -B"
W=${2:-}; V=""; R=""; [ "$W" = "--wrist" ] && V="_wristrender" && R="_wristrender"
for n in $1; do
  $PY tools/fewsrc_build.py setup $n $W
  $PY -m sfagen build configs/pringles_fewsrc$n$V.yaml --repo-id ethanCSL/openarm_pringles_lerobot_real_to_sim_${n}_3cam${R}_v01
  tools/fewsrc_qa.sh $n $V
done
