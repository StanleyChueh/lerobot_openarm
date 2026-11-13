#!/bin/bash
# build -> verify -> quality -> independent check for one export. usage: build_all.sh <strict|loose> <copied|right_wrist> <gpu>
S=/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen; R=/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run
TH=$1; C=$2; G=$3; SUF=$([ $C = right_wrist ] && echo _wristrender); CFG=configs/openarm_pringles_v03_${TH}${SUF}.yaml
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 MPLCONFIGDIR=/tmp/claude-1000/mpl /home/stanley/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
cd $S; L=$R/logs/export_${TH}_${C}
$PY -m sfagen build $CFG --device cuda:$G --repo-id ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam${SUF}_v03_${TH} --mixed-repo-id ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam${SUF}_v03_${TH} > $L.build.log 2>&1 || { echo BUILD FAIL $TH $C; exit 1; }
$PY -m sfagen verify $CFG > $L.verify.log 2>&1 || { echo VERIFY FAIL $TH $C; exit 1; }
$PY tools/quality_report.py $CFG results/v03/quality_${TH}_${C} > $L.quality.log 2>&1 || { echo QUALITY FAIL $TH $C; exit 1; }
echo DONE $TH $C
