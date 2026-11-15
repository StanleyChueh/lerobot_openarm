#!/usr/bin/env bash
# S1 driver for one RoboTwin task after `calibrate --search` and `table` on <task>_base.yaml:
# source positions -> configs with frozen positions -> generate -> render -> select -> build -> verify -> video.
# Usage: s1_task.sh TASK GPU RUN_DIR      (refuses existing exports; logs in RUN_DIR/logs)
set -euo pipefail
T=$1; G=$2; RUN=$(realpath "$3"); HERE=$(cd "$(dirname "$0")/.." && pwd); SF=$HERE/../sfagen; C=$HERE/configs/robotwin; L=$RUN/logs
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$G $HOME/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
GRPS=$(ls $C | sed -n "s/^${T}_\(.*\)\.yaml$/\1/p" | grep -v '^base$')
mkdir -p $RUN/positions
cd $SF
# configs must point at THIS run dir before anything reads them (s1c bug: positions ran against the previous run's work dirs)
$PY $C/make_configs.py $RUN $RUN/no_positions_yet $T > $L/positions_$T.log 2>&1
$PY $HERE/tools/source_positions.py $RUN/positions/$T.json $(for g in $GRPS; do echo $C/${T}_$g.yaml; done) >> $L/positions_$T.log 2>&1
$PY $C/make_configs.py $RUN $RUN/positions $T >> $L/positions_$T.log 2>&1
for g in $GRPS; do grep -q "work_dir: $RUN/$T/$g\$" $C/${T}_$g.yaml && for f in calibration.json table_height.json lowest_arm.json; do
  [ -f $RUN/$T/$g/$f ] || { echo "missing $RUN/$T/$g/$f" >&2; exit 3; }; done || { echo "config $T/$g not pointing at $RUN" >&2; exit 3; }; done
for g in $GRPS; do
  $PY -m sfagen generate $C/${T}_$g.yaml --device cuda:0 > $L/generate_${T}_$g.log 2>&1
  $PY -m sfagen render $C/${T}_$g.yaml --device cuda:0 > $L/render_${T}_$g.log 2>&1
done
$PY $HERE/tools/select_export.py $RUN/$T > $L/select_$T.log 2>&1
for g in $GRPS; do
  W=$RUN/$T/${g}_export
  [ -n "$(ls $W/body)" ] || { echo "no selected episodes in $T/$g" >> $L/build_${T}_$g.log; continue; }
  $PY -m sfagen build $C/${T}_$g.yaml --work-dir $W --device cuda:0 --repo-id local/robotwin_${T}_${g}_sfagen_v0 > $L/build_${T}_$g.log 2>&1
  $PY -m sfagen verify $C/${T}_$g.yaml --work-dir $W > $L/verify_${T}_$g.log 2>&1 || echo "verify failed $T $g" >> $L/verify_${T}_$g.log
done
echo "S1 task $T done"
