#!/bin/bash
# Render render_todo.json (body; config $1, default copied) with 4 workers over 2 GPUs. usage: render_todo.sh [config] [todo.json]
set -e
S=/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen
R=/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run; P=openarm_pringles_v03
[ "${REV:-1}" = 2 ] && { R=/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03r2_run; P=openarm_pringles_v03r2; }  # amendment 4
CFG=${1:-configs/$P.yaml}; TODO=${2:-$R/render_todo.json}
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/claude-1000/mpl /home/stanley/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
cd $S
$PY -c "
import json,collections
g=collections.defaultdict(list)
for e,k in json.load(open('$TODO')): g[e].append(k)
for i,(e,ks) in enumerate(sorted(g.items())): print(i%2, e, ','.join(ks))" > $R/logs/render_jobs.txt
cat $R/logs/render_jobs.txt | xargs -P 4 -L 1 bash -c "$PY -m sfagen render $CFG --episodes \$1 --keys \$2 --device cuda:\$0 >> $R/logs/render_\$1.log 2>&1"
grep -l Traceback $R/logs/render_*.log || echo no-tracebacks
