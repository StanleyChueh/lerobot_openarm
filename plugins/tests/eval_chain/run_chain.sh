#!/usr/bin/env bash
# record -> train -> evaluate, all with the official lerobot CLIs and the openarm plugins, on a MOCKED
# robot (no CAN traffic) and a scripted Quest. Needs the GPU and the cached lerobot/smolvla_base.
#   bash plugins/tests/eval_chain/run_chain.sh /tmp/chain
set -euo pipefail
OUT=${1:?usage: run_chain.sh <output dir>}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="$REPO/.venv/bin/python"
RUN=(env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 HF_HUB_OFFLINE=1)
RENAME='{"observation.images.right_wrist_cam": "observation.images.camera1", "observation.images.wrist_cam": "observation.images.camera2", "observation.images.body_cam": "observation.images.camera3"}'
mkdir -p "$OUT"; cd "$REPO"

echo "== 1/6 lerobot-record (3 episodes)"
"${RUN[@]}" "$PY" -u "$HERE/chain_record.py" "$OUT/ds" "$HERE" > "$OUT/record.log" 2>&1
grep "CHAIN RECORD" "$OUT/record.log"

echo "== 2/6 lerobot-train SmolVLA (30 steps)"
rm -rf "$OUT/train"
"${RUN[@]}" "$REPO/.venv/bin/lerobot-train" --policy.path=lerobot/smolvla_base --policy.push_to_hub=false \
  --policy.device=cuda --dataset.repo_id=local/chain --dataset.root="$OUT/ds" --rename_map="$RENAME" \
  --batch_size=4 --steps=30 --save_freq=30 --log_freq=10 --num_workers=0 \
  --output_dir="$OUT/train" --job_name=chain --wandb.enable=false > "$OUT/train.log" 2>&1
grep -o "step:30 .*loss:[0-9.]*" "$OUT/train.log"
CK="$OUT/train/checkpoints/last/pretrained_model"

echo "== 3/6 lerobot-rollout base (normal inference)"
"${RUN[@]}" "$PY" -u "$HERE/chain_rollout.py" base "$CK" - "$HERE" --rename_map="$RENAME" > "$OUT/rollout_base.log" 2>&1
grep "ROLLOUT" "$OUT/rollout_base.log"

echo "== 4/6 lerobot-rollout episodic (2 recorded eval episodes)"
"${RUN[@]}" "$PY" -u "$HERE/chain_rollout.py" episodic "$CK" "$OUT/eval_ds" "$HERE" --rename_map="$RENAME" > "$OUT/rollout_episodic.log" 2>&1
grep "ROLLOUT" "$OUT/rollout_episodic.log"

echo "== 5/6 lerobot-rollout base with RTC (real-time chunking)"
"${RUN[@]}" "$PY" -u "$HERE/chain_rollout.py" base "$CK" - "$HERE" --rename_map="$RENAME" \
  --inference.type=rtc --inference.rtc.mode=guided --inference.rtc.execution_horizon=10 \
  --inference.rtc.max_guidance_weight=10.0 > "$OUT/rollout_rtc.log" 2>&1
grep "ROLLOUT\|ticks over" "$OUT/rollout_rtc.log"

echo "== 6/6 async inference: lerobot_robot_openarm_umeow.policy_server + lerobot's robot_client"
CHAIN_OUT="$OUT" "${RUN[@]}" "$PY" -u "$HERE/chain_async.py" smolvla "$CK" 50 "$HERE" 15 > "$OUT/async.log" 2>&1
grep "ASYNC" "$OUT/async.log"
echo "   policy server errors: $(grep -c 'Error in StreamActions' "$OUT/policy_server_smolvla.log")"
