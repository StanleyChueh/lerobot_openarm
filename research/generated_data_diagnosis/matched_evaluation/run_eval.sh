#!/usr/bin/env bash
# One matched evaluation: a pinned checkpoint on the first N poses of protocol/eval_poses_100.json, 330 simulated
# steps each, local-only recording. Needs the sim server and the RTC policy server running (EVALUATION_PROTOCOL.md).
#
#   research/generated_data_diagnosis/matched_evaluation/run_eval.sh manual_10|generated_500 N RESULTS_DIR [--resume=true]
set -euo pipefail
cd "$(dirname "$0")/../../.."  # lerobot_openarm
case "$1" in
  manual_10)     REPO=openarm_pringles_lerobot_sim_smolvla_10_v00; REV=9ce7ce4811de11adcc4f407eb5a93586d94a6aee ;;
  generated_500) REPO=openarm_pringles_lerobot_sim_10_generated_500_smolvla_v00; REV=733c6856a507a25875ee2a75b3458756bc60547b ;;
  *) echo "model: manual_10 | generated_500" >&2; exit 2 ;;
esac
N=$2
RESULTS=$3
shift 3
SNAP=$HOME/.cache/huggingface/hub/models--ethanCSL--$REPO/snapshots/$REV  # the pinned revision, already cached
[ -f "$SNAP/model.safetensors" ] || { echo "not cached: $SNAP (huggingface-cli download ethanCSL/$REPO --revision $REV)" >&2; exit 1; }
NAME=$(basename "$(dirname "$RESULTS")")_$(basename "$RESULTS")  # e.g. smoke_results_manual_10: never another run's
mkdir -p outputs/matched_eval
# No DISPLAY and no TTY: no keyboard listener (lerobot's pynput listener hears keys pressed in ANY window).
env -u PYTHONPATH -u DISPLAY -u WAYLAND_DISPLAY LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  OPENARM_EVAL_OPEN=0 OPENARM_EVAL_LOG_DIR="$RESULTS/eval_logs" \
  .venv-v2/bin/python -m lerobot_robot_openarm_umeow.robot_client \
  --server_address=127.0.0.1:8080 --robot.type=openarm_isaac \
  --task="Pick up the Pringles can with the right arm, hand it to the left arm" \
  --policy_type=smolvla --pretrained_name_or_path="$SNAP" \
  --policy_device=cuda --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=latest_only --fps=30 \
  --num_episodes="$N" --reset_time_s=1 --display_data=false \
  --episode_steps=330 --poses=research/generated_data_diagnosis/protocol/eval_poses_100.json --results_dir="$RESULTS" \
  --record_repo_id="ethanCSL/eval_matched_$NAME" --record_root="outputs/matched_eval/eval_matched_$NAME" --push_to_hub=false \
  "$@" < /dev/null
