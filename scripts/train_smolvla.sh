#!/usr/bin/env bash
set -euo pipefail
DATASET_REPO_ID="${DATASET_REPO_ID:?Set DATASET_REPO_ID, e.g. your-hf-user/tablemind_dinner}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/tablemind_smolvla}"
JOB_NAME="${JOB_NAME:-tablemind_smolvla}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-20000}"
DEVICE="${DEVICE:-cuda}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

DATASET_ROOT_ARG=()
if [[ -n "$DATASET_ROOT" ]]; then
  DATASET_ROOT_ARG=(--dataset.root="$DATASET_ROOT")
fi

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET_REPO_ID" \
  "${DATASET_ROOT_ARG[@]}" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.device="$DEVICE" \
  --policy.push_to_hub="$PUSH_TO_HUB"
