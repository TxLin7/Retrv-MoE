#!/usr/bin/env bash
set -e

MODEL_ID=/path/to/retrv-moe-2b-4ex-top2
ORIGINAL_MODEL_ID=/path/to/base/model
IMAGE_PATH_PREFIX=""
GLOBAL_POOL=""
INSTRUCTIONS=query_instructions.tsv
TASK_CONFIG=./eval/eval_tasks.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAIN_PROCESS_PORT=29509
BATCH_SIZE=32

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch --multi_gpu --main_process_port "${MAIN_PROCESS_PORT}" eval/eval_mbeir_moe_lora_binary_alltasks.py \
    --task_config "${TASK_CONFIG}" \
    --original_model_id "${ORIGINAL_MODEL_ID}" \
    --model_id "${MODEL_ID}" \
    --query_cand_pool_path "${GLOBAL_POOL}" \
    --instructions_path "${INSTRUCTIONS}" \
    --image_path_prefix "${IMAGE_PATH_PREFIX}" \
    --use_moe \
    --batch_size "${BATCH_SIZE}"
