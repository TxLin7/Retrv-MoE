#!/usr/bin/env bash
set -e

MODEL_ID="${/checkpoints/qwen2-vl-2b_mbeir_moe_lora_stage3_2e4_4gpu_1epoch_train_val_bestv3_2temp_32bs_4ex_top2_128Rank_multigpu}"
ORIGINAL_MODEL_ID="${ORIGINAL_MODEL_ID:-/njfs/train-ali/tongxu/weibo_X2X/checkpoints/LamRA-Ret-merged-78}"
IMAGE_PATH_PREFIX="${IMAGE_PATH_PREFIX:-/njfs/train-ali/tongxu/weibo_X2X/data_binary}"
GLOBAL_POOL="${GLOBAL_POOL:-/njfs/train-ali/tongxu/weibo_X2X/data_binary/cand_pool/global/mbeir_union_test_cand_pool_bin.jsonl}"
INSTRUCTIONS="${INSTRUCTIONS:-/njfs/train-ali/tongxu/weibo_X2X/data_binary/instructions/query_instructions.tsv}"
TASK_CONFIG="${TASK_CONFIG:-./eval/eval_tasks.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29509}"
BATCH_SIZE="${BATCH_SIZE:-32}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" accelerate launch --multi_gpu --main_process_port "${MAIN_PROCESS_PORT}" eval/eval_mbeir_moe_lora_binary_alltasks.py \
    --task_config "${TASK_CONFIG}" \
    --original_model_id "${ORIGINAL_MODEL_ID}" \
    --model_id "${MODEL_ID}" \
    --query_cand_pool_path "${GLOBAL_POOL}" \
    --instructions_path "${INSTRUCTIONS}" \
    --image_path_prefix "${IMAGE_PATH_PREFIX}" \
    --use_moe \
    --batch_size "${BATCH_SIZE}"
