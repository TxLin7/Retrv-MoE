# Checkpoints

Do not commit large checkpoint files to GitHub.

This evaluation script expects two model paths:

- `ORIGINAL_MODEL_ID`: the base model directory used by `AutoProcessor.from_pretrained(...)` and `Qwen2VLForConditionalGeneration.from_pretrained(...)`.

- `MODEL_ID`: the Retrv-MoE LoRA/MoE checkpoint directory used by `LoraMoeConfig.from_pretrained(...)` and `moe_lora_weights.bin`.

  For the original server run, the intended MoE checkpoint path was:

```text
/njfs/train-ali/tongxu/weibo_X2X/checkpoints/qwen2-vl-2b_mbeir_moe_lora_stage3_2e4_4gpu_1epoch_train_val_bestv3_2temp_32bs_4ex_top2_128Rank_multigpu
```

The MoE checkpoint directory should contain at least:

```text
config.json or the MoE config files required by LoraMoeConfig.from_pretrained
moe_lora_weights.bin
```

Recommended public release layout:

```text
checkpoints/
├── README.md
├── base_model/
└── retrv_moe_lora/
    ├── config.json
    └── moe_lora_weights.bin
```

If the base model is already publicly available elsewhere, do not re-upload it. Provide the public URL and license information in the repository README, and set `ORIGINAL_MODEL_ID` to the downloaded local path when running evaluation.

If the MoE LoRA checkpoint is small enough and you are allowed to release it, upload it to Zenodo or Hugging Face and link it in the GitHub README. If it is too large or cannot be released, leave only this README and describe the limitation clearly.
