# Retrv-MoE

This repository provides the official evaluation/inference artifact for the KDD 2026 paper:

**Retrv-MoE: Scaling Unified Multimodal Retrieval with Sparse Mixture-of-Experts**

## Overview

Retrv-MoE is a unified multimodal retriever based on sparse Mixture-of-Experts. This minimal artifact release contains the evaluation entry point, model wrapper, dataset loader, data collator, and scripts needed to run evaluation with a prepared checkpoint.

Training code is not included in this minimal artifact release.

## Repository Structure

```text
.
├── checkpoints/
├── collators/
├── dataset/
├── eval/
├── models/
├── scripts/eval/
└── README.md
````

## Model Checkpoint

The Retrv-MoE checkpoint is available on ModelScope:

```text
https://www.modelscope.cn/models/JIM3766/retrv-moe-2b-4ex-top2
```

After downloading the checkpoint, set:

```bash
MODEL_ID=/path/to/retrv-moe-2b-4ex-top2
```

The base model should be prepared separately and set as:

```bash
ORIGINAL_MODEL_ID=/path/to/base/model
```

## Data

The full datasets are not included in this repository. Please prepare the M-BEIR data and candidate pools locally, then update the paths in the evaluation script or shell script.

## Evaluation

Example:

```bash
bash scripts/eval/eval_mbeir_moe_lora_binary_alltasks.sh
```

Before running, update the following paths in the shell script:

```bash
MODEL_ID=/path/to/retrv-moe-2b-4ex-top2
ORIGINAL_MODEL_ID=/path/to/base/model
IMAGE_PATH_PREFIX=/path/to/data_binary
GLOBAL_POOL=/path/to/mbeir_union_test_cand_pool_bin.jsonl
INSTRUCTIONS=/path/to/query_instructions.tsv
TASK_CONFIG=./eval/eval_tasks.json
```

## Artifact Scope

This release is intended as an evaluation/inference artifact. It includes source code and a public checkpoint link. Full training code and full datasets are not included in this minimal release.

## Citation

```bibtex
@inproceedings{lin2026retrvmoe,
  title     = {Retrv-MoE: Scaling Unified Multimodal Retrieval with Sparse Mixture-of-Experts},
  author    = {Lin, Tongxu and Xiao, Jiayin},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026}
}
```



[1]: https://kdd2026.kdd.org/call-for-artifact-badging/ "Call for Artifact Badging – KDD 2026"
[2]: https://docs.github.com/repositories/archiving-a-github-repository/referencing-and-citing-content "Referencing and citing content - GitHub Docs"
