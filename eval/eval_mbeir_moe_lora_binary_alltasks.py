import json
import sys
import os
import argparse
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import accelerate
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm
import numpy as np
import datetime


current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)


from models.configuration_lora_moe import LoraMoeConfig
from models.modelling_lora_moe import LoraMoeModel
from collators.mbeir_eval import MbeirQueryDataCollator, MbeirCandidateDataCollator

from dataset.datasets_mbeir_binary_v2 import QueryDataset, CandidateDataset

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000

def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance_score = int(parts[3])

                task_id = parts[4] if len(parts) > 4 else "unknown"

                if relevance_score > 0:
                    if query_id not in qrel:
                        qrel[query_id] = []
                    qrel[query_id].append(doc_id)
                    if query_id not in qid_to_taskid:
                        qid_to_taskid[query_id] = task_id

    return qrel, qid_to_taskid

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0
    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    relevant_docs_set = set(relevant_docs)
    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0

def get_bin_path(jsonl_path):
    if not jsonl_path:
        return None
    base_dir = os.path.dirname(jsonl_path)
    file_name = os.path.basename(jsonl_path)
    if "_bin.jsonl" in file_name:
        bin_name = file_name.replace("_bin.jsonl", "_images.bin")
    elif ".jsonl" in file_name:
        name_without_ext = os.path.splitext(file_name)[0]
        bin_name = f"{name_without_ext}_images.bin"
    else:
        return None
    full_bin_path = os.path.join(base_dir, bin_name)
    if os.path.exists(full_bin_path):
        return full_bin_path
    else:
        print(f"[Warning] Bin file not found: {full_bin_path}, using text-only mode.")
        return None

def tensors_to_device(data, device, dtype=torch.bfloat16):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            if key == 'pixel_values':
                data[key] = data[key].to(device).to(dtype)
            else:
                data[key] = data[key].to(device)
    return data


def eval_single_task(model, tokenizer, processor, args, task_config, accelerator, device):
    task_name = task_config.get("name", "unknown")
    current_query_path = task_config["query_data_path"]
    current_cand_pool_path = task_config["cand_pool_path"]
    current_qrels_path = task_config["qrels_path"]

    if accelerator.is_main_process:
        print(f"\n{'='*40}")
        print(f"Starting Task: {task_name}")
        print(f"{'='*40}")

    query_bin_path = get_bin_path(current_query_path)
    cand_bin_path = get_bin_path(current_cand_pool_path)


    query_dataset = QueryDataset(
        query_data_path=current_query_path,
        cand_pool_path=args.query_cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix,
        query_bin_path=query_bin_path
    )

    cand_dataset = CandidateDataset(
        query_data_path=current_query_path,
        cand_pool_path=current_cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix,
        cand_bin_path=cand_bin_path
    )

    query_data_collator = MbeirQueryDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = MbeirCandidateDataCollator(tokenizer=tokenizer, processor=processor)

    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    query_dataloader, candidate_dataloader = accelerator.prepare(query_dataloader, candidate_dataloader)


    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []

    with torch.no_grad():

        for batch in tqdm(candidate_dataloader, disable=not accelerator.is_main_process, desc=f"[{task_name}] Candidates"):
            batch = tensors_to_device(batch, device)
            batch_candidate_ids = batch['dids']

            hidden_states = model(output_hidden_states=True, return_dict=True, **batch).hidden_states[-1]
            embed_index = model.module.config.emb_token_ids[0]
            embed_indices = torch.argmax((batch["labels"] == embed_index).int(), dim=1)
            candidate_embed = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]

            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)

            if len(batch_candidate_ids) > len(candidate_embed):
                batch_candidate_ids = batch_candidate_ids[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)


        for batch in tqdm(query_dataloader, disable=not accelerator.is_main_process, desc=f"[{task_name}] Queries"):
            batch = tensors_to_device(batch, device)
            batch_query_ids = batch['qids']

            hidden_states = model(output_hidden_states=True, return_dict=True, **batch).hidden_states[-1]
            embed_index = model.module.config.emb_token_ids[0]
            embed_indices = torch.argmax((batch["labels"] == embed_index).int(), dim=1)
            query_embed = hidden_states[torch.arange(len(embed_indices)), embed_indices - 1]

            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)
            if len(batch_query_ids) > len(query_embed):
                batch_query_ids = batch_query_ids[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)


    target_task_score = 0.0

    if accelerator.is_main_process:
        if len(query_features) > 0:
            query_features = torch.cat(query_features, dim=0)
        if len(candidate_features) > 0:
            candidate_features = torch.cat(candidate_features, dim=0)


        index = []
        scores = []
        chunk_size = 1000
        for i in range(0, len(query_features), chunk_size):
            end_i = min(i + chunk_size, len(query_features))
            query_feature_chunk = query_features[i:end_i]
            score_chunk = query_feature_chunk @ candidate_features.T
            topk_score, topk_indexes = torch.topk(score_chunk, k=50, dim=-1)
            index.extend(topk_indexes.squeeze().tolist())
            scores.extend(topk_score.tolist())


        cand_names = np.array([[unhash_did(candidate_ids[idx]) for idx in row] for row in index])
        query_names = [unhash_qid(qid) for qid in query_ids]


        model_name = args.model_id.strip("/").split('/')[-1]
        save_dir_name = f"./LamRA_Ret_eval_results/{model_name}"
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)


        save_name_prefix = current_qrels_path.split('/')[-1].replace('_qrels.txt', '')

        save_name = f"{save_name_prefix}_{model_name}"


        print(f"Saving artifacts to {save_dir_name} ...")
        with open(f"{save_dir_name}/{save_name}_query_names.json", 'w') as f:
            json.dump(query_names, f, indent=2)
        with open(f"{save_dir_name}/{save_name}_cand_names.json", 'w') as f:
            json.dump(cand_names.tolist(), f, indent=2)
        with open(f"{save_dir_name}/{save_name}_scores.json", 'w') as f:
            json.dump(scores, f, indent=2)


        torch.save(query_features.cpu(), f"{save_dir_name}/{save_name}_query_features.pth")
        torch.save(candidate_features.cpu(), f"{save_dir_name}/{save_name}_candidate_features.pth")

        with open(f"{save_dir_name}/{save_name}_query_ids.json", 'w') as f:
            json.dump(query_ids, f, indent=2)
        with open(f"{save_dir_name}/{save_name}_candidate_ids.json", 'w') as f:
            json.dump(candidate_ids, f, indent=2)


        qrel, qid_to_taskid = load_qrel(current_qrels_path)
        k_lists = [1, 5, 10, 50]
        res = {f'recall_{k}': [] for k in k_lists}

        for ind, query_name in enumerate(query_names):
            relevant_docs = qrel.get(query_name, [])
            retrieved_indices_for_qid = cand_names[ind]
            for k in k_lists:
                recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].append(recall_at_k)


        results_txt = f"{save_dir_name}/{model_name}_summary_results.txt"
        with open(results_txt, 'a') as f:


            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\nTime: {current_time}\n")
            f.write(f"Task: {task_name} | Qrels: {current_qrels_path}\n")

            for k in k_lists:
                if res[f'recall_{k}']:
                    avg_recall = sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])
                else:
                    avg_recall = 0.0

                print(f"  recall_at_{k} = {avg_recall:.4f}")
                f.write(f"recall_at_{k} = {avg_recall:.4f}\n")

            f.write("-" * 30 + '\n')


        if "fashion" in task_name.lower():
            target_metric_list = res['recall_10']
            metric_type = "Recall@10"
        else:
            target_metric_list = res['recall_5']
            metric_type = "Recall@5"

        if target_metric_list:
            target_task_score = sum(target_metric_list) / len(target_metric_list)
        else:
            target_task_score = 0.0

        print(f"[{task_name}] Contribution to Final Average: {metric_type} = {target_task_score:.4f}")


    del query_features, candidate_features, query_dataloader, candidate_dataloader
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


    if accelerator.is_main_process:
        return target_task_score
    else:
        return None


def main(args):
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main_process = accelerator.is_main_process


    if not args.task_config or not os.path.exists(args.task_config):
        raise ValueError(f"Task config file not found: {args.task_config}")

    with open(args.task_config, 'r') as f:
        tasks = json.load(f)

    if is_main_process:
        print(f"Loaded {len(tasks)} tasks.")


    if is_main_process: print(f"Loading base model from: {args.original_model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.original_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.original_model_id, fix_mistral_regex=True)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = args.model_max_length

    emb_token = "<emb>"
    if emb_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([emb_token])
        model.resize_token_embeddings(len(tokenizer))
        model.config.emb_token_ids = [tokenizer.convert_tokens_to_ids(emb_token)]


    if hasattr(args, 'use_moe') and args.use_moe:
        if is_main_process: print(f"Initializing MoE-LoRA from {args.model_id}...")
        moe_config = LoraMoeConfig.from_pretrained(args.model_id)
        model = LoraMoeModel(model, moe_config)
        model.to(dtype=torch.bfloat16)
        weights_path = os.path.join(args.model_id, "moe_lora_weights.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    elif args.model_id != args.original_model_id:


         pass

    model.eval()
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True

    model = accelerator.prepare(model)


    all_task_scores = []

    for i, task in enumerate(tasks):

        score = eval_single_task(model, tokenizer, processor, args, task, accelerator, device)


        if is_main_process and score is not None:
            all_task_scores.append(score)


    if is_main_process:
        print("\n" + "="*40)
        print("All tasks completed. Calculating Final Average...")

        if len(all_task_scores) > 0:
            final_average = sum(all_task_scores) / len(all_task_scores)


            model_name = args.model_id.strip("/").split('/')[-1]
            save_dir_name = f"./LamRA_Ret_eval_results/{model_name}"
            if not os.path.exists(save_dir_name):
                os.makedirs(save_dir_name)

            results_txt = f"{save_dir_name}/{model_name}_summary_results.txt"

            msg = f"\nFinal Average Score (Fashion=R@10, Others=R@5): {final_average:.4f}\n"
            print(msg)

            with open(results_txt, 'a') as f:
                f.write(msg)
                f.write("="*40 + "\n")
        else:
            print("No scores collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_config', type=str, required=True, help="Path to json file containing tasks")
    parser.add_argument('--instructions_path', type=str, required=True)
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--original_model_id', type=str, required=True)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--query_cand_pool_path', type=str, required=True, help="Global pool for query prompts")
    parser.add_argument('--image_path_prefix', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_moe', action='store_true')


    parser.add_argument('--query_data_path', type=str, default=None)
    parser.add_argument('--cand_pool_path', type=str, default=None)
    parser.add_argument('--qrels_path', type=str, default=None)

    args = parser.parse_args()
    main(args)
