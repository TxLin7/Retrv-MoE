import os
import json
from torch.utils.data import Dataset
import random
from PIL import Image
import io

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000


def load_bytes(container_path, offset, length):
    with open(container_path, "rb") as f:
        f.seek(offset)
        bytes_data = f.read(length)
    return bytes_data

def decode_image_from_bytes(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        print(f"Error decoding image bytes: {e}")
        return None

def _prepare_data_dict_v2(txt, img_obj):
    if img_obj is None:
        return {'txt': txt}
    elif txt == '':
        return {'image': img_obj}
    return {"txt": txt, "image": img_obj}

def _load_and_preprocess_image(query_img_path, image_path_prefix):
    if not query_img_path:
        return None
    full_query_img_path = os.path.join(image_path_prefix, query_img_path)
    if not os.path.exists(full_query_img_path):
        return None
    try:
        return Image.open(full_query_img_path).convert("RGB")
    except:
        return None


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        query_data_path: str,
        cand_pool_path: str,
        instructions_path: str,
        image_path_prefix: str,
        tokenizer = None,
        query_bin_path: str = None,
        cand_bin_path: str = None
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.tokenizer = tokenizer
        self.image_path_prefix = image_path_prefix

        self.query_bin_path = query_bin_path
        self.cand_bin_path = cand_bin_path
        self.query_bin_handle = None
        self.cand_bin_handle = None

    def _read_from_bin(self, file_type, offset, length):
        if file_type == 'query':
            if self.query_bin_handle is None and self.query_bin_path:
                self.query_bin_handle = open(self.query_bin_path, "rb")
            f = self.query_bin_handle
        else:
            if self.cand_bin_handle is None and self.cand_bin_path:
                self.cand_bin_handle = open(self.cand_bin_path, "rb")
            f = self.cand_bin_handle

        if f:
            f.seek(offset)
            return f.read(length)
        return None

    def __len__(self) -> int:
        return len(self.query_data)

    def construct_messages(self, data_dict):


        if 'txt' in data_dict and 'image' in data_dict:
            message = [
                {"role": "user", "content": [{"type": "image", "image": data_dict['image']}, {"type": "text", "text": f"{data_dict['txt']}\nSummarize above image and sentence in one word: "}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]},
            ]
        elif 'txt' in data_dict:
            message = [
                {"role": "user", "content": [{"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]},
            ]
        elif 'image' in data_dict:
            message = [
                {"role": "user", "content": [{"type": "image", "image": data_dict['image']}, {"type": "text", "text": f"\nSummarize above image in one word: "}]},
                {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]},
            ]
        return message

    def get_instance(self, index):
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""


        query_img = None
        if 'query_img_offset' in mbeir_entry and self.query_bin_path:
            offset = mbeir_entry['query_img_offset']
            length = mbeir_entry['query_img_length']
            if offset != -1:
                img_bytes = self._read_from_bin('query', offset, length)
                query_img = decode_image_from_bytes(img_bytes)
        elif mbeir_entry.get('query_img_path'):
             query_img = _load_and_preprocess_image(mbeir_entry['query_img_path'], self.image_path_prefix)

        qid = mbeir_entry.get("qid", None)
        query_dataset_id = qid.split(":")[0] if qid else None
        query_modality = mbeir_entry.get("query_modality", None)

        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_modality = pos_cand.get("modality", None)
        pos_cand_txt = pos_cand.get("txt") or ""
        pos_cand_txt = format_string(pos_cand_txt)

        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")


        cand_img = None
        if 'img_offset' in pos_cand and self.cand_bin_path:
             offset = pos_cand['img_offset']
             length = pos_cand['img_length']
             if offset != -1:
                img_bytes = self._read_from_bin('cand', offset, length)
                cand_img = decode_image_from_bytes(img_bytes)
        elif pos_cand.get("img_path"):
             cand_img = _load_and_preprocess_image(pos_cand['img_path'], self.image_path_prefix)

        query_txt_with_prompt = self.tokenizer(query_txt_with_prompt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        query_txt_with_prompt = self.tokenizer.decode(query_txt_with_prompt['input_ids'])
        pos_cand_txt = self.tokenizer(pos_cand_txt, truncation=True, max_length=480, padding=False, return_tensors=None, add_special_tokens=False)
        pos_cand_txt = self.tokenizer.decode(pos_cand_txt['input_ids'])

        query = _prepare_data_dict_v2(query_txt_with_prompt, query_img)
        instance = {"query": query}
        pos_cand_dict = _prepare_data_dict_v2(pos_cand_txt, cand_img)
        instance.update({"pos_cand": pos_cand_dict})
        return instance

    def __getitem__(self, i):
        instance = self.get_instance(i)
        return self.construct_messages(instance['query']), self.construct_messages(instance['pos_cand'])


class QueryDataset(Dataset):
    def __init__(
        self,
        query_data_path: str,
        cand_pool_path: str,
        instructions_path: str,
        image_path_prefix: str,

        query_bin_path: str = None
    ) -> None:
        super(QueryDataset, self).__init__()
        self.query_data = _load_query_data(query_data_path)
        self.cand_pool = _load_cand_pool_as_dict(cand_pool_path)
        self.query_instructions = _load_query_instructions(instructions_path)
        self.image_path_prefix = image_path_prefix


        self.query_bin_path = query_bin_path
        self.query_bin_handle = None

    def _read_from_bin(self, offset, length):
        if self.query_bin_handle is None and self.query_bin_path:
            self.query_bin_handle = open(self.query_bin_path, "rb")

        if self.query_bin_handle:
            self.query_bin_handle.seek(offset)
            return self.query_bin_handle.read(length)
        return None

    def __len__(self) -> int:
        return len(self.query_data)

    def construct_messages(self, data_dict):

        if 'txt' in data_dict and 'image' in data_dict:
            message = [{"role": "user", "content": [{"type": "image", "image": data_dict['image']}, {"type": "text", "text": f"{data_dict['txt']}\nSummarize above image and sentence in one word: "}]}, {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]}]
        elif 'txt' in data_dict:
            message = [{"role": "user", "content": [{"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "}]}, {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]}]
        elif 'image' in data_dict:
            message = [{"role": "user", "content": [{"type": "image", "image": data_dict['image']}, {"type": "text", "text": f"\nSummarize above image in one word: "}]}, {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]}]
        return message

    def get_instance(self, index):
        mbeir_entry = self.query_data[index]
        query_txt = mbeir_entry.get('query_txt') or ""
        qid = mbeir_entry.get("qid", None)


        query_img = None
        if 'query_img_offset' in mbeir_entry and self.query_bin_path:
            offset = mbeir_entry['query_img_offset']
            length = mbeir_entry['query_img_length']
            if offset != -1:
                img_bytes = self._read_from_bin(offset, length)
                query_img = decode_image_from_bytes(img_bytes)
        elif mbeir_entry.get('query_img_path'):
             query_img = _load_and_preprocess_image(mbeir_entry['query_img_path'], self.image_path_prefix)

        query_dataset_id = qid.split(":")[0] if qid else None
        query_modality = mbeir_entry.get("query_modality", None)


        pos_cand_list = mbeir_entry.get("pos_cand_list", [])
        selected_pos_cand_did = _get_random_cand(pos_cand_list)
        pos_cand = self.cand_pool.get(selected_pos_cand_did)
        pos_cand_modality = pos_cand.get("modality", None)

        query_prompt = _get_random_query_prompt(query_dataset_id, query_modality, pos_cand_modality, self.query_instructions)
        query_txt_with_prompt = format_string(f"{query_prompt} {query_txt}")


        query = _prepare_data_dict_v2(query_txt_with_prompt, query_img)
        instance = {"query": query}
        instance['query']['qid'] = hash_qid(qid)
        return instance

    def __getitem__(self, i):
        instance = self.get_instance(i)
        query = instance['query']
        qid = query['qid']
        query_message = self.construct_messages(query)
        return query_message, qid


class CandidateDataset(Dataset):
    def __init__(
        self,
        query_data_path: str,
        cand_pool_path: str,
        instructions_path: str,
        image_path_prefix: str,

        cand_bin_path: str = None
    ) -> None:
        super(CandidateDataset, self).__init__()


        self.cand_pool = _load_cand_pool(cand_pool_path)

        self.image_path_prefix = image_path_prefix


        self.cand_bin_path = cand_bin_path
        self.cand_bin_handle = None

    def _read_from_bin(self, offset, length):
        if self.cand_bin_handle is None and self.cand_bin_path:
            self.cand_bin_handle = open(self.cand_bin_path, "rb")

        if self.cand_bin_handle:
            self.cand_bin_handle.seek(offset)
            return self.cand_bin_handle.read(length)
        return None

    def __len__(self) -> int:
        return len(self.cand_pool)

    def construct_messages(self, data_dict):

        if 'txt' in data_dict and 'image' in data_dict:
            message = [{"role": "user", "content": [{"type": "image", "image": data_dict['image']}, {"type": "text", "text": f"{data_dict['txt']}\nSummarize above image and sentence in one word: "}]}, {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]}]
        elif 'txt' in data_dict:
            message = [{"role": "user", "content": [{"type": "text", "text": f"{data_dict['txt']}\nSummarize above sentence in one word: "}]}, {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]}]
        elif 'image' in data_dict:
            message = [{"role": "user", "content": [{"type": "image", "image": data_dict['image']}, {"type": "text", "text": f"\nSummarize above image in one word: "}]}, {"role": "assistant", "content": [{"type": "text", "text": f"<emb>."}]}]
        return message

    def get_instance(self, index):
        mbeir_cand_pool_entry = self.cand_pool[index]
        did = mbeir_cand_pool_entry.get("did", None)
        cand_txt = mbeir_cand_pool_entry.get("txt") or ""
        cand_txt = format_string(f"{cand_txt}")
        cand_modality = mbeir_cand_pool_entry.get("modality", None)


        img = None
        if 'img_offset' in mbeir_cand_pool_entry and self.cand_bin_path:
             offset = mbeir_cand_pool_entry['img_offset']
             length = mbeir_cand_pool_entry['img_length']
             if offset != -1:
                img_bytes = self._read_from_bin(offset, length)
                img = decode_image_from_bytes(img_bytes)
        elif mbeir_cand_pool_entry.get("img_path"):
             img = _load_and_preprocess_image(mbeir_cand_pool_entry["img_path"], self.image_path_prefix)

        if img is not None and cand_txt != '':
            instance = {"txt": cand_txt, "image": img, "modality": cand_modality}
        elif img is not None:
            instance = {"image": img, "modality": cand_modality}
        else:
            instance = {"txt": cand_txt, "modality": cand_modality}
        instance.update({"did": hash_did(did)})
        return instance

    def __getitem__(self, i):
        candidate = self.get_instance(i)
        did = candidate['did']
        candidate_message = self.construct_messages(candidate)
        return candidate_message, did


def _load_data(data_path):
    assert os.path.exists(data_path), f"Data Path {data_path} does not exist"
    assert data_path.endswith(".jsonl"), f"Data Path {data_path} is not a jsonl file"
    data_entries = _load_data_jsonl(data_path)
    return data_entries

def _load_query_data(query_data_path):
    return _load_data(query_data_path)

def _load_cand_pool_as_dict(cand_pool_data_path):
    cand_pool = _load_data(cand_pool_data_path)
    cand_pool_dict = {}
    for cand_pool_entry in cand_pool:
        did = cand_pool_entry.get("did")
        cand_pool_dict[did] = cand_pool_entry
    return cand_pool_dict

def _load_query_instructions(instructions_path):
    assert os.path.exists(instructions_path), f"Instructions Path {instructions_path} does not exist"
    prompts_dict = {}
    with open(instructions_path, "r") as f:
        next(f)
        for line in f.readlines():
            parts = line.strip().split("\t")
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            prompts = [p for p in parts[4:] if p]
            prompts_dict[key] = prompts
    return prompts_dict

def _get_random_cand(cand_list):
    return random.choice(cand_list)

def format_string(s):
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:
        s = s[0].upper() + s[1:]
        s = s + "." if s[-1] not in [".", "?", "!"] else s
    return s

def _get_random_query_prompt(dataset_id, query_modality, cand_modality, query_instructions):
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    prompts = query_instructions.get(key, [])
    if not prompts: return ""
    prompt = format_string(random.choice(prompts))
    return prompt

def _load_data_jsonl(datapath):
    data_entries = []
    with open(datapath, "r") as fin:
        for line in fin:
            data_entries.append(json.loads(line))
    return data_entries

def hash_qid(qid):
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id

def hash_did(did):
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id

def _load_cand_pool(cand_pool_data_path):
    return _load_data(cand_pool_data_path)
