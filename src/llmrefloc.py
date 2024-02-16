import pickle
import json
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig
)
import torch
import numpy as np
from typing import Dict, List
from seal import FMIndex
from prompt_llama_0_shot import prompt_dict as prompt_dict_0
import re
from kilt.eval_downstream import evaluate
from kilt.eval_retrieval import evaluate as evaluate_retrieval
import argparse
from kmp import KMPSearch
import time

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument("--dataset", default="", type=str, help="dataset")
parser.add_argument("--inference_mode", default="", type=str, help="inference_mode")
parser.add_argument("--model_path", default="", type=str, help="model path")
parser.add_argument("--stage1_beam_size", default=15, type=int, help="stage1 beam size")
parser.add_argument("--stage2_beam_size", default=10, type=int, help="stage2_beam_size")
parser.add_argument("--stage1_return_nums", default=2, type=int, help="stage1_return_nums")
parser.add_argument("--prefix_len", default=16, type=int, help="prefix len")
parser.add_argument("--lam", default=0.9, type=float, help="lam")
args = parser.parse_args()

dataset = args.dataset
inference_mode = args.inference_mode
model_path = args.model_path
model_name = model_path.split('/')[-1]
stage1_return_nums = args.stage1_return_nums

file_prefix = f'{dataset}_{inference_mode}_{model_name}_{args.stage1_beam_size}_{args.stage2_beam_size}_{args.stage1_return_nums}_{args.prefix_len}_{args.lam}'
print(file_prefix)

prompt_0 = prompt_dict_0[inference_mode][dataset]
pattern = r"answer is (.+)\."

data = []
with open(f'./data/{dataset}-dev-kilt.jsonl', 'r', encoding='utf-8') as f:
    for row in f.readlines():
        data.append(json.loads(row))

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

generation_title_config = GenerationConfig(
    max_new_tokens=64, num_beams=args.stage1_beam_size,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)

generation_recite_config = GenerationConfig(
    max_new_tokens=args.prefix_len, num_beams=args.stage2_beam_size,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)

if dataset in ['nq', 'hotpotqa', 'triviaqa', 'fever']:
    answer_max_new_tokens = 32
elif dataset in ['wow']:
    answer_max_new_tokens = 64
else:
    answer_max_new_tokens = 256
generation_config = GenerationConfig(
    max_new_tokens=answer_max_new_tokens,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)

class get_allowed_tokens_stage2:
    def __init__(self):
        pass

    def init_para(self, tmp_index, L_input):
        self.tmp_index = tmp_index
        self.tmp_all_allowed_tokens = list(set(self.tmp_index.occurring_distinct))
        if 2 in self.tmp_all_allowed_tokens:
            self.tmp_all_allowed_tokens.remove(2)
        self.L_input = L_input

    def allowed_tokens_fn(self, batch_id: int, input_ids: List[int]):

        input_ids_list = input_ids.tolist()
        
        sub_input_ids_list = input_ids_list[self.L_input:]
        
        if 2 in sub_input_ids_list:
            return [0]
        
        if len(sub_input_ids_list) == 0:
            return self.tmp_all_allowed_tokens
        else:
            low, high = self.tmp_index.get_range(sub_input_ids_list)
            fm_index_result = self.tmp_index.get_distinct_count_multi([low], [high])
            allowed_tokens, _ = fm_index_result.pop()

        return allowed_tokens
    

class get_allowed_tokens_stage1:
    def __init__(self, tokenizer, trie):
        self.tokenizer = tokenizer
        self.trie = trie
        self.L_input = None

    def allowed_tokens_fn(self, batch_id: int, input_ids: List[int]):
        input_ids_list = input_ids.tolist()

        allowed_tokens = trie.get(input_ids_list[self.L_input:])

        return allowed_tokens
    
with open("cache/llama_kilt_w1002full_titles_trie.pkl", "rb") as f:
    trie = pickle.load(f)
    
allowed_tokens_func_1step = get_allowed_tokens_stage1(tokenizer, trie)

index = FMIndex.load(f'cache/kilt_w1002full_corpus.fm_index')
allowed_tokens_func = get_allowed_tokens_stage2()

with open('cache/w1002full_wikititle2id.json', 'r', encoding='utf-8') as f:
    wikititle2id = json.load(f)

with open('cache/title2wikiid.json', 'r', encoding='utf-8') as f:
    title2id = json.load(f)

if dataset in ['nq', 'hotpotqa', 'triviaqa', 'eli5']:
    title_recite_prompt = 'Question: {}\n\nThe Wikipedia title corresponding to the above question is:\n\nTitle:'
elif dataset in ['fever']:
    title_recite_prompt = 'Claim: {}\n\nThe Wikipedia article corresponding to the above claim is:\n\nTitle:'
elif dataset in ['wow']:
    title_recite_prompt = 'Conversation: {}\n\nThe Wikipedia article corresponding to the above conversation is:\n\nTitle:'

golds = []
preds = []
cnt = 0
all_time = 0.
tst_nums = 0
for row in tqdm(data):
    _id = row['id']
    _input = row['input']
    
    cnt += 1
    
    start_time = time.time()
    
    # stage 1
    input_prompt = title_recite_prompt.format(_input)
    inputs = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].cuda()
    allowed_tokens_func_1step.L_input = input_ids.shape[1]
    res = model.generate(
        input_ids=input_ids,
        generation_config=generation_title_config,
        prefix_allowed_tokens_fn=allowed_tokens_func_1step.allowed_tokens_fn,
        num_return_sequences=stage1_return_nums,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = res["sequences"]
    scores = res["sequences_scores"]
    title_scores = {}
    needids = set()
    tmp_id2title = {}
    for j in range(len(sequences)):
        tokens = sequences[j].tolist()
        text = tokenizer.decode(tokens[input_ids.shape[1]:], skip_special_tokens=True)
        needids.update(wikititle2id[text])
        d_id = title2id[text]
        tmp_id2title[d_id] = text
        title_scores[d_id] = scores[j].item()
    needids = list(needids)
    
    
    re_allow_tokens = []
    doc_labels = []
    for doc_id in needids:
        doc_token = index.get_doc(doc_id)
        doc_idx = index.labels[doc_id]
        doc_labels.append(doc_idx)
        re_allow_tokens.append(doc_token)
    tmp_index = FMIndex()
    tmp_index.initialize(re_allow_tokens, in_memory=True)
    tmp_index.labels = doc_labels
    
    input_prompt = prompt_0[0].format(_input)
    inputs = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].cuda()
    L = input_ids.shape[1]

    # stage 2
    allowed_tokens_func.init_para(tmp_index, L)
    
    res = model.generate(
        input_ids=input_ids,
        generation_config=generation_recite_config,
        prefix_allowed_tokens_fn=allowed_tokens_func.allowed_tokens_fn,
        num_return_sequences=args.stage2_beam_size,
        return_dict_in_generate=True,
        output_scores=True,
    )
    sequences = res["sequences"]
    scores = res["sequences_scores"]

    texts = []
    evidence = ''
    provenance = []
    para_scores = []
    evi_infos = []
    for j in range(args.stage2_beam_size):
        seq = sequences[j].tolist()

        tokens = seq[L:]
        while tokens[-1] in (0,):
            tokens.pop()
        low, high = tmp_index.get_range(tokens)
        docs_tokens = []
        doc_ids = []
        for idx_row in range(low, high):
            doc_id = tmp_index.get_doc_index_from_row(idx_row)
            doc_token = tmp_index.get_doc(doc_id)
            _doc_id = tmp_index.labels[doc_id]
            if _doc_id not in doc_ids:
                doc_ids.append(_doc_id)
                docs_tokens.append(doc_token)
        for _j in range(len(docs_tokens)):
            doc_tokens = docs_tokens[_j]

            t_sts = KMPSearch(tokens, doc_tokens)
            if len(t_sts) == 0:
                tst_nums += 1
                continue
            t_st = t_sts[0]
            t_evidence = tokenizer.decode(doc_tokens[t_st:t_st+150], skip_special_tokens=True)

            _doc_id = doc_ids[_j]
            
            lam = args.lam
            para_scores.append(lam * title_scores[_doc_id] + (1-lam) * scores[j].item())
            evi_infos.append({'evi': t_evidence, 'doc_id': _doc_id})
    
    np_arr = np.array(para_scores) 
    top_indexes = np.argsort(-np_arr)[:10]
    
    evidences = []
    for idx in top_indexes:
        _doc_id = evi_infos[idx]['doc_id']
        _title  = tmp_id2title[_doc_id]
        t_evidence = evi_infos[idx]['evi']
        evidences.append(t_evidence)
        provenance.append({"wikipedia_id": _doc_id, "wikipedia_title": _title, "text": t_evidence})
    title = provenance[0]["wikipedia_title"]
    evidence = '\n'.join(evidences[:1])
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    all_time += elapsed_time
    
    # reader
    input_prompt = prompt_0[1].format(evidence, _input)
    inputs = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].cuda()
    
    out = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip()

    if dataset in ['fever']:
        tmp_ans = response.split('\n')[0].strip()
        if 'true' in tmp_ans.lower():
            ans = 'SUPPORTS'
        elif 'false' in tmp_ans.lower():
            ans = 'REFUTES'
        else:
            ans = tmp_ans
    else:
        ans = response.split('\n')[0].strip()

    if cnt % 100 == 0:
        print('prompt:\n', input_prompt)
        print('response:\n', response)
        print('pred_ans:', ans)
        print('correct ans:', row['output'][0]['answer'])
        print('-'*50)

    golds.append(row)
    tmp = {"id": _id, "input":  _input, "output": [{"answer": ans, "provenance": provenance}]}
    preds.append(tmp)

golds_file = f'predictions/{dataset}_golds.jsonl'
preds_file = f'predictions/{file_prefix}_preds.jsonl'

with open(golds_file, 'w', encoding='utf-8') as f:
    for row in golds:
        f.write(json.dumps(row) + '\n')

with open(preds_file, 'w', encoding='utf-8') as f:
    for row in preds:
        f.write(json.dumps(row) + '\n')

evaluate(golds_file, preds_file)

evaluate_retrieval(golds_file, preds_file, [1, 2, 3, 4, 5, 10], ['wikipedia_id'])

print(f"The program took {all_time} seconds to complete")
print("avg time", 1. *  all_time / len(preds))