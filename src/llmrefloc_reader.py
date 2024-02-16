import pickle
import json
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig
)
import torch
from prompt_llama_0_shot import prompt_dict as prompt_dict_0
import re
from kilt.eval_downstream import evaluate
from kilt.eval_retrieval import evaluate as evaluate_retrieval
import argparse

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
        
pred_data = []
with open(f'predictions/{file_prefix}_preds.jsonl', 'r', encoding='utf-8') as f:
    for row in f.readlines():
        pred_data.append(json.loads(row))

model = AutoModelForCausalLM.from_pretrained(
    "llama-2-13b",
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained("llama-2-13b")

model.eval()

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if dataset in ['nq', 'hotpotqa', 'triviaqa', 'fever']:
    answer_max_new_tokens = 32
elif dataset in ['wow']:
    answer_max_new_tokens = 64
else:
    answer_max_new_tokens = 256
generation_config = GenerationConfig(
    max_new_tokens=answer_max_new_tokens, #num_beams=1, do_sample=True, top_p=0.7, #temperature=0.95
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)

golds = []
preds = []
cnt = 0
for row in tqdm(data):
    _id = row['id']
    _input = row['input']
    
    provenance = pred_data[cnt]["output"][0]["provenance"]
    title = provenance[0]["wikipedia_title"]
    evidence = provenance[0]["text"]
    
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
    # tmp = {"id": _id, "input":  _input, "output": [{"answer": ans}]}
    preds.append(tmp)
    cnt += 1
    
golds_file = f'predictions/{dataset}_golds.jsonl'
preds_file = f'predictions/reader_{file_prefix}_preds.jsonl'

# with open(golds_file, 'w', encoding='utf-8') as f:
#     for row in golds:
#         f.write(json.dumps(row) + '\n')

with open(preds_file, 'w', encoding='utf-8') as f:
    for row in preds:
        f.write(json.dumps(row) + '\n')

evaluate(golds_file, preds_file)

evaluate_retrieval(golds_file, preds_file, [1, 2, 3, 4, 5, 10], ['wikipedia_id'])

print(cnt)
print(len(preds) - cnt)