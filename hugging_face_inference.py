import pandas as pd
import random
import numpy as np
import torch
import os
import argparse
import json
import re
import string
import sys
from collections import Counter
from itertools import zip_longest
import pdb
from tqdm import tqdm
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
set_seed()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default=None,
    help="if specified, we will load the tokenizer from here.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    help="if specified, we will load the tokenizer from here.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="if specified, we will load the tokenizer from here.",
)
parser.add_argument(
    "--shot",
    type=int,
    default=0,
    help="if specified, we will load the tokenizer from here.",
)
parser.add_argument(
    "--local_dir",
    type=bool,
    default=False,
    help="if specified, we will load the tokenizer from here.",
)
args = parser.parse_args()
model_name=args.model_name
dataset=args.dataset
device=args.device
shot=args.shot
local_dir=args.local_dir

with open(dataset, "r") as file:
    data=json.load(file)
context = [i['context'] for i in data]
question = [i['question'] for i in data]
if("indic_QA" in dataset):
    if("hi.json" in dataset or "gu.json" in dataset):
        answer_text = [i['answer'] for i in data]
    else:
        answer_text = [i['answer']['text'][0] for i in data]
else:
    answer_text = [i['answer'] for i in data]


prompts=[]
start="Answer the following question based on the information in the given passage.:"
prompt=""
prompt=start+prompt
run=shot
while(run>0):
    prompt=prompt+f'\n\nPassage:{context[run-1]}\nQuestion:{question[run-1]}\nAnswer:{answer_text[run-1]}'
    run=run-1
for i in range(shot,len(context)):
    pro=prompt+f'\n\nPassage:{context[i]}\nQuestion:{question[i]}\nAnswer:'    
    prompts.append(pro)       

local_data_dir="/dccstor/cssblr/abhishek_langb/checkpoints"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    load_in_8bit=False,
    cache_dir=local_data_dir,
    torch_dtype=torch.float16,
    attn_implementation="eager"
)
model.to(device)

answer=[]
prediction=[]
for index,item in tqdm(enumerate(prompts)):
    inputs = tokenizer(item, return_tensors="pt",truncation=True,max_length=8142).to(device)
    try:
        generate_ids = model.generate(inputs.input_ids, max_new_tokens=50)
        prediction.append(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        answer.append(answer_text[index+shot])
    except:
        continue    

def extract_answer_content(passage):
    delim="\nAnswer:"
    index = passage.find(delim)

    if index != -1:
        content = passage[index + len(delim):]
        return content
    else:
        return " "
        
output=[]
for i in range(len(prediction)):
    output.append(extract_answer_content(prediction[i]))

""" Official evaluation script for v1.1 of the SQuAD dataset. """

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s.strip().split("\n")[0]))))

def f1_score(prediction_, ground_truth):
    prediction_tokens = normalize_answer(prediction_).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def jacard(prediction_, ground_truth):
    a = set(prediction_.lower().split()) 
    b = set(ground_truth.lower().split())
    c = a.intersection(b)
    return float(len(c)) / abs(len(a) + len(b) - len(c))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction_, ground_truths):
    score = metric_fn(str(prediction_),str(ground_truths))
    return score

def evaluate(reference, prediction):
    f1 = exact_match = total = jacard_sim = 0
    for i in range(len(reference)):
        ground_truths = reference[i]
        prediction_ = prediction[i]
        exact_match += metric_max_over_ground_truths(exact_match_score, prediction_, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction_, ground_truths)
        jacard_sim += jacard(prediction_, ground_truths)
        total=total+1
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    jacard_sim = 100.0 * jacard_sim / total
    return {"exact_match": exact_match, "f1": f1, "jacard":jacard_sim}

combined_lists = [[x, y] for x, y in zip(answer, output)]

directory_path = os.path.dirname(dataset)
if not directory_path.endswith('/'):
    filename = os.path.basename(dataset)
    filename=model_name[-9:]+filename

with open(f'{directory_path}/Results_trans_eng/{filename}___.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ground_truth','answer'])
    writer.writerows(combined_lists)

file_path = "result.txt"
with open(file_path, "a") as file:
    file.write(f'{model_name} on {dataset} {str(evaluate(answer,output))}\n')