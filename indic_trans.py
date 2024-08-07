from inference.engine import Model
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm
import evaluate
import argparse
import pdb
from transformers import AutoTokenizer
from pynvml import *
import gc
# forward_dir="./model_checkpoints/en-indic-preprint/fairseq_model"
# backward_dir="./model_checkpoints/indic-en-preprint/fairseq_model"
# bidirectional_dir="./model_checkpoints/indic-indic-preprint/fairseq_model"

def translate_eng_to_in(data,src_lang,tgt_lang):
    forward_dir="../IndicTrans2/model_checkpoints/en-indic-preprint/fairseq_model"
    model = Model(forward_dir, model_type="fairseq")
    translation=[]
    for text in tqdm(data):
        try:
            output=model.translate_paragraph(text,src_lang,tgt_lang)
            translation.append(output)
        except:
            translation.append("")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return translation

def translate_in_to_eng(data,src_lang,tgt_lang):
    forward_dir="../IndicTrans2/model_checkpoints/indic-en-preprint/fairseq_model"
    model = Model(forward_dir, model_type="fairseq")
    translation=[]
    for text in tqdm(data):
        try:
            output=model.translate_paragraph(text,src_lang,tgt_lang)
            translation.append(output)
        except:
            translation.append("")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return translation