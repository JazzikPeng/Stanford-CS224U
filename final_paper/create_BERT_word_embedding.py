"""
This will serve as control task. 
Compare accuracy on BERT contextual representation and word embedding.
So we create BERT word embedding for the PPBD dataset corpus
"""
from typing import Optional
import argparse
import collections
import re
import os
import json
import numpy as np
from pathlib import Path
from utils import write_to_json_file

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from transformers import BertModel, BertTokenizer
from classifier import logisticRegressionClassifier

data_path = "./data/ppdb_mix"
# Loop through ppdb Dataset and fine all vocab
def BERT_encode_vocab(vocab: str, hf_weights_name='bert_base_uncased') -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for serializing")
    hf_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)
    bert_model = BertModel.from_pretrained(hf_weights_name)
    for param in bert_model.parameters():
        param.requires_grad = False
    
    bert_ids = bert_tokenizer.batch_encode_plus(vocab, 
                                     add_special_tokens=True, 
                                     return_attention_masks=True,
                                     pad_to_max_length=False)
    X = torch.tensor(bert_ids['input_ids'])
    X_mask = torch.tensor(bert_ids['attention_mask'])    

    X.to(device)                            
    X_mask.to(device)
    with torch.no_grad():
        output = encoder(X, attention_mask = X_mask)
    return 
                        


def main():
    vocab = set()
    with open(data_path, mode='r', encoding='utf-8') as fp:
        ppdb_pairs = json.load(fp)
        ppdb_size = len(ppdb_pairs)
        print(f"load {ppdb_size} ppdb pairs")

    # Construct Vocab
    corpus = []
    for pair in ppdb_pairs:
        text1, text2 = pair['text1'].lower(), pair['text2'].lower()
        corpus.extend(text1.split())
        corpus.extend(text2.split())
    vocab = set(corpus)

    # Use BERT to Process Vocab

    
if __name__ == "__main__":
    pass
