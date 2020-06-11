"""
Read ppdb_train, ppdb_negative dataset and train semantic classifier
"""

from typing import Optional
import argparse
import collections
import re
import json
import numpy as np
from tqdm import tqdm, trange
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import BertModel, BertTokenizer
from classifier import logisticRegressionClassifier

log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("logistic_regression_classifier.log")
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


SEP_TOKEN = "[SEP]"

def convert_ppdb_pairs_to_input_text(pair):
    text1, text2, rel = pair['text1'], pair['text2'], pair['relationship']
    label = 0 if rel=='Negative' else 1
    text = text1 + " " + SEP_TOKEN + " "+ text2
    return text, label

def create_train_data():
    """Create data for training classifier"""
    with open('./data/ppdb_mix', mode='r', encoding='utf-8') as fp:
        ppdb_pairs = json.load(fp)
        ppdb_size = len(ppdb_pairs)
        print(f"load {ppdb_size} ppdb negative pairs")

    hf_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)
    bert_model = BertModel.from_pretrained(hf_weights_name)

    # Random shuffle ppdb_pairs
    ppdb_pairs = np.array(ppdb_pairs)
    np.random.shuffle(ppdb_pairs)

    input_text = []
    labels = []
    for pair in ppdb_pairs:
        text, label = convert_ppdb_pairs_to_input_text(pair)
        input_text.append(text)
        labels.append(label)

    return input_text, labels
    # bert_ids = bert_tokenizer.batch_encode_plus(input_text, 
    #                                             add_special_tokens=True,
    #                                             return_attention_masks=True,
    #                                             pad_to_max_length=True)

    # X = torch.tensor(bert_ids['input_ids'])
    # X_mask = torch.tensor(bert_ids['attention_mask'])

    # with torch.no_grad():
    #     bert_final_hidden_states, cls_output = bert_model(
    #         X, attention_mask = X_mask)

def cls_featurizer(encoder_output):
    """Construct Features From BERT.forward() ouput"""
    final_hidden_state, cls_output = encoder_output
    return cls_output

class PPDBDataset(Dataset):
    def __init__(self, corpus_path: str, tokenizer: BertTokenizer, encoder: BertModel, 
        seq_len: int, encoding: str = 'utf-8'):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.encoding = encoding

        self.pairs = []
        self.pair_counter = 0 
        self.count = 0

        with open(corpus_path, mode='r', encoding='utf-8') as fp:
            self.ppdb_pairs = json.load(fp)
            self.count = len(self.ppdb_pairs)
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, item):
        curr_id = self.pair_counter
        self.pair_counter += 1
        pair = self.ppdb_pairs[item]
        input_text, label = convert_ppdb_pairs_to_input_text(pair)
        bert_ids = self.tokenizer.batch_encode_plus([input_text], 
                                                max_length = self.seq_len,
                                                add_special_tokens=True,
                                                return_attention_masks=True,
                                                pad_to_max_length=True)
        X = torch.tensor(bert_ids['input_ids'])
        X_mask = torch.tensor(bert_ids['attention_mask'])

        # with torch.no_grad():
        #     output = self.encoder(X, attention_mask = X_mask)

        # feat = cls_featurizer(output)
        # train_tensor = (feat.squeeze(), torch.tensor(label))
        train_tensor = (X.squeeze(), X_mask.squeeze(), torch.tensor(label))
        return train_tensor

def train(dataset,
          classifier, 
          encoder,
          epochs=100, 
          lr=0.01,
          batch_size=2560):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    # Construct PyTorch DataLoader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_step = len(train_dataloader)
    classifier.to(device)
    encoder.to(device)
    classifier.train()
    for epoch in trange(epochs, desc='Epochs'):
        tr_loss = 0.
        nb_tr_examples, nb_tr_steps = 0, 0
        for step,  (X, X_mask, labels) in enumerate(tqdm(train_dataloader, desc="Iteration")):
            X = X.to(device)
            X_mask = X_mask.to(device)
            # BERT Encoder
            output = encoder(X, attention_mask = X_mask)

            inputs = cls_featurizer(output) # cls_token

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = classifier(inputs)
            classifier.zero_grad()
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_examples += inputs.size(0)
            nb_tr_steps += 1
        logger.info('Total loss at epoch %d: %.5f' % (epoch, tr_loss))
        logger.info('Avrg  loss at epoch %d: %.5f' % (epoch, tr_loss / nb_tr_examples))

if __name__ == "__main__":
    hf_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)
    bert_model = BertModel.from_pretrained(hf_weights_name)
    for param in bert_model.parameters():
        param.requires_grad = False
    train_dataset = PPDBDataset(corpus_path='./data/ppdb_mix',
                        tokenizer=bert_tokenizer,
                        encoder=bert_model,
                        seq_len=128)
    model = logisticRegressionClassifier(2, input_dim=768)

    train(train_dataset, model, encoder=bert_model)






