"""
Read ppdb_train, ppdb_negative dataset and train semantic classifier
"""

from typing import Optional
import argparse
import collections
import re
import json
import os
import random
import time
import numpy as np
from tqdm import tqdm, trange
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from transformers import BertModel, BertTokenizer
from classifier import logisticRegressionClassifier

from sklearn.metrics import f1_score, accuracy_score
from utils import write_to_json_file, create_directory
from featurizer import cls_featurizer

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


def test(dataloader, classifier, encoder, device):
    classifier.eval()
    y_true, y_pred = [], []
    for step,  (X, X_mask, labels) in enumerate(tqdm(dataloader, desc="Iteration")):
        X = X.to(device)
        X_mask = X_mask.to(device)
        # BERT Encoder
        output = encoder(X, attention_mask = X_mask)
        inputs = cls_featurizer(output) # cls_token
        inputs = inputs.to(device)
        outputs = classifier(inputs)
        y_true.extend(list(labels.numpy()))
        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        # print("load pred", pred)
        y_pred.extend(list(pred))
    classifier.train()
    # Compute F1 Score
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return f1, acc

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
          featurizer=cls_featurizer,
          path = "./model_checkpoint",
          epochs=100, 
          lr=0.01,
          batch_size=1024,
          ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    # Construct PyTorch DataLoader
    train_test_split = [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    train_set, val_set = torch.utils.data.random_split(dataset, train_test_split)

    train_dataloader = DataLoader(train_set, 
        sampler=RandomSampler(train_set), 
        batch_size=batch_size,
        num_workers = 4)

    eval_dataloader = DataLoader(val_set, 
        sampler=RandomSampler(val_set),
        batch_size=batch_size,
        num_workers = 4)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_step = len(train_dataloader)
    classifier.to(device)
    encoder.to(device)
    classifier.train()

    # Setup train loss, eval loss tracking every epoch
    train_loss = []
    # eval_loss = [] 
    for epoch in trange(epochs, desc='Epochs'):
        tr_loss = 0.
        nb_tr_examples, nb_tr_steps = 0, 0
        print("Start Training")
        for step,  (X, X_mask, labels) in enumerate(tqdm(train_dataloader, desc="Iteration")):
            X = X.to(device)
            X_mask = X_mask.to(device)
            # BERT Encoder
            with torch.no_grad():
                output = encoder(X, attention_mask = X_mask)

            inputs = featurizer(output) # cls_token

            inputs = inputs.to(device)
            labels = labels.to(device)
            classifier.zero_grad()
            outputs = classifier(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_examples += inputs.size(0)
            nb_tr_steps += 1
            train_loss.append((loss.item(), nb_tr_steps))   

        logger.info('Total loss at epoch %d: %.5f' % (epoch+1, tr_loss))
        logger.info('Avrg  loss at epoch %d: %.5f' % (epoch+1, tr_loss / nb_tr_examples))
        
        # Evaluate the model f-1
        start = time.time()
        print("Testing {}".format(epoch))
        f1_test, acc_test = test(eval_dataloader, classifier, encoder, device)
        f1_train, acc_train = test(train_dataloader, classifier, encoder, device)
        logger.info('[F1, Accuracy] score at epoch %d | train: (%.5f, %.5f) | test: (%.5f, %.5f)' \
            % (epoch+1, f1_test, f1_train, acc_test, acc_train))
        end = time.time()
        if epoch == 0: print(f"Test cost {end-start}")

        if epoch % 1 == 0:
            # Save Model Checkpoint
            create_directory(path)
            torch.save(model.state_dict(), os.path.join(
                path, f'{type(classifier).__name__}-{epoch+1}'))


    # Write train loss per step      
    write_to_json_file("train_loss_per_step", train_loss)


if __name__ == "__main__":
    # Set Seed
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    hf_weights_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(hf_weights_name)
    bert_model = BertModel.from_pretrained(hf_weights_name)
    for param in bert_model.parameters():
        param.requires_grad = False
    train_dataset = PPDBDataset(corpus_path='./data/ppdb_train',
                        tokenizer=bert_tokenizer,
                        encoder=bert_model,
                        seq_len=128)

    model = logisticRegressionClassifier(2, input_dim=768)
    train(train_dataset, model, encoder=bert_model, epochs=20)






