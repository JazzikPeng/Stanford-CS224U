"""
Same and train_semantic_probe, but read preprocessed tensor instead
"""

# Read in serialized tensor
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
from classifier import logisticRegressionClassifier, MLP1Classifier

from sklearn.metrics import f1_score, accuracy_score
from utils import write_to_json_file, create_directory, fix_random_seeds
import featurizer

TIMESTAMP = time.time()
log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("./log/train_MLP1_probe_serialized_{}.log".format(TIMESTAMP))
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Tested for 5M tensors, no memory issue
class PPDBSerializedDataset(Dataset):
    def __init__(self, tensor_dir: str, featurizer_name: str):
        self.tensor_dir = tensor_dir
        self.featurizer_name = featurizer_name
        all_file_name = os.listdir(tensor_dir)
        all_file_name = [f for f in all_file_name if self.featurizer_name in f]
        self.tensor_count = sum([1 for f in all_file_name if self.featurizer_name in f])

        self.inputs = []
        self.labels = []
        self.size = 0
        self.counter = 0
        for f in all_file_name:
            tensor = np.load(os.path.join(self.tensor_dir, f), allow_pickle=True)
            self.inputs.extend(tensor[0])
            self.labels.extend(tensor[1].detach().cpu().numpy())
            self.size += tensor[0].shape[0]
    
    def __len__(self):
        return self.size

    def __getitem__(self, item):
        curr_id = self.counter
        self.counter += 1
        train_tensor = (torch.tensor(self.inputs[item]), torch.tensor(self.labels[item]))
        return train_tensor

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def early_stopping_monitor(f1_score, monitor='val_f1', patience=10):
    """
    Early stopping after number of epochs with no val_loss 
    improvement
    Args
        loss: dictionary with {'val_f1': [float], 'train_f1': [float]}
    Return
        True if stop, false if continue
    """
    f1_score = f1_score[monitor]
    recent_f1 = f1_score[:patience]
    if sorted(recent_f1, reverse=True) == recent_f1 and len(recent_f1) > patience:
        return True
    else:
        return False


def test(dataloader, classifier, device):
    classifier.eval()
    y_true, y_pred = [], []
    for step,  (inputs, labels) in enumerate(tqdm(dataloader, desc="Iteration")):
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

def serialized_train(dataset,
          classifier, 
          encoder,
          featurizer,
          path = "./model_checkpoint",
          epochs=100, 
          lr=0.01,
          batch_size=8,
          ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training on serialized data")
    # Construct PyTorch DataLoader
    train_test_split = [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    train_set, val_set = torch.utils.data.random_split(dataset, train_test_split)

    train_dataloader = DataLoader(train_set, 
        sampler=RandomSampler(train_set), 
        batch_size=batch_size,
        num_workers = 1)

    eval_dataloader = DataLoader(val_set, 
        sampler=RandomSampler(val_set),
        batch_size=batch_size,
        num_workers = 1)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_step = len(train_dataloader)
    classifier.to(device)
    classifier.train()

    # Setup train loss, eval loss tracking every epoch
    train_loss = []
    file_name_head = f"{encoder}-{featurizer}-{type(classifier).__name__}"
    # eval_loss = [] 
    train_f1_monitor, test_f1_monitor = [], []
    for epoch in trange(epochs, desc='Epochs'):
        tr_loss = 0.
        nb_tr_examples, nb_tr_steps = 0, 0
        print(f"Start Training Epoch {epoch}")
        for step,  (inputs, labels) in enumerate(tqdm(train_dataloader, desc="Iteration")):
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
        f1_test, acc_test = test(eval_dataloader, classifier, device)
        f1_train, acc_train = test(train_dataloader, classifier, device)
        train_f1_monitor.append(f1_train)
        test_f1_monitor.append(f1_test)
        logger.info('[F1, Accuracy] score at epoch %d | train: (%.5f, %.5f) | test: (%.5f, %.5f)' \
            % (epoch+1, f1_test, f1_train, acc_test, acc_train))
        end = time.time()
        if epoch == 0: print(f"Test cost {end-start}")

        if epoch % 1 == 0:
            # Save Model Checkpoint
            create_directory(path)
            torch.save(model.state_dict(), os.path.join(
                path, f"{file_name_head}-{epoch+1}"))
        if (epoch+1) % 10 == 0: adjust_learning_rate(optimizer, lr*0.1) # every 10 epochs reduce lr by factor of 10

        early_stop = early_stopping_monitor({'val_f1': test_f1_monitor, 'train_f1': train_f1_monitor})
        if early_stop:
            break
    # Write train loss per step      
    write_to_json_file(os.path.join(path,
        f"{file_name_head}_train_loss_per_epoch"), train_loss)

if __name__ == "__main__":
    fix_random_seeds(seed=42)
    featurizer = "avg_pooling_featurizer"
    dataset = PPDBSerializedDataset("./serialized_data", featurizer)
    dataset.__getitem__(1)
    model = MLP1Classifier(2, input_dim=768)
    serialized_train(dataset,
          model, 
          encoder='BertModel',
          featurizer=featurizer,
          path = "./model_checkpoint",
          epochs=100, 
          lr=0.01,
          batch_size=1024,
          )
