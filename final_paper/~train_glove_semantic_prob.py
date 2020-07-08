"""
Train semantic probe using glove embeding instead of BERT,
This can show the improvement on Contextual representation 
over word level representation.

This is bad comparison to BERT.
Instead lets use BERT Word Embedding.  
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
from utils import write_to_json_file, create_directory, fix_random_seeds
import featurizer
from utils import glove2dict, convert_ppdb_pairs_to_input_text

TIMESTAMP = time.time()
log_level = logging.INFO
logger = logging.getLogger()
logger.setLevel(log_level)
handler = logging.FileHandler("./log/train_glove_semantic_prob_{}.log".format(TIMESTAMP))
handler.setLevel(log_level)
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define GloVe Lookup table. 
GLOVE_HOME = "/Users/z0p00bj/Google Drive/GaTech/AI_Cert/CS224U_NLU/cs224u/data/glove.6B"
glove_lookup = glove2dict(
    os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))

print(glove_lookup.get('fair'))  

# Create a PyTorch Dataset classes that contain only string
