""" Combine negative pair and positive pair together
Random sample the same amount of negative pairs as positive
pairs.
"""

from typing import Optional
import argparse
import collections
import re
import json
import numpy as np
from utils import *

# Read both negative pairs and positive pairs
corpus_path = "./data/ppdb_train"
negative_path = "./data/ppdb_negative"

with open(corpus_path, mode='r', encoding='utf-8') as fp:
    ppdb_pairs = json.load(fp)
    ppdb_size = len(ppdb_pairs)
    print(f"load {ppdb_size} ppdb positive pairs")

with open(negative_path, mode='r', encoding='utf-8') as fp:
    neg_pairs = json.load(fp)
    neg_size = len(neg_pairs)
    print(f"load {neg_size} ppdb negative pairs")

print(f"Randomly Sample {ppdb_size} negative pairs from {negative_path}")
neg_pairs = np.random.choice(neg_pairs, ppdb_size, replace=False)
train_pairs = []
train_pairs.extend(ppdb_pairs)
train_pairs.extend(neg_pairs)

train_pairs = np.array(train_pairs)
np.random.shuffle(train_pairs)
assert len(train_pairs) == ppdb_size*2

train_pairs = list(train_pairs)
fp = open("./data/ppdb_mix", mode='w+', encoding='utf-8')
json.dump(train_pairs, fp, indent=4)
fp.close()

    