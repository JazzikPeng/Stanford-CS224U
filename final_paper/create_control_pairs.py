"""
This is a file to create control data.
Control data is the same as training data but with random labels
"""
from typing import Union, Optional
import argparse
import collections
import re
from tqdm import tqdm
import json
import nltk
import numpy as np 
from nltk.tokenize.treebank import TreebankWordDetokenizer

fix_random_seeds(seed=42)

"""
Generate negative sample by randomly replacing vocabulary from 
paraphrase pairs with same POS tags. 

Negative pairs will have relationship = "Negative"
"""
from typing import Union, Optional
import argparse
import collections
import re
from tqdm import tqdm
import json
import nltk
import numpy as np 
from nltk.tokenize.treebank import TreebankWordDetokenizer

fix_random_seeds(seed=42)

if __name__ == "__main__":
    train_path = './data/ppdb_train'
    test_path = './data/ppdb_test'
    