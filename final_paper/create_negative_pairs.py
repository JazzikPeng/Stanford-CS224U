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

np.random.seed(seed=42)

PPDBExample = collections.namedtuple('PPDBExample', 'text1 text2 relationship')
# Tags to replace for negative pairs
TAG_LIST = ["VBN", "NN", "JJ", "VBP", "VB", "VBZ", "RB", "NNS", "VBG", "CD", "VBD"]

tags_dict = collections.defaultdict(set)
corpus_path = "./data/ppdb_train"

def replace_vocab(pair: PPDBExample, binary_random_num: int, tags_dict=tags_dict) -> Union[PPDBExample, int]:
    """
    Randomly replace one word from text1 with the same tag word in tags_dict
    Args
        pair: PPDBExample
        tags_dict: dictionary, key: POS tags, value: set of vocabulary
        binary_random: 0 or 1
    """
    text1, text2, rel = pair['text1'], pair['text2'], pair['relationship']
    neg_text1 = [text1, text2][binary_random_num]
    neg_text2 = [text1, text2][1 - binary_random_num]
    token = nltk.word_tokenize(neg_text1)
    pos_tag = nltk.pos_tag(token)
    # Randomly replace vocab with tag in the TAG_LIST
    random_idx = np.random.choice(range(len(pos_tag)), len(pos_tag), replace=False)
    for i in random_idx:
        vocab = pos_tag[i][0]
        tag = pos_tag[i][1]        
        if tag in TAG_LIST:
            # TODO: Add GloVe similarity Check=
            replace_vocab = np.random.choice(tuple(tags_dict[tag]))
            token[i] = replace_vocab
            neg_text1 = TreebankWordDetokenizer().detokenize(token)
            negative_pair = PPDBExample(neg_text1, neg_text2, "Negative")
            return negative_pair
    return 0

def write_examples_to_json_file(file_path: str, examples: [PPDBExample]) -> None:
    fp = open(file_path, mode='w+', encoding='utf-8')
    examples = [example._asdict() for example in examples]
    json.dump(examples, fp, indent=4)
    fp.close()


with open(corpus_path, mode='r', encoding='utf-8') as fp:
    ppdb_pairs = json.load(fp)
    ppdb_size = len(ppdb_pairs)
    print(f"load {ppdb_size} ppdb pairs")

# Create POS dictionary
for pair in tqdm(ppdb_pairs):
    text1, text2, rel = pair['text1'], pair['text2'], pair['relationship']
    token = nltk.word_tokenize(text1)
    pos_tag = nltk.pos_tag(token)
    
    token = nltk.word_tokenize(text2)
    pos_tag.extend(nltk.pos_tag(token))
    # print(pos_tag)
    _ = [tags_dict[tag].add(vocab) for vocab, tag in pos_tag]

# Replace vocabulary randomly with the same POS tags to create negative pair
negative_size = ppdb_size
negative_pairs = []
for idx in tqdm(range(negative_size)):
    pair = ppdb_pairs[idx]
    negative_pair1 = replace_vocab(pair, 0, tags_dict=tags_dict)
    negative_pair2 = replace_vocab(pair, 1, tags_dict=tags_dict)
    if negative_pair1: negative_pairs.append(negative_pair1)
    if negative_pair2: negative_pairs.append(negative_pair2)

write_examples_to_json_file("./data/ppdb_negative", negative_pairs)