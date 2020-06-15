from typing import Optional
import argparse
import collections
import re
import os
import json
import numpy as np
from pathlib import Path
from utils import write_to_json_file

def resample_train_test(data_path: str, lengths: [int, int]) -> [str, str]:
    with open(data_path, mode='r', encoding='utf-8') as fp:
        ppdb_pairs = json.load(fp)
        ppdb_size = len(ppdb_pairs)
        print(f"load {ppdb_size} ppdb pairs")
    train_size, test_size = lengths[0], lengths[1]
    train_pairs = list(np.random.choice(ppdb_pairs, train_size, replace=False))
    test_pairs = list(np.random.choice(ppdb_pairs, test_size, replace=True))
    P = Path(data_path)
    train_path = os.path.join(P.parent, "ppdb_train")
    test_path = os.path.join(P.parent, "ppdb_test")
    write_to_json_file(train_path, train_pairs)
    write_to_json_file(test_path, test_pairs)
    return train_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input ppdb pairs.")

    parser.add_argument("--train_size",
                        default=None,
                        type=int,
                        required=True,
                        help="Number of train sample")
    
    parser.add_argument("--test_size",
                    default=None,
                    type=int,
                    required=True,
                    help="Number of test sample")
                    

    args = parser.parse_args()

    train_path, test_path = resample_train_test(args.data_path, [args.train_size, args.test_size])
    print(f"Save train, test data at {train_path}, {test_path}")
