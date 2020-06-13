from typing import Optional
import argparse
import collections
import re
import os
import json
import numpy as np

PPDBExample = collections.namedtuple('PPDBExample', 'text1 text2 relationship')

def write_to_json_file(file_path: str, data) -> None:
    fp = open(file_path, mode='w+', encoding='utf-8')
    json.dump(data, fp, indent=4)
    fp.close()

def create_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)