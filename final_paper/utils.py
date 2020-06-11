from typing import Optional
import argparse
import collections
import re
import json
import numpy as np

PPDBExample = collections.namedtuple('PPDBExample', 'text1 text2 relationship')

def write_examples_to_json_file(file_path: str, examples: [PPDBExample]) -> None:
    fp = open(file_path, mode='w+', encoding='utf-8')
    examples = [example._asdict() for example in examples]
    json.dump(examples, fp, indent=4)
    fp.close()

    