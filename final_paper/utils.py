from typing import Optional
import argparse
import random
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

def glove2dict(src_filename):
    """GloVe Reader.

    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.

    Returns
    -------
    dict
        Mapping words to their GloVe vectors as `np.array`.

    """
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '840B.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.strip().split()
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data


def convert_ppdb_pairs_to_input_text(pair):
    SEP_TOKEN = "[SEP]"
    text1, text2, rel = pair['text1'], pair['text2'], pair['relationship']
    label = 0 if rel=='Negative' else 1
    text = text1 + " " + SEP_TOKEN + " "+ text2
    return text, label


def fix_random_seeds(
        seed=42,
        set_system=True,
        set_torch=True,
        set_tensorflow=True,
        set_torch_cudnn=True):
    """Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_tensorflow : bool
        Whether to set `tf.random.set_random_seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    set_torch_cudnn: bool
        Flag for whether to enable cudnn deterministic mode.
        Note that deterministic mode can have a performance impact, depending on your model.
        https://pytorch.org/docs/stable/notes/randomness.html

    Notes
    -----
    The function checks that PyTorch and TensorFlow are installed
    where the user asks to set seeds for them. If they are not
    installed, the seed-setting instruction is ignored. The intention
    is to make it easier to use this function in environments that lack
    one or both of these libraries.

    Even though the random seeds are explicitly set,
    the behavior may still not be deterministic (especially when a
    GPU is enabled), due to:

    * CUDA: There are some PyTorch functions that use CUDA functions
    that can be a source of non-determinism:
    https://pytorch.org/docs/stable/notes/randomness.html

    * PYTHONHASHSEED: On Python 3.3 and greater, hash randomization is
    turned on by default. This seed could be fixed before calling the
    python interpreter (PYTHONHASHSEED=0 python test.py). However, it
    seems impossible to set it inside the python program:
    https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program

    """
    # set system seed
    if set_system:
        np.random.seed(seed)
        random.seed(seed)

    # set torch seed
    if set_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)

    # set torch cudnn backend
    if set_torch_cudnn:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # set tf seed
    # if set_tensorflow:
    #     try:
    #         from tensorflow.compat.v1 import set_random_seed as set_tf_seed
    #     except ImportError:
    #         from tensorflow.random import set_seed as set_tf_seed
    #     except ImportError:
    #         pass
    #     else:
    #         set_tf_seed(seed)
