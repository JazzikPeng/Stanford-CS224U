"""
Featurizer as the input to classification model
Construct Features From BERT.forward() ouput
"""
import numpy as np
import torch

def cls_featurizer(encoder_output):
    """Return CLS Token"""
    final_hidden_state, cls_output = encoder_output
    return cls_output

def avg_pooling_featurizer(hidden_state):
    """Construct Average pooling"""
    return torch.mean(hidden_state, axis=1)

