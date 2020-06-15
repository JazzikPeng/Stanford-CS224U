"""
Featurizer as the input to classification model
Construct Features From BERT.forward() ouput
"""
import numpy as np

def cls_featurizer(encoder_output):
    """Return CLS Token"""
    final_hidden_state, cls_output = encoder_output
    return cls_output

def avg_pooling_featurizer(encoder_output):
    """Construct Average pooling"""
    