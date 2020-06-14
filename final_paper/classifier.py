"""
Define classifier for semantic probe.
1. Logistic Regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class logisticRegressionClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, input_dim = 768):
        super(logisticRegressionClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, feature_vec):
        return F.log_softmax(self.linear(feature_vec), dim=1)
        # return self.linear(feature_vec)

    