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

"""
Define classifier for semantic probe.
1. MLP
"""
class MLP1Classifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, hidden_size=512, input_dim = 768):
        super(MLP1Classifier, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_size)
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, feature_vec):
        layer1 = self.linear(feature_vec)
        layer2 = self.output(layer1)
        return F.log_softmax(layer2, dim=1)
        # return self.linear(feature_vec)
                   