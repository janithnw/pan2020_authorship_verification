import torch
import torch.nn as nn
from torch.utils import data

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(self.batchnorm1(out))
        out = self.fc2(out)
        out = self.act(out)
        return out