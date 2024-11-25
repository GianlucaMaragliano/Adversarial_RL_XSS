import torch
import torch.nn as nn
import torch.nn.functional as F

from src.detection_models.XSS_dataset import XSSDataset


class MLPDetector(nn.Module):
    def __init__(self, vocab_dim, embedding_dim):
        super(MLPDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        # First hidden layer
        self.hidden1 = nn.Linear(XSSDataset.MAX_LENGTH * embedding_dim, 30)
        # Second hidden layer
        self.hidden2 = nn.Linear(30, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # print("Input | Expected shape: [batch, 30, 1] | Actual shape:", x.shape)
        x = self.embedding(x)
        # print("Embedded | Expected shape: [batch, 30, 8] | Actual shape:", x.shape)
        x = x.permute(0, 2, 1)
        # print("Permuted | Expected shape: [batch, 8, 30] | Actual shape:", x.shape)
        x = self.flatten(x)
        # print("Flattened | Expected shape: [batch, 8 * 30] | Actual shape:", x.shape)
        x = F.relu(self.hidden1(x))
        # print("First Hidden | Expected shape: [batch, 30] | Actual shape:", x.shape)
        x = self.hidden2(x)
        # print("Second Hidden | Expected shape: [batch, 15] | Actual shape:", x.shape)
        x = F.sigmoid(x)
        # print("Output | Expected shape: [batch, 1] | Actual shape:", x.shape)
        return x
