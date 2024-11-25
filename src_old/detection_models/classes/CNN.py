import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDetector(nn.Module):
    def __init__(self, vocab_dim, embedding_dim):
        super(CNNDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.flatten = nn.Flatten()
        self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 2, 1)

    def forward(self, x):
        # print("Input | Expected shape: [batch, 30, 1] | Actual shape:", x.shape)
        x = self.embedding(x)
        # print("Embedded | Expected shape: [batch, 30, 8] | Actual shape:", x.shape)
        x = x.permute(0, 2, 1)  # swap channels
        # print("Permuted | Expected shape: [batch, 8, 30] | Actual shape:", x.shape)
        x = F.relu(self.conv1_1(x))
        # First Conv Layer
        # print("First Conv | Expected shape: [batch, 64, 30] | Actual shape:", x.shape)
        x = F.relu(self.conv1_2(x))
        # print("Second Conv | Expected shape: [batch, 64, 30] | Actual shape:", x.shape)
        x = F.relu(self.pool(x))
        # print("First Pool | Expected shape: [batch, 64, 15] | Actual shape:", x.shape)
        # Second Conv Layer
        x = F.relu(self.conv2_1(x))
        # print("Third Conv | Expected shape: [batch, 128, 15] | Actual shape:", x.shape)
        x = F.relu(self.conv2_2(x))
        # print("Fourth Conv | Expected shape: [batch, 128, 15] | Actual shape:", x.shape)
        x = F.relu(self.pool(x))
        # print("Second Pool | Expected shape: [batch, 128, 50] | Actual shape:", x.shape)
        x = self.flatten(x)
        # print("Flattened | Expected shape: [batch, 128 * 41] | Actual shape:", x.shape)
        x = self.fc1(x)
        # print("Fully Connected | Expected shape: [batch, 1] | Actual shape:", x.shape)
        x = F.sigmoid(x)
        return x
