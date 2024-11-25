import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMDetector(nn.Module):
    def __init__(self, vocab_dim, embedding_dim):
        super(LSTMDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 128, batch_first=True)

        self.hidden2tag = nn.Linear(128, 1)


    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        # print("LSTM | Expected shape: [batch, 30, 128] | Actual shape:", x.shape)

        # Take the output from the last LSTM time step
        x = x[:, -1, :]  # [batch, 128]
        # print("Last Time Step | Expected shape: [batch, 128] | Actual shape:", x.shape)

        # Pass the last hidden state through the linear layer
        tag_space = self.hidden2tag(x)  # [batch, 2]
        # print("Hidden2Tag | Expected shape: [batch, 2] | Actual shape:", tag_space.shape)

        # Apply softmax to get class probabilities
        # tag_scores = F.softmax(tag_space, dim=1)  # Apply along the class dimension
        tag_scores = F.sigmoid(tag_space)
        # print("Softmax | Expected shape: [batch, 2] | Actual shape:", tag_scores.shape)

        return tag_scores
