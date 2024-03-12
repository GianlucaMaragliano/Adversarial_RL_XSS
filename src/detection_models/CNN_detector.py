if __name__ == "__main__":
    import sys

from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils.general import process_payloads


class XSSDataset(Dataset):
    class_encoder = LabelEncoder()
    features_encoder = LabelEncoder()
    MAX_LENGTH = 35

    def __init__(self, features, labels):
        encoded_labels = self.class_encoder.fit_transform(labels)
        self.labels = torch.tensor(encoded_labels).unsqueeze(dim=-1)
        features = [tokens[:self.MAX_LENGTH] for tokens in features]
        features = [tokens + ['None'] * (self.MAX_LENGTH - len(tokens)) for tokens in features]
        encoded_features = [self.features_encoder.fit_transform(tokens) for tokens in features]
        encoded_features = np.array(encoded_features)
        self.features = torch.tensor(encoded_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class CNNDetector(nn.Module):
    def __init__(self, vocab_dim, embedding_dim):
        super(CNNDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.flatten = nn.Flatten()
        self.conv1_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=4)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2_1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 3, 1)

    def forward(self, x):
        # print("Input | Expected shape: batch, 200, 1 | Actual shape:", x.shape)
        x = self.embedding(x)
        # print("Embedded | Expected shape: batch, 200, 32 | Actual shape:", x.shape)
        x = x.permute(0, 2, 1)  # swap channels
        # print("Permuted | Expected shape: batch, 32, 200 | Actual shape:", x.shape)
        x = F.relu(self.conv1_1(x))
        # First Conv Layer
        # print("First Conv | Expected shape: batch, 64, 200 | Actual shape:", x.shape)
        x = F.relu(self.conv1_2(x))
        # print("Second Conv | Expected shape: batch, 64, 200 | Actual shape:", x.shape)
        x = F.relu(self.pool(x))
        # print("First Pool | Expected shape: batch, 64, 100 | Actual shape:", x.shape)
        # Second Conv Layer
        x = F.relu(self.conv2_1(x))
        # print("Third Conv | Expected shape: batch, 128, 100 | Actual shape:", x.shape)
        x = F.relu(self.conv2_2(x))
        # print("Fourth Conv | Expected shape: batch, 128, 100 | Actual shape:", x.shape)
        x = F.relu(self.pool(x))
        # print("Second Pool | Expected shape: batch, 128, 50 | Actual shape:", x.shape)
        x = self.flatten(x)
        # print("Flattened | Expected shape: batch, 128 * 41 | Actual shape:", x.shape)
        x = F.relu(self.fc1(x))
        # print("Fully Connected | Expected shape: batch, 1 | Actual shape:", x.shape)
        x = F.sigmoid(x)
        return x


# Set the seed value all over the place to make this reproducible.
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 16
learning_rate = 0.001

vector_size = 8

train_set = pd.read_csv("../../data/train.csv").sample(frac=1)
sorted_tokens, train_cleaned_tokenized_payloads = process_payloads(train_set)
print("Input length:", len(train_cleaned_tokenized_payloads))

train_class_labels = train_set['Class']

# Get the vocab size
vocab_size = len(sorted_tokens)

# Create a dataset and dataloader
train_dataset = XSSDataset(train_cleaned_tokenized_payloads, train_class_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Validation set
validation_set = pd.read_csv("../../data/val.csv").sample(frac=1)
validation_cleaned_tokenized_payloads = process_payloads(validation_set)[1]
validation_class_labels = validation_set['Class']
validation_dataset = XSSDataset(validation_cleaned_tokenized_payloads, validation_class_labels)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

# Model
model = CNNDetector(vocab_size, vector_size).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0

    for i, (payloads, labels) in enumerate(train_loader):
        # Zero the parameter gradients
        labels = labels.to(torch.float32)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(payloads)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f"[{epoch_index + 1}, {i + 1}] loss: {last_loss}")
            running_loss = 0.0

    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vlabels = vlabels.to(torch.float32)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

            voutputs = torch.where(voutputs > 0.51, 1, 0)
            # print(voutputs, vlabels)

            # Accuracy on validation
            _, predicted = torch.max(voutputs.data, 1)
            # print(predicted, vlabels)
            n_samples += vlabels.size(0)
            n_correct += (voutputs == vlabels).sum().item()
    accuracy = 100.0 * n_correct / n_samples

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('ACCURACY: {}'.format(accuracy))

    # Track the best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        # torch.save(model.state_dict(), model_path)

    epoch_number += 1

# Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (x_f, y_f) in enumerate(train_loader):
#         x_f = x_f.to(device)
#         y_f = y_f.to(device).to(torch.float32)
#
#         # Forward pass
#         outputs = model(x_f)
#         loss = criterion(outputs, y_f)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 100 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
# print("Finished Training")
