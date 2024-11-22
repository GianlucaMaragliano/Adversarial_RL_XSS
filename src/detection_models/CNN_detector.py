if __name__ == "__main__":
    import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from utils.general import process_payloads
from XSS_dataset import XSSDataset
from src.detection_models.classes.CNN import CNNDetector


# Set the seed value all over the place to make this reproducible.
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 150
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
    n_correct = 0
    n_samples = 0
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

        predicted = torch.round(outputs)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        if i % 250 == 0:
            last_loss = running_loss / 250
            print(f"[{epoch_index + 1}, {i + 1}] loss: {last_loss}")
            running_loss = 0.0
    accuracy = 100.0 * n_correct / n_samples
    print('ACCURACY TRAIN: {}'.format(accuracy))
    writer.add_scalar('Loss/train', last_loss, 1 + epoch_index)
    writer.add_scalar('Accuracy/train', accuracy, 1 + epoch_index)
    return last_loss


writer = SummaryWriter('../../runs/CNN_detector')

epoch_number = 0

best_vloss = 1_000_000.

accuracy = 0.0
last_accuracy = 0.0
total_repeat = 0

for epoch in range(num_epochs):
    last_accuracy = accuracy

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

            # predicted = torch.where(voutputs > 0.51, 1, 0)
            predicted = torch.round(voutputs)
            # print(voutputs, vlabels)

            # Accuracy on validation
            # print(predicted, vlabels)
            n_samples += vlabels.size(0)
            n_correct += (predicted == vlabels).sum().item()
    accuracy = 100.0 * n_correct / n_samples
    avg_vloss = running_vloss / (i + 1)

    writer.add_scalar('Loss/valid', avg_vloss, 1 + epoch_number)
    writer.add_scalar('Accuracy/valid', accuracy, 1 + epoch_number)

    # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('ACCURACY VALIDATION: {}'.format(accuracy))
    print()

    if accuracy <= last_accuracy:
        total_repeat += 1
        if total_repeat == 5:
            print("Early stopping")
            break
    else:
        total_repeat = 0

    epoch_number += 1
print("Finished training")

# Save the model
torch.save({'epoch': epoch_number, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss}, "../../reproduction/models/CNN_detector.pth")


