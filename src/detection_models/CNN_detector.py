if __name__ == "__main__":
    import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.preprocessor import preprocess_payload
from src.utils.tokenizer import xss_tokenizer, clean_tokenized_payloads
from src.utils.vectorizer import xss_payloads_vectorizer, get_sorted_tokens


class XSSDataset(Dataset):
    class_encoder = LabelEncoder()
    features_encoder = LabelEncoder()

    def __init__(self, features, labels):
        encoded_labels = self.class_encoder.fit_transform(labels)
        self.labels = torch.tensor(encoded_labels)
        features = [tokens[:200] for tokens in features]
        features = [tokens + ['None'] * (200 - len(tokens)) for tokens in features]
        encoded_features = [self.features_encoder.fit_transform(tokens) for tokens in features]
        self.features = torch.tensor(encoded_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class CNNDetector(nn.Module):
    def __init__(self, vocab_dim, embedding_dim):
        super(CNNDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_dim, embedding_dim)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=vector_size, out_channels=64, kernel_size=64),
            # batch, 64, 200
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=64),
            # batch, 64, 200
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
        # batch, 64, 100
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=128),
            # batch, 128, 100
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=128),
            # batch, 128, 100
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
        # batch, 128, 50
        self.flatten = nn.Flatten()
        self.conv1_1 = nn.Conv1d(in_channels=vector_size, out_channels=64, kernel_size=64)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # self.conv2_1 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5)
        self.fc1 = nn.Linear(128 * 50, 1)

    def forward(self, x):
        print("Input | Expected shape: batch, 200, 1 | Actual shape:", x.shape)
        x = self.embedding(x)
        print("Embedded | Expected shape: batch, 200, 32 | Actual shape:", x.shape)
        x = x.permute(0, 2, 1)  # swap channels
        print("Permuted | Expected shape: batch, 32, 200 | Actual shape:", x.shape)
        x = F.relu(self.conv1_1(x))
        print("First Conv | Expected shape: batch, 64, 200 | Actual shape:", x.shape)
        x = F.relu(self.conv1_2(x))
        print("Second Conv | Expected shape: batch, 64, 200 | Actual shape:", x.shape)
        x = x.view(-1, 128 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = F.sigmoid(x)
        return x


# Set the seed value all over the place to make this reproducible.
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

vector_size = 32

train_set = pd.read_csv("../../data/train.csv").sample(frac=1)
preprocessed_payloads = preprocess_payload(train_set['Payloads'])

class_labels = train_set['Class']
class_encoder = LabelEncoder()
y_encoded = class_encoder.fit_transform(class_labels)
# print("Class labels:", class_labels[:10].to_list())
# print("Encoded y:", y_encoded[:10])

# Tokenize Payloads
tokenized_payloads = [xss_tokenizer(payload) for payload in preprocessed_payloads]

# Vectorize Payloads
X_tfidf = xss_payloads_vectorizer(preprocessed_payloads, tokenized_payloads)
sorted_tokens = get_sorted_tokens(X_tfidf)
cleaned_tokenized_payloads = clean_tokenized_payloads(tokenized_payloads, sorted_tokens)
# cleaned_tokenized_payloads = [tokens[:200] for tokens in cleaned_tokenized_payloads]
# cleaned_tokenized_payloads = [tokens + ['None'] * (200 - len(tokens)) for tokens in cleaned_tokenized_payloads]
print("Input length:", len(cleaned_tokenized_payloads))

# Append None to the sorted tokens
sorted_tokens.append('None')

# Encode tokens
x_encoder = LabelEncoder()
encoded_tokenized_payloads = [x_encoder.fit_transform(tokens) for tokens in cleaned_tokenized_payloads]
# print("Clean tokens:", cleaned_tokenized_payloads[0])
# print("Encoded tokens:", encoded_tokenized_payloads[0])

# Embedding layer
vocab_size = len(sorted_tokens)

# embedding_layer = nn.Embedding(vocab_size, vector_size)
#
# test_embedded_x = embedding_layer(torch.tensor(encoded_tokenized_payloads[0]))
# print("Embedding test:", test_embedded_x)
# print("Embedding test shape:", test_embedded_x.shape)

# embedding_layer(torch.tensor(encoded_tokenized_payloads[0]))
#
# embedded_x = [embedding_layer(torch.tensor(tokens)) for tokens in encoded_tokenized_payloads]
# print(embedded_x[0])

# Truncate and pad the input sequences
MAX_LENGTH = 200
# padded_x = nn.utils.rnn.pad_sequence(embedded_x, batch_first=True, padding_value=0).narrow(1, 0, MAX_LENGTH)

# print(padded_x[0])
# print(padded_x.shape)

# x_train_tensor = embedded_x
# print(x_train_tensor.shape)

train_dataset = XSSDataset(cleaned_tokenized_payloads, class_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = CNNDetector(vocab_size, vector_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x_f, y_f) in enumerate(train_loader):
        print(x_f.shape)

        x_f = x_f.to(device)
        y_f = y_f.to(device)

        # Forward pass
        outputs = model(x_f)
        loss = criterion(outputs, y_f)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
print("Finished Training")
