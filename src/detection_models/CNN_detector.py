from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd

from src.utils.preprocessor import preprocess_payload
from src.utils.tokenizer import xss_tokenizer, clean_tokenized_payloads
from src.utils.vectorizer import xss_payloads_vectorizer, get_sorted_tokens


class XSSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class CNNDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CNNDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.layer1 = nn.Sequential(

        pass

    def forward(self, x):
        x = self.embedding(x)
        pass


num_epochs = 5
batch_size = 64
learning_rate = 0.001

vector_size = 32


train_set = pd.read_csv("../../../data/set/train_set.csv").sample(frac=1).head(100)
preprocessed_payloads = preprocess_payload(train_set['Payloads'])

class_labels = train_set['Class']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(class_labels)

tokenized_payloads = [xss_tokenizer(payload) for payload in preprocessed_payloads]

X_tfidf = xss_payloads_vectorizer(preprocessed_payloads, tokenized_payloads)
sorted_tokens = get_sorted_tokens(X_tfidf)
cleaned_tokenized_payloads = clean_tokenized_payloads(tokenized_payloads, sorted_tokens)

x_train_tensor = torch.tensor(cleaned_tokenized_payloads)
y_train_tensor = torch.tensor(y_encoded)

train_dataset = XSSDataset(x_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

print(cleaned_tokenized_payloads[0])
