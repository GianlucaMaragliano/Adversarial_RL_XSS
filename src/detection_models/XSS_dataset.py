import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset


class XSSDataset(Dataset):
    class_encoder = LabelEncoder()
    features_encoder = LabelEncoder()
    MAX_LENGTH = 30

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