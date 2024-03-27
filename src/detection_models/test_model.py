if __name__ == "__main__":
    import sys

import torch
import pandas as pd

from XSS_dataset import XSSDataset
from src.detection_models.classes.MLP import MLPDetector
from src.detection_models.classes.CNN import CNNDetector
from src.detection_models.utils.general import process_payloads

from src.utils.tokenizer import xss_tokenizer

vector_size = 8

train_set = pd.read_csv("../../data/train.csv").sample(frac=1)
sorted_tokens, train_cleaned_tokenized_payloads = process_payloads(train_set)
train_class_labels = train_set['Class']
vocab_size = len(sorted_tokens)

CNN_model = CNNDetector(vocab_size, vector_size)
CNN_checkpoint = torch.load("../../models/CNN_detector.pth")
CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])

MLP_model = MLPDetector(vocab_size, vector_size)
MLP_checkpoint = torch.load("../../models/MLP_detector.pth")
MLP_model.load_state_dict(MLP_checkpoint['model_state_dict'])

# Test the model
test_set = pd.read_csv("../../data/test.csv")
test_cleaned_tokenized_payloads = process_payloads(test_set)[1]
test_class_labels = test_set['Class']
test_dataset = XSSDataset(test_cleaned_tokenized_payloads, test_class_labels)

test_set["Tokenized Payloads"] = test_cleaned_tokenized_payloads

CNN_results = []
MLP_results = []

for i in range(len(test_dataset)):
    test_data = test_dataset[i][0][None, ...]
    CNN_output = CNN_model(test_data)
    MLP_output = MLP_model(test_data)

    CNN_prediction = torch.round(CNN_output)
    MLP_prediction = torch.round(MLP_output)

    CNN_prediction = "Malicious" if CNN_prediction == 1 else "Benign"
    MLP_prediction = "Malicious" if MLP_prediction == 1 else "Benign"

    CNN_results.append(CNN_prediction)
    MLP_results.append(MLP_prediction)

test_set['CNN_Prediction'] = CNN_results
test_set['MLP_Prediction'] = MLP_results

test_set["Wrong CNN Prediction"] = test_set["CNN_Prediction"] != test_set["Class"]
test_set["Wrong MLP Prediction"] = test_set["MLP_Prediction"] != test_set["Class"]

# Display the results
wrong_CNN_predictions = test_set[test_set["Wrong CNN Prediction"]]
wrong_MLP_predictions = test_set[test_set["Wrong MLP Prediction"]]
print("Wrong CNN predictions:", f"{wrong_CNN_predictions.shape[0]}/{test_set.shape[0]}")
print("Wrong MLP predictions:", f"{wrong_MLP_predictions.shape[0]}/{test_set.shape[0]}")

print("CNN Accuracy:", (1 - wrong_CNN_predictions.shape[0] / test_set.shape[0]) * 100, "%")
print("MLP Accuracy:", (1 - wrong_MLP_predictions.shape[0] / test_set.shape[0]) * 100, "%")
