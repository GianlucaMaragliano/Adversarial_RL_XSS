
if __name__ == "__main__":
    import sys

import torch
import pandas as pd
from torchinfo import summary

from XSS_dataset import XSSDataset
from src.detection_models.classes.MLP import MLPDetector
from src.detection_models.classes.CNN import CNNDetector
from src.detection_models.classes.LSTM import LSTMDetector
from src.detection_models.utils.general import process_payloads
from src.utils.tokenizer import xss_tokenizer

root_dir = "../../reproduction"

vector_size = 8

sorted_tokens = pd.read_csv(root_dir+"/data/vocabulary.csv")['tokens'].tolist()
vocab_size = len(sorted_tokens)

# Load the models
CNN_model = CNNDetector(vocab_size, vector_size)
CNN_checkpoint = torch.load("../../reproduction/models/CNN_detector.pth")
CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])

MLP_model = MLPDetector(vocab_size, vector_size)
MLP_checkpoint = torch.load("../../reproduction/models/MLP_detector.pth")
MLP_model.load_state_dict(MLP_checkpoint['model_state_dict'])

LSTM_model = LSTMDetector(vocab_size, vector_size)
LSTM_checkpoint = torch.load("../../reproduction/models/LSTM_detector.pth")
LSTM_model.load_state_dict(LSTM_checkpoint['model_state_dict'])

# Test the model
test_set = pd.read_csv("../../data/test.csv")
test_cleaned_tokenized_payloads = process_payloads(test_set)[1]
test_class_labels = test_set['Class']
test_dataset = XSSDataset(test_cleaned_tokenized_payloads, test_class_labels)

test_set["Tokenized Payloads"] = test_cleaned_tokenized_payloads

CNN_results = []
MLP_results = []
LSTM_results = []

for i in range(len(test_dataset)):
    test_data = test_dataset[i][0][None, ...]
    CNN_output = CNN_model(test_data)
    MLP_output = MLP_model(test_data)
    LSTM_output = LSTM_model(test_data)

    CNN_prediction = torch.round(CNN_output)
    MLP_prediction = torch.round(MLP_output)
    LSTM_prediction = torch.round(LSTM_output)

    CNN_prediction = "Malicious" if CNN_prediction == 1 else "Benign"
    MLP_prediction = "Malicious" if MLP_prediction == 1 else "Benign"
    LSTM_prediction = "Malicious" if LSTM_prediction == 1 else "Benign"

    CNN_results.append(CNN_prediction)
    MLP_results.append(MLP_prediction)
    LSTM_results.append(LSTM_prediction)

test_set['CNN_Prediction'] = CNN_results
test_set['MLP_Prediction'] = MLP_results
test_set['LSTM_Prediction'] = LSTM_results

test_set["Wrong CNN Prediction"] = test_set["CNN_Prediction"] != test_set["Class"]
test_set["Wrong MLP Prediction"] = test_set["MLP_Prediction"] != test_set["Class"]
test_set["Wrong LSTM Prediction"] = test_set["LSTM_Prediction"] != test_set["Class"]

# Display the results
wrong_CNN_predictions = test_set[test_set["Wrong CNN Prediction"]]
wrong_MLP_predictions = test_set[test_set["Wrong MLP Prediction"]]
wrong_LSTM_predictions = test_set[test_set["Wrong LSTM Prediction"]]
print("Wrong CNN predictions:", f"{wrong_CNN_predictions.shape[0]}/{test_set.shape[0]}")
print("Wrong MLP predictions:", f"{wrong_MLP_predictions.shape[0]}/{test_set.shape[0]}")
print("Wrong LSTM predictions:", f"{wrong_LSTM_predictions.shape[0]}/{test_set.shape[0]}")
print()

print("CNN Accuracy:", round((1 - wrong_CNN_predictions.shape[0] / test_set.shape[0]) * 100, 2), "%")

CNN_true_positives = test_set[(test_set["CNN_Prediction"] == "Malicious") & (test_set["Class"] == "Malicious")].shape[0]
CNN_true_negatives = test_set[(test_set["CNN_Prediction"] == "Benign") & (test_set["Class"] == "Benign")].shape[0]
CNN_false_positives = test_set[(test_set["CNN_Prediction"] == "Malicious") & (test_set["Class"] == "Benign")].shape[0]
CNN_false_negatives = test_set[(test_set["CNN_Prediction"] == "Benign") & (test_set["Class"] == "Malicious")].shape[0]

CNN_precision = CNN_true_positives / (CNN_true_positives + CNN_false_positives)
CNN_recall = CNN_true_positives / (CNN_true_positives + CNN_false_negatives)
CNN_f1 = 2 * (CNN_precision * CNN_recall) / (CNN_precision + CNN_recall)

print("CNN Precision:", f"{(CNN_precision * 100):.2f}%")
print("CNN Recall:", f"{(CNN_recall * 100):.2f}%")
print("CNN F1 Score:", f"{(CNN_f1 * 100):.2f}%")
print()

print("MLP Accuracy:", round((1 - wrong_MLP_predictions.shape[0] / test_set.shape[0]) * 100, 2), "%")

MLP_true_positives = test_set[(test_set["MLP_Prediction"] == "Malicious") & (test_set["Class"] == "Malicious")].shape[0]
MLP_true_negatives = test_set[(test_set["MLP_Prediction"] == "Benign") & (test_set["Class"] == "Benign")].shape[0]
MLP_false_positives = test_set[(test_set["MLP_Prediction"] == "Malicious") & (test_set["Class"] == "Benign")].shape[0]
MLP_false_negatives = test_set[(test_set["MLP_Prediction"] == "Benign") & (test_set["Class"] == "Malicious")].shape[0]

MLP_precision = MLP_true_positives / (MLP_true_positives + MLP_false_positives)
MLP_recall = MLP_true_positives / (MLP_true_positives + MLP_false_negatives)
MLP_f1 = 2 * (MLP_precision * MLP_recall) / (MLP_precision + MLP_recall)

print("MLP Precision:", f"{(MLP_precision * 100):.2f}%")
print("MLP Recall:", f"{(MLP_recall * 100):.2f}%")
print("MLP F1 Score:", f"{(MLP_f1 * 100):.2f}%")
print()

print("LSTM Accuracy:", round((1 - wrong_LSTM_predictions.shape[0] / test_set.shape[0]) * 100, 2), "%")

LSTM_true_positives = test_set[(test_set["LSTM_Prediction"] == "Malicious") & (test_set["Class"] == "Malicious")].shape[0]
LSTM_true_negatives = test_set[(test_set["LSTM_Prediction"] == "Benign") & (test_set["Class"] == "Benign")].shape[0]
LSTM_false_positives = test_set[(test_set["LSTM_Prediction"] == "Malicious") & (test_set["Class"] == "Benign")].shape[0]
LSTM_false_negatives = test_set[(test_set["LSTM_Prediction"] == "Benign") & (test_set["Class"] == "Malicious")].shape[0]

LSTM_precision = LSTM_true_positives / (LSTM_true_positives + LSTM_false_positives)
LSTM_recall = LSTM_true_positives / (LSTM_true_positives + LSTM_false_negatives)
LSTM_f1 = 2 * (LSTM_precision * LSTM_recall) / (LSTM_precision + LSTM_recall)

print("LSTM Precision:", f"{(LSTM_precision * 100):.2f}%")
print("LSTM Recall:", f"{(LSTM_recall * 100):.2f}%")
print("LSTM F1 Score:", f"{(LSTM_f1 * 100):.2f}%")
print()

summary(CNN_model)  # CNN model summary
summary(MLP_model)  # MLP model summary
summary(LSTM_model)  # LSTM model summary


# test_case = "<script>alert('hi')</script>"
# test_case_df = pd.DataFrame([test_case], columns=['Payloads'])
# test_case_df['Class'] = "Malicious"
#
# _, test_case_tokenized = process_payloads(test_case_df, sorted_tokens)
# print(test_case_tokenized)
# test_case_dataset = XSSDataset(test_case_tokenized, test_case_df['Class'])
# test_case_data = test_case_dataset[0][0][None, ...]
#
# MLP_output = MLP_model(test_case_data)
