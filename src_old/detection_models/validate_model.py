from src.detection_models.classes.LSTM import LSTMDetector

if __name__ == "__main__":
    import sys

import torch
import pandas as pd

from XSS_dataset import XSSDataset
from src.detection_models.classes.MLP import MLPDetector
from src.detection_models.classes.CNN import CNNDetector
from src.detection_models.utils.general import process_payloads

from src.utils.tokenizer import xss_tokenizer

from urllib.parse import unquote_plus, urlsplit, urlunsplit
import html
import re

root_dir = "../../reproduction"

vector_size = 8

sorted_tokens = pd.read_csv(root_dir+"/data/vocabulary.csv")['tokens'].tolist()
vocab_size = len(sorted_tokens)

# Load the models
CNN_model = CNNDetector(vocab_size, vector_size)
CNN_checkpoint = torch.load(root_dir+"/models/CNN_detector.pth")
CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])

MLP_model = MLPDetector(vocab_size, vector_size)
MLP_checkpoint = torch.load(root_dir+"/models/MLP_detector.pth")
MLP_model.load_state_dict(MLP_checkpoint['model_state_dict'])

LSTM_model = LSTMDetector(vocab_size, vector_size)
LSTM_checkpoint = torch.load(root_dir+"/models/LSTM_detector.pth")
LSTM_model.load_state_dict(LSTM_checkpoint['model_state_dict'])

# Validate the model
validation_set = pd.read_csv("../../data/val.csv")
validation_cleaned_tokenized_payloads = process_payloads(validation_set)[1]
validation_class_labels = validation_set['Class']
validation_dataset = XSSDataset(validation_cleaned_tokenized_payloads, validation_class_labels)

validation_set["Tokenized Payloads"] = validation_cleaned_tokenized_payloads

CNN_results = []
MLP_results = []
LSTM_results = []

for i in range(len(validation_dataset)):
    validation_data = validation_dataset[i][0][None, ...]
    CNN_output = CNN_model(validation_data)
    MLP_output = MLP_model(validation_data)
    LSTM_output = LSTM_model(validation_data)

    CNN_prediction = torch.round(CNN_output)
    MLP_prediction = torch.round(MLP_output)
    LSTM_prediction = torch.round(LSTM_output)

    # print(CNN_prediction, MLP_prediction)

    CNN_prediction = "Malicious" if CNN_prediction == 1 else "Benign"
    MLP_prediction = "Malicious" if MLP_prediction == 1 else "Benign"
    LSTM_prediction = "Malicious" if LSTM_prediction == 1 else "Benign"

    CNN_results.append(CNN_prediction)
    MLP_results.append(MLP_prediction)
    LSTM_results.append(LSTM_prediction)

validation_set['CNN_Prediction'] = CNN_results
validation_set['MLP_Prediction'] = MLP_results
validation_set['LSTM_Prediction'] = LSTM_results

validation_set["Wrong CNN Prediction"] = validation_set["CNN_Prediction"] != validation_set["Class"]
validation_set["Wrong MLP Prediction"] = validation_set["MLP_Prediction"] != validation_set["Class"]
validation_set["Wrong LSTM Prediction"] = validation_set["LSTM_Prediction"] != validation_set["Class"]


# validation_set.to_csv("../../data/validation_results.csv", index=False)


def payload_preprocess(payload):
    processed_payload = payload.lower()

    # Simplify urls to http://u
    sep = "="
    test = processed_payload.split(sep, 1)[0]
    if test != processed_payload:
        processed_payload = processed_payload.replace(test, "http://u")
    else:
        url = list(urlsplit(processed_payload))
        url[0] = "http"
        url[1] = "u"
        processed_payload = urlunsplit(url)

    # Decode HTML entities
    processed_payload = str(html.unescape(processed_payload))

    # Remove special HTML tags
    processed_payload = processed_payload.replace("<br>", "")

    # Decoding the payload
    processed_payload = unquote_plus(processed_payload)

    # Remove special characters
    processed_payload = re.sub(r'\\+', '', processed_payload)  # NOT WORKING

    # Replace numbers with 0, if not after %
    processed_payload = re.sub(r'(?<!%)\d', '0', processed_payload)
    processed_payload = re.sub(r'0+', '0', processed_payload)

    return processed_payload


# Check wrong labeled payloads
wrong_cnn_labeled = validation_set[validation_set["Wrong CNN Prediction"]]
print("Wrong CNN:", wrong_cnn_labeled.shape[0])
wrong_mlp_labeled = validation_set[validation_set["Wrong MLP Prediction"]]
print("Wrong MLP:", wrong_mlp_labeled.shape[0])
wrong_lstm_labeled = validation_set[validation_set["Wrong LSTM Prediction"]]
print("Wrong LSTM:", wrong_lstm_labeled.shape[0])

for i, row in wrong_cnn_labeled.iterrows():
    print()
    print("Payload:", row["Payloads"])
    preprocessed = payload_preprocess(row["Payloads"])
    print("Preprocessed", preprocessed)
    tokenized = xss_tokenizer(preprocessed)
    print("Tokenized:", tokenized)
    input("Press any key to continue...")
