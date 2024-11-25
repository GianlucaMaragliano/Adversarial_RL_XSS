
from utils.utils import init_argument_parser
from utils.path_utils import get_last_run_number
import os
import pandas as pd
from utils.utils import init_argument_parser
import torch
from utils.preprocess import process_payloads
from datasets.xss_dataset import XSSDataset
from models.CNN import CNNDetector
from models.MLP import MLPDetector
from models.LSTM import LSTMDetector
import json 

def val_epoch(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    n_correct = 0
    n_samples = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    with torch.no_grad():
        for i, (payloads, labels) in enumerate(val_loader):
            payloads = payloads.to(device)
            labels = labels.to(torch.float32).to(device)
            outputs = model(payloads)
            predicted = torch.round(outputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            # Update TP, TN, FP, FN directly on tensors
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()
 
        accuracy = 100.0 * n_correct / n_samples
        print(n_samples)
        print("TP:",TP)
        print("FP:",FP)
        print("TN:",TN)
        print("FN:",FN)
 
        # Calculate precision, recall, and F2 score
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f2_score = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0
    
    return total_loss / len(val_loader), accuracy, precision, recall, f2_score

def test(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = pd.read_csv(opt.trainset).sample(frac=1)
    sorted_tokens, train_cleaned_tokenized_payloads = process_payloads(train_set, percentage=opt.vocab_size)
    train_class_labels = train_set['Class']

    # Get the vocab size
    vocab_size = len(sorted_tokens)
    # Test set
    test_set = pd.read_csv(opt.testset).sample(frac=1)
    test_cleaned_tokenized_payloads = process_payloads(test_set)[1]
    test_class_labels = test_set['Class']
    test_dataset = XSSDataset(test_cleaned_tokenized_payloads, test_class_labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


    # Load the checkpoint
    checkpoint_path = os.path.join(opt.checkpoint_folder, opt.checkpoint_name)
    
    if opt.model == 'mlp':
        model_architecture = MLPDetector
    elif opt.model == 'cnn':
        model_architecture = CNNDetector
    elif opt.model == 'lstm':
        model_architecture = LSTMDetector
    else:
        raise ValueError("Model not supported")

    model = model_architecture(vocab_size, opt.embedding_dim, XSSDataset.MAX_LENGTH).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    _, accuracy, precision, recall, f2_score = val_epoch(test_loader, model, torch.nn.BCELoss(), device)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F2score: {f2_score}")
    # Save the results
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f2score": f2_score,
        "model": opt.model,
        "run": opt.checkpoint_folder
    }
    results_path = os.path.join(opt.checkpoint_folder, opt.test_file_name)
    with open(results_path, 'w') as f:
        json.dump(results, f)


   

def add_parse_arguments(parser):

    parser.add_argument('--testset', type=str, default="data/detectors/test.csv", help='Testing dataset')
    parser.add_argument('--trainset', type=str, default="data/detectors/train.csv", help='Training dataset')
    parser.add_argument('--vocab_size', type=float, default=0.1, help='Percentage of the most common tokens to keep in the vocab')

    parser.add_argument('--checkpoint_folder', type=str, required='True', help='Folder of the run with the checkpoint')
    parser.add_argument('--test_file_name', type=str, default="test_results.json", help='Name of the json results file')
    parser.add_argument('--checkpoint_name', type=str, default="checkpoint.pth", help='Checkpoint name')
    parser.add_argument('--model', type=str, default = "mlp", help='mlp | cnn | lstm')
    parser.add_argument('--embedding_dim', type=float, default=8, help='size of the embeddings')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    test(opt)

if __name__ == '__main__':
    main()