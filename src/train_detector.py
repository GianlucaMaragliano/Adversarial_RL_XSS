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
import torch.nn as nn
import json

def train_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for i, (payloads, labels) in enumerate(train_loader):
        payloads = payloads.to(device)
        labels = labels.to(torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(payloads)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def val_epoch(val_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    n_correct = 0
    n_samples = 0
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
        accuracy = 100.0 * n_correct / n_samples
    return total_loss / len(val_loader), accuracy

def train(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = pd.read_csv(opt.trainset).sample(frac=1)
    sorted_tokens = pd.read_csv(opt.vocabulary)['tokens'].tolist()

    _, train_cleaned_tokenized_payloads = process_payloads(train_set, sorted_tokens=sorted_tokens)
    train_class_labels = train_set['Class']

    # Get the vocab size
    vocab_size = len(sorted_tokens)

    # Create a dataset and dataloader
    train_dataset = XSSDataset(train_cleaned_tokenized_payloads, train_class_labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True)

    # Validation set
    validation_set = pd.read_csv(opt.valset).sample(frac=1)
    validation_cleaned_tokenized_payloads = process_payloads(validation_set)[1]
    validation_class_labels = validation_set['Class']
    validation_dataset = XSSDataset(validation_cleaned_tokenized_payloads, validation_class_labels)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=opt.batch_size, shuffle=False)

    if opt.model == 'mlp':
        model_architecture = MLPDetector
    elif opt.model == 'cnn':
        model_architecture = CNNDetector
    elif opt.model == 'lstm':
        model_architecture = LSTMDetector
    else:
        raise ValueError("Model not supported")
    
    model = model_architecture(vocab_size, opt.embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

    runs_folder = os.path.join(opt.runs_folder, opt.model, str(int(opt.vocab_size*100)))
    os.makedirs(runs_folder, exist_ok=True)

    opt_dict = vars(opt)

    os.makedirs(runs_folder, exist_ok=True)
    last_run = get_last_run_number(runs_folder)
    runs_folder = os.path.join(runs_folder, f"run_{last_run + 1}")
    os.makedirs(runs_folder, exist_ok=True)
    opt_dict["runs_folder"] = runs_folder
    opt_dict["checkpoint"] = os.path.join(runs_folder, 'checkpoint.pth')
    with open(os.path.join(runs_folder, 'config.json'), 'w') as f:
        json.dump(opt_dict, f,ensure_ascii=False,indent=4)
    epochs_without_improvement = 0
    
    for epoch in range(opt.epochs):
        train_loss = train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_accuracy = val_epoch(validation_loader, model, criterion, device)

        print(f"Epoch {epoch} - Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

        if epoch == 0 or val_loss < best_val_loss:
            epochs_without_improvement = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(runs_folder, 'checkpoint.pth'))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > opt.patience:
                print("Early stopping")
                break

def add_parse_arguments(parser):

    parser.add_argument('--trainset', type=str, required = True, help='Training dataset')
    parser.add_argument('--valset', type=str, required = True, help='Validation dataset')
    parser.add_argument('--vocabulary', type=str, required = True, help='Vocaboulary file')

    parser.add_argument('--model', type=str, default='mlp', help='mlp | cnn | lstm')
    parser.add_argument('--runs_folder', type=str, default="runs", help='Runs Folder')
    parser.add_argument('--vocab_size', type=float, default=0.1, help='Percentage of the most common tokens to keep in the vocab')


    #hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--embedding_dim', type=float, default=8, help='size of the embeddings')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    train(opt)

if __name__ == '__main__':
    main()