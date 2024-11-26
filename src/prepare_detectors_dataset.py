from utils.utils import init_argument_parser
from utils.dataset_utils import filter_dataset_with_oracle, filter_detector_dataset, dataset_split
import pandas as pd
import os 
import numpy as np

def prepare_detectors_dataset(opt):
    #fix the seed for reproducibility
    np.random.seed(opt.seed)

    df = pd.read_csv(opt.dataset, encoding="ISO-8859-1")
    df_filtered = filter_detector_dataset(df)
    df_filtered.to_csv(os.path.join(opt.save_path, "filtered.csv"), index=False)

    df_filtered_with_oracle = filter_dataset_with_oracle(df_filtered, opt.endpoint)
    df_filtered_with_oracle.to_csv(os.path.join(opt.save_path, "filtered_oracle.csv"), index=False)
    trainval_set, testset = dataset_split(df_filtered_with_oracle, opt.trainval_percentage)
    trainset, valset = dataset_split(trainval_set, opt.train_percentage)
    trainset.to_csv(os.path.join(opt.save_path, "train.csv"), index=False)
    valset.to_csv(os.path.join(opt.save_path, "val.csv"), index=False)
    testset.to_csv(os.path.join(opt.save_path, "test.csv"), index=False)

def add_parse_arguments(parser):

    parser.add_argument('--dataset', type=str, default="data/detectors/FMereani.csv", help='Dataset')
    parser.add_argument('--save_path', type=str, default="data/detectors", help='Destination')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:5555/vuln_backend/1.0/endpoint/', help='Endpoint of the backend used by the oracle')
    parser.add_argument('--trainval_percentage', type=float, default=0.8, help='Percentage of the dataset to use for training and validation')
    parser.add_argument('--train_percentage', type=float, default=0.8, help='Percentage of the trainval dataset to use for training')

    #seed
    parser.add_argument('--seed', type=int, default=156, help='seed for reproducibility')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    prepare_detectors_dataset(opt)

if __name__ == '__main__':
    main()