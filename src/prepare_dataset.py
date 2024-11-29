from utils.utils import init_argument_parser
from utils.dataset_utils import filter_dataset_with_oracle, filter_detector_dataset, dataset_split, balance_classes
import pandas as pd
import os 
import numpy as np
from utils.preprocess import process_payloads

def prepare_dataset(opt):
    #fix the seed for reproducibility
    np.random.seed(opt.seed)

    df = pd.read_csv(opt.dataset, encoding="ISO-8859-1")
    df_filtered = filter_detector_dataset(df)
    df_filtered.to_csv(os.path.join(opt.save_path, "filtered.csv"), index=False)

    sorted_tokens, df_filtered["tokenized"] = process_payloads(df_filtered, column='Payloads',percentage=opt.vocab_size)
    df_filtered['reconstructed_str'] = df_filtered['tokenized'].map(lambda x: ' '.join(x))

    df_filtered_with_oracle = filter_dataset_with_oracle(df_filtered, opt.endpoint)
    df_filtered_with_oracle = filter_dataset_with_oracle(df_filtered_with_oracle, opt.endpoint, column='reconstructed_str')

    df_filtered_with_oracle = balance_classes(df_filtered_with_oracle)
    #drop the tokenized and reconstructed_str columns
    df_filtered_with_oracle = df_filtered_with_oracle.drop(columns=['tokenized', 'reconstructed_str'])


    df_filtered_with_oracle.to_csv(os.path.join(opt.save_path, "filtered_oracle.csv"), index=False)
    trainval_set, testset = dataset_split(df_filtered_with_oracle, opt.trainval_percentage)
    trainset, valset = dataset_split(trainval_set, opt.train_percentage)

    train_det, train_rl = dataset_split(trainset, 0.5)
    val_det, val_rl = dataset_split(valset, 0.5)
    test_det, test_rl = dataset_split(testset, 0.5)

    voc_folder = os.path.join(opt.save_path,str(int(opt.vocab_size*100)))
    if not os.path.exists(voc_folder):
        os.makedirs(voc_folder)
    pd.DataFrame({"tokens": sorted_tokens}).to_csv(os.path.join(voc_folder,'vocabulary.csv'))


    det_folder = os.path.join(opt.save_path,str(int(opt.vocab_size*100)), "detectors")
    if not os.path.exists(det_folder):
        os.makedirs(det_folder)
    train_det.to_csv(os.path.join(det_folder, "train.csv"), index=False)
    val_det.to_csv(os.path.join(det_folder, "val.csv"), index=False)
    test_det.to_csv(os.path.join(det_folder, "test.csv"), index=False)

    rl_folder = os.path.join(opt.save_path,str(int(opt.vocab_size*100)),"adversarial_agents")
    if not os.path.exists(rl_folder):
        os.makedirs(rl_folder)

    #discard all the benign payloads
    train_rl = train_rl[train_rl['Class'] == 'Malicious']
    val_rl = val_rl[val_rl['Class'] == 'Malicious']
    test_rl = test_rl[test_rl['Class'] == 'Malicious']
    train_rl.to_csv(os.path.join(rl_folder, "train.csv"), index=False)
    val_rl.to_csv(os.path.join(rl_folder, "val.csv"), index=False)
    test_rl.to_csv(os.path.join(rl_folder, "test.csv"), index=False)



def add_parse_arguments(parser):

    parser.add_argument('--dataset', type=str, default="data/FMereani.csv", help='Dataset')
    parser.add_argument('--save_path', type=str, default="data/", help='Destination')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:5555/vuln_backend/1.0/endpoint/', help='Endpoint of the backend used by the oracle')
    parser.add_argument('--trainval_percentage', type=float, default=0.8, help='Percentage of the dataset to use for training and validation')
    parser.add_argument('--train_percentage', type=float, default=0.8, help='Percentage of the trainval dataset to use for training')
    parser.add_argument('--vocab_size', type=float, default=0.1, help='Percentage of the most common tokens to keep in the vocab')

    #seed
    parser.add_argument('--seed', type=int, default=156, help='seed for reproducibility')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    prepare_dataset(opt)

if __name__ == '__main__':
    main()