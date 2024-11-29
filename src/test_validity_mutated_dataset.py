

from utils.utils import init_argument_parser
from utils.dataset_utils import check_column_validity_with_oracle
from utils.preprocess import process_payloads
import pandas as pd
import os 
import numpy as np

def test_validity_mutated_dataset(opt):
    test_set = pd.read_csv(opt.dataset, on_bad_lines='skip')
    vocab = pd.read_csv(opt.vocab)['tokens'].tolist()
    test_set["tokenized_original"] = process_payloads(test_set, column='Original', sorted_tokens=vocab)[1]
    test_set['reconstructed_str_payload_original'] = test_set['tokenized_original'].map(lambda x: ' '.join(x))
    test_set['is_tokenized_original_ok'] = (check_column_validity_with_oracle(test_set, 'reconstructed_str_payload_original', opt.endpoint)).map(lambda x: not x)

    test_set['is_mutated_ok'] = (check_column_validity_with_oracle(test_set, 'Payloads', opt.endpoint)).map(lambda x: not x)

    test_set['tokenized'] = process_payloads(test_set, sorted_tokens=vocab)[1]
    test_set['reconstructed_str_payload'] = test_set['tokenized'].map(lambda x: ' '.join(x))
    test_set['is_tokenized_ok'] = (check_column_validity_with_oracle(test_set, 'reconstructed_str_payload', opt.endpoint)).map(lambda x: not x)


    test_set.to_csv(f"{'/'.join(opt.dataset.split('/')[:-1])}/validity.csv", index=False)



def add_parse_arguments(parser):

    parser.add_argument('--dataset', type=str, required = True, help='Dataset with mutated payloads')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:5555/vuln_backend/1.0/endpoint/', help='Endpoint of the backend used by the oracle')
    parser.add_argument('--vocab', type=str, required=True, help='path of the config of the detector')

    #seed
    parser.add_argument('--seed', type=int, default=156, help='seed for reproducibility')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    test_validity_mutated_dataset(opt)

if __name__ == '__main__':
    main()