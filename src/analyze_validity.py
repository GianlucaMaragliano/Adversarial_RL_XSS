from utils.utils import init_argument_parser
import pandas as pd
import os
import json

def none_rate(token_list):
    #token list is a string, but it is a python list
    token_list = token_list.replace("[","").replace("]","").replace("'","").replace(",","").split()
    return token_list.count("None")/len(token_list)


def test_validity_mutated_dataset(opt):
    test_set = pd.read_csv(opt.dataset, on_bad_lines='skip')
    file_to_save = os.path.join('/'.join(opt.dataset.split('/')[:-1]),"ruin_rate.json")
    count_original_ok = len(test_set[test_set['is_tokenized_original_ok']])
    count_mutated_ok = len(test_set[test_set['is_mutated_ok']])
    count_tokenized_ok = len(test_set[test_set['is_tokenized_ok']])

    test_set["original_none_rate"] = test_set["tokenized_original"].map(none_rate)
    test_set["mutated_none_rate"] = test_set["tokenized"].map(none_rate)


    print(f"Original: {count_original_ok}")
    print(f"Mutated: {count_mutated_ok}")
    print(f"Tokenized: {count_tokenized_ok}")
    print(f"Total test set: {len(test_set)}")
    print(f"Original None rate: {test_set['original_none_rate'].mean()}")
    print(f"Mutated None rate: {test_set['mutated_none_rate'].mean()}")

    to_save = {
        "rr_original":1- count_original_ok/len(test_set),
        "rr_rq1": 1- count_mutated_ok/len(test_set),
        "rr_rq2": 1- count_tokenized_ok/len(test_set),
        "original_none_rate": test_set['original_none_rate'].mean(),
        "mutated_none_rate": test_set['mutated_none_rate'].mean()
    }
    with open(file_to_save, 'w') as f:
        json.dump(to_save, f,ensure_ascii=False,indent=4)



def add_parse_arguments(parser):

    parser.add_argument('--dataset', type=str, required = True, help='Dataset with mutated payloads')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:5555/vuln_backend/1.0/endpoint/', help='Endpoint of the backend used by the oracle')

    #seed
    parser.add_argument('--seed', type=int, default=156, help='seed for reproducibility')

    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    test_validity_mutated_dataset(opt)

if __name__ == '__main__':
    main()