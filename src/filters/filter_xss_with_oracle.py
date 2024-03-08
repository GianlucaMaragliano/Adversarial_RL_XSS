import pandas as pd
import argparse

from src.utils.request_tools import do_xss_post_request
from src.utils.html_tools import is_same_dom


def filter_dataset_with_oracle(opt):
    endpoint = opt.endpoint
    df = pd.read_csv(opt.data)
    basic_payload = "abc"
    basic_html = do_xss_post_request(endpoint, basic_payload)
    df['is_same'] = df['Payloads'].apply(lambda x: is_same_dom(do_xss_post_request(endpoint, x), basic_html))
    # keep only the Payloads where Class is Malicious and is_same is False and Payloads where Class is Benign and
    # is_same is True
    df = df[((df['Class'] == 'Malicious') & (df['is_same'] == False)) | (
            (df['Class'] == 'Benign') & (df['is_same'] == True))]
    df = df.drop(columns=['is_same'])
    df.to_csv(opt.dest, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/filtered_dataset.csv', help='source')
    parser.add_argument('--dest', type=str, default='data/filtered_by_oracle_dataset.csv', help='destination')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:5555/vuln_backend/1.0/endpoint/',
                        help='endpoint of the template server')

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_arguments()
    filter_dataset_with_oracle(opt)


if __name__ == '__main__':
    main()
