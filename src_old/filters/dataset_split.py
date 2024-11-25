import pandas as pd
from ruamel.yaml import YAML
import argparse
# fix seed for pandas sampling
import numpy as np

np.random.seed(42)


def dataset_split(opt, params):
    df = pd.read_csv(opt.data, encoding="ISO-8859-1")
    train_percentage = params[opt.split_arguments]["train_size"]

    classes = list(df['Class'].unique())
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    for c in classes:
        class_df = df[df['Class'] == c]
        subset_df = class_df.sample(frac=train_percentage)
        train_df = pd.concat([train_df, subset_df])
        test_df = pd.concat([test_df, (class_df.drop(subset_df.index))])

    train_df.to_csv(opt.dest_train, index=False)
    test_df.to_csv(opt.dest_test, index=False)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dataset.csv', help='source')
    parser.add_argument('--dest_train', type=str, default='data/train.csv', help='train set')
    parser.add_argument('--dest_test', type=str, default='data/test.csv', help='test set')
    parser.add_argument('--split_arguments', type=str, default='dataset_trainval_test_split',
                        help='parameters containing split numbers')

    parser.add_argument('--params', type=str, default='params.yaml', help='params')  # file/folder, 0 for webcam

    opt = parser.parse_args()
    return opt


def main():
    opt = parse_arguments()
    with open(opt.params) as f:
        yaml = YAML(typ="safe")
        params = yaml.load(f)
    dataset_split(opt, params)


if __name__ == '__main__':
    main()
