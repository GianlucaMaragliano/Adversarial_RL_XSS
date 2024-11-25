import argparse


def init_argument_parser(add_arguments_fn):
    parser = argparse.ArgumentParser()
    parser = add_arguments_fn(parser)
    opt = parser.parse_args()
    return opt