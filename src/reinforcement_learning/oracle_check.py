import pandas as pd
import ast

from src.filters.utils.html_tools import is_same_dom
from src.filters.utils.request_tools import do_xss_post_request


def main():
    endpoint = "http://127.0.0.1:5555/vuln_backend/1.0/endpoint/"
    basic_payload = "abc"
    basic_html = do_xss_post_request(endpoint, basic_payload)

    results_df = pd.read_csv("../../data/mutated_attacks.csv")
    results_df =results_df[results_df['Not Detected'] == True]

    counter = 0

    for i, row in results_df.iterrows():
        initial_payload = row['Initial Payload']
        # print("Initial Payload:", initial_payload)

        initial_payload_tokenized = row['Initial Payload Tokenized']
        initial_payload_tokenized = initial_payload_tokenized[1:-1]
        initial_payload_tokenized = ast.literal_eval(initial_payload_tokenized)
        initial_payload_tokenized = [x for x in initial_payload_tokenized if x != 'None']
        initial_payload_tokenized = ''.join(initial_payload_tokenized)
        # print("Initial Tokenized Payload:", initial_payload_tokenized)

        if is_same_dom(do_xss_post_request(endpoint, initial_payload_tokenized), basic_html):
            counter += 1
            continue

        mutated_payload = row['Mutated Payload']
        # print("Mutated Payload:", mutated_payload)
        # if is_same_dom(do_xss_post_request(endpoint, mutated_payload), basic_html):
            # counter += 1
            # continue

        mutated_payload_tokenized = row['Mutated Payload Tokenized']
        mutated_payload_tokenized = mutated_payload_tokenized[1:-1]
        mutated_payload_tokenized = ast.literal_eval(mutated_payload_tokenized)
        mutated_payload_tokenized = [x for x in mutated_payload_tokenized if x != 'None']
        mutated_payload_tokenized = ''.join(mutated_payload_tokenized)
        # print("Mutated Tokenized Payload:", mutated_payload_tokenized)
        if is_same_dom(do_xss_post_request(endpoint, mutated_payload_tokenized), basic_html):
            counter += 1
            continue
    print(f"{counter}/{len(results_df)} payloads are indeed benign")


if __name__ == "__main__":
    main()