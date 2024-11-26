
import pandas as pd
from utils.request_tools import do_xss_post_request
from utils.html_tools import is_same_dom


def filter_dataset_with_oracle(df, endpoint):
    basic_payload = "abc"
    basic_html = do_xss_post_request(endpoint, basic_payload)
    df['is_same'] = df['Payloads'].apply(lambda x: is_same_dom(do_xss_post_request(endpoint, x), basic_html))
    # keep only the Payloads where Class is Malicious and is_same is False and Payloads where Class is Benign and
    # is_same is True
    df = df[((df['Class'] == 'Malicious') & (df['is_same'] == False)) | (
            (df['Class'] == 'Benign') & (df['is_same'] == True))]
    df = df.drop(columns=['is_same'])
    return df

def filter_detector_dataset(df):
    # filter df keeping only the rows where the payloads start with any of the elements in starting_strs
    df_filtered = df[df.Payloads.str.startswith("http")]

    # drop duplicates in Payloads attribute
    df_filtered = df_filtered.drop_duplicates(subset=['Payloads'])
    return df_filtered

def dataset_split(df, train_percentage):

    classes = list(df['Class'].unique())
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    for c in classes:
        class_df = df[df['Class'] == c]
        subset_df = class_df.sample(frac=train_percentage)
        train_df = pd.concat([train_df, subset_df])
        test_df = pd.concat([test_df, (class_df.drop(subset_df.index))])

    return train_df, test_df

def from_text_to_csv(lines):
    df = pd.DataFrame(lines, columns=['Payloads'])
    df['Class'] = 'Malicious'
    return df