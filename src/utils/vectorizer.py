import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain

from src.utils.tokenizer import xss_tokenizer

global vectorizer


def xss_payloads_vectorizer(processed_payloads, tokenized_payloads):
    percent = 0.05
    # Get number of different tokens
    token_size = (len(set(chain.from_iterable(tokenized_payloads))))
    # Set max_features to % of the number of different tokens
    max_features = token_size if token_size < 100 else int(token_size * percent)
    global vectorizer
    vectorizer = TfidfVectorizer(tokenizer=xss_tokenizer, max_features=max_features, token_pattern=None)
    x_tfidf = vectorizer.fit_transform(processed_payloads)

    return x_tfidf


def get_sorted_tokens(x_tfidf):
    global vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Sum TF-IDF Scores
    total_tfidf_scores = x_tfidf.sum(axis=0)

    # Create a DataFrame and Sort Tokens by TF-IDF Score
    df_tfidf = pd.DataFrame(total_tfidf_scores, columns=feature_names)
    sorted_tokens = df_tfidf.sum().sort_values(ascending=False).index.tolist()

    return sorted_tokens
