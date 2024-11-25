import numpy as np
import pandas as pd
from urllib.parse import unquote_plus, urlsplit, urlunsplit
import html
import re
import validators
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
global vectorizer


def xss_payloads_vectorizer(processed_payloads, tokenized_payloads, percentage=0.1):
    # Percentage of dictionary to keep, originally 5%, in paper 10%
    percent = percentage
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

    # Sort tokens by TF-IDF score into dataframe
    sorted_tokens_df = df_tfidf.sum().sort_values(ascending=False)

    # # if not present, save sorted tokens df into the vocabolary csv file
    # sorted_tokens_df.to_csv('../../reproduction/data/vocabolary.csv')

    # return sorted tokens
    sorted_tokens = sorted_tokens_df.index.tolist()

    return sorted_tokens


def xss_tokenizer(payload):
    # Tokenization rules

    # Rules:
    # 1. Function : [\w\.]+?\(
    # 2. Contents contained in double quotes: ”\w+?”
    # 3. Contents contained in single quotes: \'\w+?\'
    # 4. URLs: http://\w+
    # 5. Closing HTML tags: </\w+>
    # 6. Opening HTML tags: <\w+>
    # 7. Window activities: \b\w+=
    # 8. Contents contained in parentheses: (?<=\()\S+(?=\))
    # 9. Non-closing HTML tags: <(?<=\<)\S+
    # 10. Closing parentheses and HTML tags: \) | \>

    rules = (r'''(?x)[\w\.]+?\(
             | ”\w+?”
             | \'\w+?\'
             | http://\w+
             | </\w+>
             | <.+?>
             | \b\w+=
             | \w+:
             | (?<=\()\S+(?=\))
             | <(?<=\<)\S+
             | \) | \>
             ''')

    tokenizer = RegexpTokenizer(rules)
    tokens = tokenizer.tokenize(payload)

    return tokens


def uncommon_token_replacer(tokens, common_tokens):
    # Replace uncommon tokens with 'None'
    return ['None' if token not in common_tokens else token for token in tokens]


def clean_tokenized_payloads(tokenized_payloads, sorted_tokens):
    cleaned_tokenized_payloads = []
    for i in range(len(tokenized_payloads)):
        cleaned_tokenized_payload = uncommon_token_replacer(tokenized_payloads[i], sorted_tokens)
        cleaned_tokenized_payloads.append(cleaned_tokenized_payload)
    return cleaned_tokenized_payloads


def preprocess_payload(payload):
    preprocessed_payloads = []
    for p in payload:
        processed_payload = p.lower()
        # Simplify urls to http://u
        sep = "="
        test = processed_payload.split(sep, 1)[0]
        if test != processed_payload:
            processed_payload = processed_payload.replace(test, "http://u")
        else:
            if validators.url(processed_payload):
                url = list(urlsplit(processed_payload))
                url[0] = "http"
                url[1] = "u"
                processed_payload = urlunsplit(url)
        # Decode HTML entities
        processed_payload = str(html.unescape(processed_payload))
        # Remove special HTML tags
        processed_payload = processed_payload.replace("<br>", "")
        # Decoding the payload
        processed_payload = unquote_plus(processed_payload)
        # Remove special characters
        processed_payload = re.sub(r'\\+', '', processed_payload)  # NOT WORKING
        # Replace numbers with 0, if not after %
        processed_payload = re.sub(r'(?<!%)\d', '0', processed_payload)
        processed_payload = re.sub(r'0+', '0', processed_payload)
        preprocessed_payloads.append(processed_payload)
    return preprocessed_payloads


def process_payloads(payloads, sorted_tokens=None, percentage = 0.1):
    # Preprocess payloads
    preprocessed_payloads = preprocess_payload(payloads['Payloads'])

    # Tokenize payloads
    tokenized_payloads = [xss_tokenizer(payload) for payload in preprocessed_payloads]

    # Vectorize payloads
    if sorted_tokens is None:
        x_tfidf = xss_payloads_vectorizer(preprocessed_payloads, tokenized_payloads, percentage)
        sorted_tokens = get_sorted_tokens(x_tfidf)
        sorted_tokens.append('None')

    cleaned_tokenized_payloads = clean_tokenized_payloads(tokenized_payloads, sorted_tokens)
    return sorted_tokens, cleaned_tokenized_payloads