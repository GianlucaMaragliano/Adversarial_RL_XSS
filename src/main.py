import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from itertools import chain
from gensim.models import Word2Vec

from src.utils.vectorizer import xss_payloads_vectorizer, get_sorted_tokens
from src.utils.preprocessor import preprocess_payload
from src.utils.tokenizer import xss_tokenizer, uncommon_token_replacer, clean_tokenized_payloads

cleaned_data = pd.read_csv('../data/Payloads_checked.csv')
cleaned_data.drop_duplicates(subset=['Payloads'], inplace=True)

payloads = cleaned_data['Payloads']
preprocessed_payloads = preprocess_payload(payloads)

labels = cleaned_data['Class']
# Convert labels to numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

tokenized_payloads = [xss_tokenizer(payload) for payload in preprocessed_payloads]

X_tfidf = xss_payloads_vectorizer(preprocessed_payloads, tokenized_payloads)

sorted_tokens = get_sorted_tokens(X_tfidf)

# Display the most common tokens
print(sorted_tokens[:10])

cleaned_tokenized_payloads = clean_tokenized_payloads(tokenized_payloads, sorted_tokens)

# Word2Vec representation for each attack payload
vector_size = 32
word2vec = Word2Vec(cleaned_tokenized_payloads, vector_size=vector_size, window=5, min_count=1, workers=4)


# Get word vectors
def get_word_vectors(tokens):
    word_vectors = [word2vec.wv[token] for token in tokens if token in word2vec.wv]
    return word_vectors


payload_word_vectors = [get_word_vectors(tokens) for tokens in cleaned_tokenized_payloads]