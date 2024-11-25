import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from itertools import chain

from src.utils.vectorizer import xss_payloads_vectorizer, get_sorted_tokens
from src.utils.preprocessor import preprocess_payload
from src.utils.tokenizer import xss_tokenizer, uncommon_token_replacer, clean_tokenized_payloads

cleaned_data = pd.read_csv('../data/Payloads_checked.csv')
cleaned_data.drop_duplicates(subset=['Payloads'], inplace=True)

payloads = cleaned_data['Payloads']
preprocessed_payloads = preprocess_payload(payloads)

cleaned_data['Payloads'] = preprocessed_payloads
cleaned_data.to_csv('../data/Preprocessed.csv', index=False)

labels = cleaned_data['Class']
# Convert labels to numerical format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

tokenized_payloads = [xss_tokenizer(payload) for payload in preprocessed_payloads]

cleaned_data['Payloads'] = tokenized_payloads
cleaned_data.to_csv('../data/Tokenized.csv', index=False)

X_tfidf = xss_payloads_vectorizer(preprocessed_payloads, tokenized_payloads)

sorted_tokens = get_sorted_tokens(X_tfidf)

# Display the most common tokens
print(sorted_tokens[:10])

cleaned_tokenized_payloads = clean_tokenized_payloads(tokenized_payloads, sorted_tokens)
