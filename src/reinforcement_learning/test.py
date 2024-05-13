import pandas as pd

from src.utils.preprocessor import preprocess_payload
from src.utils.tokenizer import xss_tokenizer

if __name__ == '__main__':
    mutated_xss_example = "http://www.productiontrax.com/subcategory.php?id=&quot;&lt;%00script&gt;alert`document.cookie`&lt;%00/script&gt;"
    xss_df = pd.DataFrame([mutated_xss_example], columns=['Payloads'])
    preprocessed_payloads = preprocess_payload(xss_df['Payloads'])
    tokenized_payloads = [xss_tokenizer(payload) for payload in preprocessed_payloads]
    print(tokenized_payloads)