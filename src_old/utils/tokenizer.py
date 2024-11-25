from nltk.tokenize import RegexpTokenizer


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
