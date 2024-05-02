from urllib.parse import unquote_plus, urlsplit, urlunsplit
import html
import re
import validators


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
