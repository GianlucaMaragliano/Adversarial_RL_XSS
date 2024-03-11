import requests
from urllib.parse import unquote


def do_xss_post_request(endpoint, payload):
    html_page = requests.post(url=endpoint, json={"payload": [payload]})
    return unquote(html_page.content)


def write_xss_post_request_output(endpoint, payload, output):
    html_page = do_xss_post_request(endpoint, payload)
    with open(output, 'w') as f:
        f.write(str(html_page))
