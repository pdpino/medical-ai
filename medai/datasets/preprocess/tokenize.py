import re

NUMBER_TOKEN = 'NUMBER'


def remove_consecutive_dots(tokens):
    clean_tokens = []
    last_was_dot = False
    for token in tokens:
        is_dot = (token == '.')
        if last_was_dot and is_dot:
            continue

        clean_tokens.append(token)
        last_was_dot = is_dot

    return clean_tokens


def _text_to_tokens(text):
    text = text.lower()
    # Remove html tags
    text = re.sub(r'(\[)?&amp;[gl]t;(\])?', ' ', text)

    # Replace dr. with doctor
    text = re.sub(r'dr\.', 'doctor', text)

    # Separate PM or AM token
    text = re.sub(r'(a|p)\.?m\.?', r' \1m ', text)

    # Replace two dots
    text = re.sub(r': ', ' ', text)
    text = re.sub(r'(\d):(\d)', r'\1\2', text)

    # Replace multiple comma/semicolon with simple coma
    text = re.sub(r'(;|,+)', r',', text)

    # Replace multiple lodash as ofuscator token
    text = re.sub(r'__+', 'xxxx', text)

    # Replace numbers with decimals by token
    text = re.sub(r'\d+(\.|/)\d+', NUMBER_TOKEN, text)

    # Replace break line tag
    text = re.sub(r'< ?br ?\\?>', ' ', text)
    text = re.sub(r'[\[\]<>]', '', text) # Remove brackets [] <>
    text = re.sub(r'(\(|\))', r' \1 ', text) # Give space to parenthesis

    text = re.sub(r'\.[.,]*', r'.', text) # Replace multiple dots with one dot

    # Number as enumerators, like "1. bla bla, 2. bla bla"
    text = re.sub(r'(\W|\A)\d+\.[^\d]', r' . ', text)
    # text = re.sub(r'(\d)\.', r'\1 .', text)

    # Add space between text and dot/comma
    text = re.sub(r'([a-zA-Z0-9])(\.|,|/)', r'\1 \2', text)
    text = re.sub(r'(\.|,|/)([a-zA-Z0-9])', r'\1 \2', text)

    # Other numbers
    text = re.sub(r'(\W|\A)\d+(a|st|nd|th|rd|\%|mm|xxxx)?', r'\1 {}'.format(NUMBER_TOKEN), text)
    # text = re.sub(r'\A\d+(a|st|nd|th|rd|\%|mm|xxxx)?', r'\1 {}'.format(NUMBER_TOKEN), text)

    # Remove apostrophe
    text = re.sub(r'(\w+)\'[st]?', r'\1 ', text) # XXXX't is a typo

    text = re.sub(r'NUMBER([\.-])', r'NUMBER \1', text)

    tokens = remove_consecutive_dots(text.split())
    # Remove starting dot
    if tokens[0] == '.':
        tokens = tokens[1:]

    # Assure ending dot
    if len(tokens) >= 1 and tokens[-1] != '.':
        tokens.append('.')

    return tokens


def is_text_null(text):
    return not text or not isinstance(text, str)


def text_to_tokens(text, ignore_tokens=[]):
    """Converts a string to a list of tokens."""
    if is_text_null(text):
        return []

    ignore_tokens = set(ignore_tokens)

    return [
        token
        for token in _text_to_tokens(text)
        if token not in ignore_tokens
    ]
