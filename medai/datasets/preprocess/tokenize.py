import re

NUMBER_TOKEN = 'NUMBER'
NUMBER_TOKEN_PAD = ' NUMBER '

_VALID_WORD_TOKEN = re.compile(r'\A\w+')

def remove_punctuation_after_dots(tokens):
    clean_tokens = []
    last_was_dot = False
    for token in tokens:
        is_valid_word = _VALID_WORD_TOKEN.search(token)
        if last_was_dot and not is_valid_word:
            continue

        clean_tokens.append(token)
        last_was_dot = (token == '.')

    return clean_tokens


def remove_initial_punctuation(tokens):
    if not tokens:
        return []

    first_word_token = next(
        filter(lambda i: _VALID_WORD_TOKEN.search(tokens[i]), range(len(tokens))),
        0,
    )
    return tokens[first_word_token:]


_TYPOS = {
    'telehpone': 'telephone',
    'addedd': 'added',
    'antibioic': 'antibiotic',
    'conven8tional': 'conven8tional',
    'possiblle': 'possible',
    'possiby': 'possibly',
    'lkiely': 'likely',
    'clinnically': 'clinically',
    'cardkiopulmonary': 'cardiopulmonary',
    'differntial': 'differential',
    'dgenerative': 'degenerative',
    'possibilitiy': 'possibility',
    'mildf': 'mild',
    'litttle': 'little',
    'proces': 'process',
    'pacification': 'opacification',
    'cut=rrent': 'current',
    'neertheless': 'nevertheless',
    'morel': 'more',
    'lessl': 'less',
    'represet': 'represent',
    'iscussed': 'discussed',
    'wered': 'were',
    'asdiscussed': ('as', 'discussed'),
    'howeve': 'however',
    'consolidatio': 'consolidation',
    'betweeen': 'between',
}

_ENDS_WITH_PUNCTUATION = re.compile(r'([A-Za-z]+)[\-_\.]+\Z')
_STARTS_WITH_PUNCTUATION = re.compile(r'\A[\-_\.]+([A-Za-z]+)')
def clean_token(token):
    match = _ENDS_WITH_PUNCTUATION.search(token)
    if match:
        token = match.group(1)

    match = _STARTS_WITH_PUNCTUATION.search(token)
    if match:
        token = match.group(1)

    if '-' in token:
        return token.split('-')

    token = _TYPOS.get(token, token)

    return token


def _text_to_tokens(text):
    text = text.lower()
    # Remove html tags
    text = re.sub(r'(\[)?&amp;[gl]t;(\])?', ' ', text)

    # Replace common words-phrases
    text = re.sub(r'\+\/\-', ' more or less ', text)
    text = re.sub(r'x[\-\s]+ray', ' xray ', text)
    text = re.sub(r'e[\-\s]+mail', ' email ', text)
    text = re.sub(r'o[\'\-]+clock', ' ', text)
    text = re.sub(r'd[\/\.]w', ' discussed with ', text)
    text = re.sub(r's[\/\.]p', ' status post ', text)
    text = re.sub(r'f[\/\.]u', ' follow up ', text)
    text = re.sub(r'c[\/\.]w', ' consistent with ', text)
    text = re.sub(r'b[\/\.]l', ' bilateral ', text)
    text = re.sub(r'e\.?g\.?', ' example ', text)

    # Replace dr. with doctor
    text = re.sub(r'dr[\.\s]', ' doctor ', text)
    text = re.sub(r'm\.?d\.?', ' doctor ', text)

    # Separate PM or AM token
    text = re.sub(r'(a|p)\.?m(\.|\s|\Z)', r' \1m ', text)

    # Remove common signature
    text = re.sub(r'[_\-\=]+\w[_\-\=]+\Z', r' ', text)

    # Replace two dots
    text = re.sub(r':[\s\.]', ' ', text)

    # Replace multiple comma/semicolon with simple coma
    text = re.sub(r'(;|,)+', r',', text)

    # Replace low-dash with % or # (are ofuscated numbers)
    text = re.sub(r'_+%', NUMBER_TOKEN_PAD, text)
    text = re.sub(r'#_+', NUMBER_TOKEN_PAD, text)

    # Replace multiple lodash as ofuscator token
    text = re.sub(r'[-_]*__+\s*[-_]*', ' xxxx ', text)

    # Replace break line tag
    text = re.sub(r'< ?br ?\\?>', ' ', text)
    text = re.sub(r'[\[\]<>]', '', text) # Remove brackets [] <>

    text = re.sub(r'\.[.,]*', r'.', text) # Replace multiple dots with one dot

    # Number as enumerators, like "1. bla bla, 2. bla bla"
    text = re.sub(r'(\W|\A)\d+\s*(\.|\))[^\d]', r' . ', text)
    # text = re.sub(r'(\d)\.', r'\1 .', text)

    # Give space to symbols
    text = re.sub(r'([\(\)¿\?\!])', r' \1 ', text)

    # Words that have one digit are typos
    text = re.sub(r'(\w\w+)\d(\w*)', r' \1\2 ', text)

    # Other numbers
    text = re.sub(
        r'(\W|\A)(\#+\s*)?\d[\d\.\-\/:h]*\s*(a|st|nd|th|rd|erd|\%|cm|mm|xxxx|pm|am|s)?',
        r' NUMBER ',
        text,
    )

    # Add space between text and dot/comma
    text = re.sub(r'([a-zA-Z0-9])(\.|,|/)', r'\1 \2', text)
    text = re.sub(r'(\.|,|/)([a-zA-Z0-9])', r'\1 \2', text)

    # Remove apostrophe
    text = re.sub(r'(\w*)\'[st]?', r'\1 ', text) # XXXX't is a typo

    # Remove quotes or other symbols
    text = re.sub(r'[\`\'\"\*:\\~]', ' ', text)

    # Replace ampersand
    text = re.sub(r'\s\&\s', ' and ', text)

    # Separate numbers from end of sentence
    text = re.sub(r'NUMBER([\.-])', r' NUMBER \1', text)

    # Replace -- with comma
    text = re.sub('--', ' , ', text)

    # Rest of starting - are garbage
    text = re.sub(r'[\W]-(\w+)', r' \1', text)
    text = re.sub(r'\A-(\w+)', r' \1', text)

    # Dots next to other punctuation
    text = re.sub(r'[,=]\.', ' . ', text)
    text = re.sub(r'\.\/', ' / ', text)

    tokens = []
    for token in text.split():
        token = clean_token(token)
        if isinstance(token, (tuple, list)):
            tokens.extend(token)
        elif isinstance(token, str):
            tokens.append(token)

    tokens = remove_punctuation_after_dots(tokens)
    tokens = remove_initial_punctuation(tokens)

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
