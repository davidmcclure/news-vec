

import re

from boltons.iterutils import pairwise


SEP_TOKENS = {':', '-', '–', '—', '|', 'via', '[', ']'}


# TODO: Reuters hack for "brief-XXX". Learn with char LSTM?
def split_first_token(tokens):
    """If the first token has a hyphen, break into separate tokens.
    """
    first = re.split('(-)', tokens[0])
    return [*first, *tokens[1:]]


def scrub_paratext(tokens):
    """Try to prune out "paratext" around headlines. Hacky.
    """
    sep_idxs = [
        i for i, t in enumerate(tokens)
        if t.lower() in SEP_TOKENS
    ]

    if not sep_idxs:
        return tokens

    if sep_idxs[0] != 0:
        sep_idxs = [-1] + sep_idxs

    if sep_idxs[-1] != len(tokens)-1:
        sep_idxs = sep_idxs + [len(tokens)]

    widths = [
        (i1, i2, i2-i1)
        for i1, i2 in pairwise(sep_idxs)
    ]

    widths = sorted(
        widths,
        key=lambda x: x[2],
        reverse=True,
    )

    i1 = widths[0][0]+1
    i2 = widths[0][1]

    return tokens[i1:i2]


CURLY_STRAIGHT = (('“', '"'), ('”', '"'), ('‘', "'"), ('’', "'"))


def uncurl_quotes(text):
    """Curly -> straight.
    """
    for c, s in CURLY_STRAIGHT:
        text = text.replace(c, s)

    return text


QUOTES = {'\'', '"'}


def scrub_quotes(tokens):
    """Remove quote tokens.
    """
    return [t for t in tokens if uncurl_quotes(t) not in QUOTES]


def drop_empty_strings(tokens):
    """Remove empty tokens.
    """
    return [t for t in tokens if len(t)]


def clean_headline(tokens):
    """Raw tokens -> clf tokens.
    """
    tokens = split_first_token(tokens)
    tokens = scrub_paratext(tokens)
    tokens = scrub_quotes(tokens)
    tokens = drop_empty_strings(tokens)
    return tokens
