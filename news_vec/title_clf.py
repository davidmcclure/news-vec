

import ujson
import gzip

from boltons.iterutils import pairwise
from glob import glob
from tqdm import tqdm
from collections import Counter
from itertools import islice

from . import logger


SEP_TOKENS = {':', '-', '–', '—', '|', 'via', '[', ']'}


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


def clean_headline(tokens):
    """Raw tokens -> clf tokens.
    """
    tokens = scrub_paratext(tokens)
    tokens = scrub_quotes(tokens)
    return tokens


class Line:

    def __init__(self, tokens, label, lower=True):
        self.tokens = [t.lower() for t in tokens] if lower else tokens
        self.label = label

    def __repr__(self):

        pattern = '{cls_name}<{token_count} tokens -> {label}>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            token_count=len(self.tokens),
            label=self.label,
        )


def read_json_lines(root, lower=True):
    """Read JSON corpus.

    Yields: Line
    """
    for path in glob('%s/*.gz' % root):
        with gzip.open(path) as fh:
            for line in fh:

                data = ujson.loads(line)

                tokens = data.get('tokens')
                tokens = clean_headline(tokens)

                if not tokens:
                    continue

                yield Line(tokens, data['label'], lower=lower)


class Corpus:

    def __init__(self, root, skim=None, lower=True):
        """Read lines.
        """
        logger.info('Parsing line corpus.')

        lines_iter = islice(read_json_lines(root, lower), skim)

        self.lines = list(tqdm(lines_iter))

    def __repr__(self):

        pattern = '{cls_name}<{line_count} lines>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            line_count=len(self),
        )

    def __len__(self):
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)

    def token_counts(self):
        """Collect all token -> count.
        """
        logger.info('Gathering token counts.')

        counts = Counter()
        for line in tqdm(self):
            counts.update(line.tokens)

        return counts

    def label_counts(self):
        """Label -> count.
        """
        logger.info('Gathering label counts.')

        counts = Counter()
        for line in tqdm(self):
            counts[line.label] += 1

        return counts

    def labels(self):
        counts = self.label_counts()
        return [label for label, _ in counts.most_common()]
