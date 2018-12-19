

import ujson
import gzip
import random

from glob import glob
from itertools import islice, chain
from tqdm import tqdm
from collections import Counter, defaultdict
from torch.utils.data import Dataset

from . import logger


def read_json_lines(root):
    """Read JSON corpus.

    Yields: Line
    """
    for path in glob('%s/*.gz' % root):
        with gzip.open(path) as fh:
            for line in fh:
                data = ujson.loads(line)
                yield Line(data['tokens'], data['label'], data['first_ts'])


class Line:

    def __init__(self, tokens, label, timestamp):
        self.tokens = tokens
        self.label = label
        self.timestamp = timestamp

    def __repr__(self):

        pattern = '{cls_name}<{token_count} tokens -> {label}>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            token_count=len(self.tokens),
            label=self.label,
        )


class Corpus:

    def __init__(self, root, skim=None):
        """Read lines.
        """
        logger.info('Parsing line corpus.')

        lines_iter = islice(read_json_lines(root), skim)

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

    def label_counts(self):
        """Label -> count.
        """
        logger.info('Gathering label counts.')

        counts = Counter()
        for line in tqdm(self):
            counts[line.label] += 1

        return counts

    def min_label_count(self):
        """Count of the most infrequent label.
        """
        counts = self.label_counts()
        return counts.most_common()[-1][1]

    def sample_all_vs_all(self):
        """All domains.
        """
        logger.info('Sampling all-label corpus.')

        groups = defaultdict(list)

        for line in tqdm(self):
            groups[line.label].append(line)

        return LineDataset.downsample(groups)

    def sample_a_vs_b(self, a, b, size):
        """Domain A vs domain B.
        """
        logger.info('Sampling A-vs-B corpus.')

        groups = defaultdict(list)

        for line in tqdm(self):
            if line.label in (a, b):
                groups[line.label].append(line)

        return LineDataset.downsample(groups, size)


class LineDataset(Dataset):

    @classmethod
    def downsample(cls, groups, size=None):
        """Downsample grouped lines.

        Args:
            groups (dict<str, list<Line>>)
        """
        size = size or min([len(lines) for _, lines in groups.items()])

        pairs = [
            random.sample([(line, label) for line in lines], size)
            for label, lines in groups.items()
        ]

        return cls(list(chain(*pairs)))

    def __init__(self, pairs):
        self.pairs = pairs
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

    def token_counts(self):
        """Collect all token -> count.
        """
        logger.info('Gathering token counts.')

        counts = Counter()
        for line, _ in tqdm(self.pairs):
            counts.update(line.tokens)

        return counts

    def label_counts(self):
        """Label -> count.
        """
        logger.info('Gathering label counts.')

        counts = Counter()
        for _, label in tqdm(self.pairs):
            counts[label] += 1

        return counts

    def labels(self):
        counts = self.label_counts()
        return [label for label, _ in counts.most_common()]
