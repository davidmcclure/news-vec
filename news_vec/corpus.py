

from itertools import islice
from tqdm import tqdm
from collections import Counter, UserDict, UserList

from . import logger
from .utils import read_json_gz_lines


class Headline(UserDict):

    def __repr__(self):

        pattern = '{cls_name}<{token_count} tokens -> {domain}>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            token_count=len(self['clf_tokens']),
            domain=self['domain'],
        )


class HeadlineDataset(UserList):

    def token_counts(self):
        """Collect all token -> count.
        """
        logger.info('Gathering token counts.')

        counts = Counter()
        for hl, _ in tqdm(self):
            counts.update(hl['tokens'])

        return counts

    def label_counts(self):
        """Label -> count.
        """
        logger.info('Gathering label counts.')

        counts = Counter()
        for _, label in tqdm(self):
            counts[label] += 1

        return counts

    def labels(self):
        counts = self.label_counts()
        return [label for label, _ in counts.most_common()]


class HeadlineCorpus:

    def __init__(self, root, skim=None):
        """Read lines.
        """
        logger.info('Parsing line corpus.')

        rows_iter = islice(read_json_gz_lines(root), skim)

        self.hls = {
            d['article_id']: Headline(d)
            for d in tqdm(rows_iter)
        }

    def __repr__(self):

        pattern = '{cls_name}<{hl_count} headlines>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            hl_count=len(self),
        )

    def __len__(self):
        return len(self.hls)

    def build_dataset(self, pairs):
        """(id, label) -> (Headline, label)
        """
        pairs = [(self.hls[id], label) for id, label in pairs]
        return HeadlineDataset(pairs)
