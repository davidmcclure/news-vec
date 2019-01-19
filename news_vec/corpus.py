

import pandas as pd
import pickle
import random

from cached_property import cached_property
from collections import Counter, UserList, UserDict
from tqdm import tqdm
from itertools import chain, islice
from functools import lru_cache

from torch.utils.data import random_split

from .utils import read_json_gz_lines
from . import logger


class HeadlineDataset(UserList):

    @classmethod
    def from_df(cls, df, label_col='domain', **kwargs):
        pairs = [(d, d[label_col]) for d in df.to_dict('records')]
        return cls(pairs, **kwargs)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    def __init__(self, pairs, test_frac=0.1):
        """Set train/val/test splits.
        """
        test_size = round(len(pairs) * test_frac)
        train_size = len(pairs) - (test_size * 2)

        # Set train/val/test.
        sizes = (train_size, test_size, test_size)
        self.train, self.val, self.test = random_split(pairs, sizes)

        # Zip splits onto headlines.
        for split in ('train', 'val', 'test'):
            for hl, _ in getattr(self, split):
                hl['split'] = split

    def __repr__(self):

        pattern = '{cls_name}<{train_size}/{val_size}/{test_size}>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            train_size=len(self.train),
            val_size=len(self.val),
            test_size=len(self.test),
        )

    def __iter__(self):
        return chain(self.train, self.val, self.test)

    def skim(self, n, *args, **kwargs):
        """Downsample to N pairs.

        Returns: cls
        """
        pairs = random.sample(list(iter(self)), n)

        return self.__class__(pairs, *args, **kwargs)

    def token_counts(self):
        """Collect all token -> count.
        """
        logger.info('Gathering token counts.')

        counts = Counter()
        for hl, _ in tqdm(self):
            counts.update(hl['clf_tokens'])

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

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)


class Corpus:

    def __init__(self, headline_root, skim=None):
        """Read headline df.
        """
        logger.info('Reading headlines.')
        lines = islice(read_json_gz_lines(headline_root), skim)
        self.df = pd.DataFrame(list(tqdm(lines)))

    def __repr__(self):

        pattern = '{cls_name}<{hl_count} headlines>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            hl_count=len(self.df),
        )

    @cached_property
    def min_count(self):
        return self.df.groupby('domain').size().min()

    def sample_ava(self):
        return (self.df
            .groupby('domain')
            .apply(lambda x: x.sample(self.min_count)))

    @lru_cache(None)
    def filter_ab(self, d1, d2):
        return (self.df
            [self.df.domain.isin([d1, d2])]
            .groupby('domain'))

    def sample_ab(self, d1, d2):
        return (self
            .filter_ab(d1, d2)
            .apply(lambda x: x.sample(self.min_count)))
