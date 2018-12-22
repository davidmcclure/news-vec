

import numpy as np

from itertools import islice
from tqdm import tqdm
from collections import Counter, UserDict, UserList, defaultdict
from boltons.iterutils import pairwise
from cached_property import cached_property

from . import logger
from .utils import read_json_gz_lines


class LinkCorpus:

    def __init__(self, root, n_ts_buckets=10):
        """Read lines.
        """
        self.n_ts_buckets = 10

        logger.info('Parsing line corpus.')

        rows_iter = read_json_gz_lines(root)
        self.rows = list(tqdm(rows_iter))

    def __iter__(self):
        return iter(self.rows)

    @cached_property
    def min_ts(self):
        return min(r['timestamp'] for r in self)

    @cached_property
    def max_ts(self):
        return max(r['timestamp'] for r in self)

    @cached_property
    def ts_buckets(self):
        """Split out N temporal buckets.
        """
        buckets = np.linspace(self.min_ts, self.max_ts,
            self.n_ts_buckets, dtype='int')

        return pairwise(buckets)

    @cached_property
    def idx(self):
        """domain -> article id -> ts bucket -> impressions.
        """
        idx = defaultdict(lambda: defaultdict(Counter))

        for r in tqdm(self):
            for i, (ts1, ts2) in enumerate(self.ts_buckets):
                if ts1 <= r['timestamp'] <= ts2:
                    idx[r['domain']][r['article_id']][i] += r['fc']

        return idx

    def articles_by_domain_iter(self, domain, min_imp):
        """Generate articles from a domain.
        """
        for aid, counts in self.idx[domain].items():
            if sum(counts.values()) > min_imp:
                yield aid

    def min_domain_count(self, min_imp):
        """Number of articles in most infrequent domain.
        """
        counts = []
        for d in self.idx:
            articles = self.articles_by_domain_iter(d, min_imp)
            counts.append(len(list(articles)))

        return min(counts)


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
