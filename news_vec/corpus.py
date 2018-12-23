

import pandas as pd

from cached_property import cached_property
from collections import Counter, UserList, UserDict
from tqdm import tqdm

from .utils import read_json_gz_lines
from . import logger


class Headline(UserDict):

    def __repr__(self):

        pattern = '{cls_name}<{token_count} tokens -> {domain}>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            token_count=len(self['clf_tokens']),
            domain=self['domain'],
        )


class HeadlineDataset(UserList):

    def __repr__(self):

        pattern = '{cls_name}<{size} pairs>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            size=len(self),
        )

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


class Corpus:

    def __init__(self, link_root, headline_root):
        """Read links df, article index.
        """
        logger.info('Reading links.')

        rows = list(tqdm(read_json_gz_lines(link_root)))
        self.links = pd.DataFrame(rows)

        logger.info('Reading headlines.')

        self.headlines = {
            row['article_id']: Headline(row)
            for row in tqdm(read_json_gz_lines(headline_root))
        }

    def __repr__(self):

        pattern = '{cls_name}<{link_count} links, {hl_count} headlines>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            link_count=len(self.links),
            hl_count=len(self.headlines),
        )

    def make_dataset(self, df):
        """Index out a list of (Headline, domain) pairs.

        Args:
            df (DataFrame with 'article_id' and 'domain')
        """
        pairs = df[['article_id', 'domain']].values.tolist()

        return HeadlineDataset([
            (self.headlines[aid], domain)
            for aid, domain in pairs
        ])

    @cached_property
    def unique_articles(self):
        return self.links[['domain', 'article_id']].drop_duplicates()

    @cached_property
    def min_domain_count(self):
        """Smallest number of unique articles per domain.
        """
        return self.unique_articles.groupby('domain').size().min()

    def sample_all_vs_all(self):
        """Sample evenly from all domains.
        """
        rows = (self.unique_articles
            .groupby('domain')
            .apply(lambda x: x.sample(self.min_domain_count)))

        return self.make_dataset(rows)

    def sample_a_vs_b(self, a, b):
        """Sample evenly from two domains.
        """
        rows = (self.unique_articles
            [self.unique_articles.domain.isin([a, b])]
            .groupby('domain')
            .apply(lambda x: x.sample(self.min_domain_count)))

        return self.make_dataset(rows)
