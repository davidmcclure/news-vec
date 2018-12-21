

import gzip
import ujson
import os

from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, event

from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Column, Integer, BigInteger, String, func
from sqlalchemy.schema import Index
from sqlalchemy.ext.declarative import declarative_base

from glob import glob
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from collections import UserDict, UserList, Counter
from itertools import islice

from . import logger


def read_json_gz_lines(root):
    """Read JSON corpus.

    Yields: dict
    """
    for path in glob('%s/*.gz' % root):
        with gzip.open(path) as fh:
            for line in fh:
                yield ujson.loads(line)


def connect_db(db_path):
    """Get database connection.

    Args:
        db_path (str)

    Returns: engine, session
    """
    url = URL(drivername='sqlite', database=db_path)
    engine = create_engine(url)

    factory = sessionmaker(bind=engine)
    session = scoped_session(factory)

    return engine, session


DB_PATH = os.path.join(os.path.dirname(__file__), 'newsvec.db')
engine, session = connect_db(DB_PATH)


class BaseModel:

    @classmethod
    def reset(cls):
        """Drop and recreate table.
        """
        cls.metadata.drop_all(engine, tables=[cls.__table__])
        cls.metadata.create_all(engine, tables=[cls.__table__])

    @classmethod
    def load_spark_lines(cls, root, n=1000):
        """Bulk-insert spark lines.
        """
        cls.reset()

        pages = tqdm(chunked_iter(read_json_gz_lines(root), n))

        for mappings in pages:
            session.bulk_insert_mappings(cls, mappings)

        session.commit()

    @classmethod
    def add_index(cls, *cols, **kwargs):
        """Add an index to the table.
        """
        # Make slug from column names.
        col_names = '_'.join([c.name for c in cols])

        # Build the index name.
        name = f'idx_{cls.__tablename__}_{col_names}'

        idx = Index(name, *cols, **kwargs)

        # Render the index.
        try:
            idx.create(bind=engine)
        except Exception as e:
            print(e)

        print(col_names)

    def columns(self):
        """Get a list of column names.

        Returns: list
        """
        return [c.name for c in self.__table__.columns]

    def __iter__(self):
        """Generate column / value tuples.

        Yields: (key, val)
        """
        for key in self.columns():
            yield (key, getattr(self, key))


BaseModel = declarative_base(cls=BaseModel)
BaseModel.query = session.query_property()


class Link(BaseModel):

    __tablename__ = 'link'
    id = Column(BigInteger, primary_key=True)
    actor_id = Column(String, nullable=False)
    article_id = Column(Integer, nullable=False)
    domain = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)

    @classmethod
    def add_indexes(cls):
        cls.add_index(cls.domain)

    @classmethod
    def domains(cls):
        """Unique domains.
        """
        query = session.query(cls.domain.distinct())
        return [r[0] for r in query]

    @classmethod
    def domain_article_counts(cls):
        """Total articles per domain.
        """
        query = (session
            .query(cls.domain, func.count(cls.article_id.distinct()))
            .group_by(cls.domain))

        return Counter(dict(query))

    @classmethod
    def min_domain_article_count(cls):
        """Count of most infrequent domain.
        """
        counts = cls.domain_article_counts()

        return counts.most_common()[-1][1]

    @classmethod
    def sample_domain(cls, domain, n):
        """Sample N random headlines from a domain.
        """
        query = (session
            .query(cls.article_id.distinct())
            .filter(cls.domain==domain)
            .order_by(func.random())
            .limit(n))

        return [r[0] for r in query]

    @classmethod
    def sample_all_vs_all_iter(cls, n=None):
        """Sample N articles from each domain.
        """
        n = n or cls.min_domain_article_count()

        for domain in cls.domains():
            for id in cls.sample_domain(domain, n):
                yield (id, domain)


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
