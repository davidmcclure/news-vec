

import os

from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine

from sqlalchemy import Column, Integer, BigInteger, String, func
from sqlalchemy.schema import Index
from sqlalchemy.ext.declarative import declarative_base

from glob import glob
from tqdm import tqdm
from collections import Counter
from itertools import chain, islice
from cached_property import cached_property
from boltons.iterutils import chunked_iter

from . import logger
from .utils import read_json_gz_lines


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
    def load_spark_lines(cls, root, page_size=1000, skim=None):
        """Bulk-insert spark lines.
        """
        cls.reset()

        rows = islice(read_json_gz_lines(root), skim)

        pages = tqdm(chunked_iter(rows, page_size))

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
    article_id = Column(BigInteger, nullable=False)
    domain = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)
    fc = Column(Integer, nullable=False)

    @classmethod
    def add_indexes(cls):
        cls.add_index(cls.domain, cls.article_id)


class ArticleSampler:

    @cached_property
    def domains(self):
        """Unique domains.
        """
        query = session.query(Link.domain.distinct())
        return set([r[0] for r in query])

    def sample_ava(self, min_imp):
        """All vs all.
        """
        articles = (session
            .query(Link.article_id, Link.domain)
            .group_by(Link.article_id)
            .having(func.sum(Link.fc) > min_imp))

        return articles.all()

    # def domain_counts(self, min_imp):
    #     """Total articles per domain.
    #     """
    #     articles = (session
    #         .query(Link.domain, Link.article_id)
    #         .group_by(Link.article_id)
    #         .having(func.sum(Link.fc) > min_imp)
    #         .subquery())
    #
    #     query = (session
    #         .query(articles.c.domain, func.count(articles.c.domain))
    #         .group_by(articles.c.domain))
    #
    #     return Counter(dict(query))

    # @classmethod
    # def min_domain_article_count(cls):
    #     """Count of most infrequent domain.
    #     """
    #     counts = cls.domain_article_counts()
    #
    #     return counts.most_common()[-1][1]

    # @cached_property
    # def min_ts(self):
    #     return session.query(func.min(Link.timestamp)).one()[0]
    #
    # @cached_property
    # def max_ts(self):
    #     return session.query(func.max(Link.timestamp)).one()[0]
    #
    # @classmethod
    # def sample_domain(cls, domain, n):
    #     """Sample N random headlines from a domain.
    #     """
    #     query = (session
    #         .query(cls.article_id, cls.domain)
    #         .filter(cls.domain==domain)
    #         .group_by(cls.article_id, cls.domain)
    #         .order_by(func.random())
    #         .limit(n))
    #
    #     return query.all()
    #
    # @classmethod
    # def sample_not_domain(cls, domain, n):
    #     """Sample N random headlines *not* from a domain.
    #     """
    #     query = (session
    #         .query(cls.article_id, cls.domain)
    #         .filter(cls.domain!=domain)
    #         .group_by(cls.article_id, cls.domain)
    #         .order_by(func.random())
    #         .limit(n))
    #
    #     return query.all()
    #
    # @classmethod
    # def sample_all_vs_all(cls, n):
    #     """Sample N articles from each domain.
    #     """
    #     pairs = [cls.sample_domain(domain, n) for domain in cls.domains()]
    #     return list(chain(*pairs))
    #
    # @classmethod
    # def sample_a_vs_b(cls, a, b, n):
    #     """Sample N articles from two domains.
    #     """
    #     pairs = [cls.sample_domain(domain, n) for domain in (a, b)]
    #     return list(chain(*pairs))
    #
    # @classmethod
    # def sample_one_vs_all(cls, domain):
    #     """Sample N a domain, N from all others.
    #     """
    #     fg = cls.sample_domain(domain, n)
    #     bg = cls.sample_not_domain(domain, n)
    #     return list(chain(fg, bg))
