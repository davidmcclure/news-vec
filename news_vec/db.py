

import gzip
import ujson

from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, event

from sqlalchemy import Column, Integer, BigInteger, String, ForeignKey
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base

from glob import glob
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from itertools import islice


def read_json_gz_lines(root):
    """Read JSON corpus.

    Yields: dict
    """
    for path in tqdm(glob('%s/*.gz' % root)):
        with gzip.open(path) as fh:
            for line in fh:
                yield ujson.loads(line)


def connect_db(db_name):
    """Get database connection.

    Args:
        db_name (str)

    Returns: engine, session
    """
    url = URL(drivername='postgres', database=db_name)
    engine = create_engine(url)

    factory = sessionmaker(bind=engine)
    session = scoped_session(factory)

    return engine, session


engine, session = connect_db('newsvec')


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

        pages = chunked_iter(read_json_gz_lines(root), n)

        for mappings in pages:
            session.bulk_insert_mappings(cls, mappings)
            session.flush()

        session.commit()

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
    actor_id = Column(String, nullable=False, index=True)
    article_id = Column(Integer, nullable=False, index=True)
    domain = Column(String, nullable=False, index=True)
    timestamp = Column(Integer, nullable=False)


class Headline(BaseModel):

    __tablename__ = 'headline'
    article_id = Column(BigInteger, primary_key=True)
    domain = Column(String, nullable=False, index=True)
    tokens = Column(ARRAY(String), nullable=False)
    clf_tokens = Column(ARRAY(String), nullable=False)
