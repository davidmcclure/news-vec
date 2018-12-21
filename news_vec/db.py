

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
from itertools import islice


def read_json_gz_lines(root):
    """Read JSON corpus.

    Yields: dict
    """
    for path in glob('%s/*.gz' % root):
        print(path)
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
