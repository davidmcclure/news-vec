

import os
import ujson
import gzip
import click

from glob import glob
from boltons.iterutils import chunked_iter
from tqdm import tqdm
from itertools import islice

from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.schema import Index

from news_vec import logger


DB_PATH = os.path.join(os.path.dirname(__file__), 'links.db')


def connect_db(db_path):
    """Get database connection.

    Args:
        db_path (str)

    Returns: engine, session
    """
    url = URL(drivername='sqlite', database=db_path)
    engine = create_engine(url)

    # Fix transaction bugs in pysqlite.
    # http://docs.sqlalchemy.org/en/latest/dialects/sqlite.html#pysqlite-serializable

    @event.listens_for(engine, 'connect')
    def on_connect(conn, record):
        conn.execute('pragma foreign_keys=ON')
        conn.isolation_level = None

    @event.listens_for(engine, 'begin')
    def on_begin(conn):
        conn.execute('BEGIN')

    factory = sessionmaker(bind=engine)
    session = scoped_session(factory)

    return engine, session


class BaseModel:

    @classmethod
    def add_index(cls, *cols, **kwargs):
        """Add an index to the table.
        """
        # Make slug from column names.
        col_names = '_'.join([c.name for c in cols])

        # Build the index name.
        name = 'idx_{}_{}'.format(cls.__tablename__, col_names)

        idx = Index(name, *cols, **kwargs)

        # Render the index.
        try:
            idx.create(bind=engine)
        except Exception as e:
            print(e)


engine, session = connect_db(DB_PATH)
BaseModel = declarative_base(cls=BaseModel)
BaseModel.query = session.query_property()


def read_json_lines(root):
    """Read JSON line dump.

    Yields: dict
    """
    for path in tqdm(glob('%s/*.gz' % root)):
        with gzip.open(path) as fh:
            for line in fh:
                yield ujson.loads(line)


class Link(BaseModel):

    __tablename__ = 'link'

    id = Column(Integer, primary_key=True)
    domain = Column(String, nullable=False)
    article_id = Column(Integer, nullable=False)
    actor_id = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)

    @classmethod
    def load(cls, root, page_size=10000, limit=None):
        """Batch-insert rows.
        """
        rows = islice(read_json_lines(root), limit)

        for mappings in chunked_iter(rows, page_size):
            session.bulk_insert_mappings(cls, mappings)

        session.commit()

    @classmethod
    def add_indexes(cls):
        cls.add_index(cls.domain)
        cls.add_index(cls.article_id)
        cls.add_index(cls.actor_id)


@click.command()
@click.argument('root', type=click.Path())
@click.option('--limit', type=int, default=None)
def load(root, limit):
    """Reset + load db.
    """
    logger.info('Resetting database.')
    BaseModel.metadata.drop_all(engine)
    BaseModel.metadata.create_all(engine)

    logger.info('Inserting rows.')
    Link.load(root, limit=limit)

    logger.info('Building indexes.')
    Link.add_indexes()


if __name__ == '__main__':
    load()
