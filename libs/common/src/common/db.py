from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from common.config import get_settings


@contextmanager
def connect():
    with psycopg.connect(get_settings().postgres_dsn, row_factory=dict_row) as conn:
        yield conn
