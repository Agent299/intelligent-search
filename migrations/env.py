from alembic import context
from sqlalchemy import create_engine

from common.config import get_settings


def run_migrations_online() -> None:
    dsn = get_settings().postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    with create_engine(dsn).connect() as conn:
        context.configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
