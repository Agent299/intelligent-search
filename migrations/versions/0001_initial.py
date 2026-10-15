"""Initial schema: catalog, user profile, smart query cache, taxonomy, co-purchase (doc p.161-167).

Postgres stands in for DynamoDB; partition keys from the doc become primary keys and the
category/price GSI becomes a composite index.
"""
from alembic import op

revision = "0001"
down_revision = None


def upgrade() -> None:
    op.execute("""
    CREATE TABLE product_catalog (
        product_id    TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        description   TEXT NOT NULL DEFAULT '',
        brand         TEXT,
        attributes    JSONB NOT NULL DEFAULT '{}',
        price         DOUBLE PRECISION NOT NULL,
        currency      TEXT NOT NULL DEFAULT 'INR',
        image_url     TEXT,
        category_id   TEXT,
        category_path TEXT,
        in_stock      BOOLEAN NOT NULL DEFAULT TRUE,
        rating        REAL,
        review_count  INTEGER NOT NULL DEFAULT 0,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX product_catalog_category_price ON product_catalog (category_id, price);

    CREATE TABLE taxonomy (
        category_id TEXT PRIMARY KEY,
        path        TEXT NOT NULL UNIQUE,   -- e.g. 'Groceries > Rice, Flour & Pulses > Basmati Rice'
        parent_id   TEXT REFERENCES taxonomy(category_id),
        level       SMALLINT NOT NULL
    );

    CREATE TABLE user_profile (
        user_id          TEXT PRIMARY KEY,
        search_history   TEXT[] NOT NULL DEFAULT '{}',
        purchase_history TEXT[] NOT NULL DEFAULT '{}',
        preferences      JSONB NOT NULL DEFAULT '{}',  -- {"brand_affinities": {"Nike": 0.8}, "sizes": [...]}
        updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    -- Persistent tier of the smart query cache: normalized query -> compiled ES query.
    CREATE TABLE smart_query_cache (
        normalized_query TEXT PRIMARY KEY,
        es_query         JSONB NOT NULL,
        related_searches TEXT[] NOT NULL DEFAULT '{}',
        source           TEXT NOT NULL,          -- batch | nrt | fallback
        version          BIGINT NOT NULL,
        frequency        INTEGER NOT NULL DEFAULT 0,
        updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE co_purchase (
        product_key TEXT NOT NULL,   -- primary product name/category ('brownie')
        co_item     TEXT NOT NULL,
        support     INTEGER NOT NULL,
        lift        REAL NOT NULL,
        PRIMARY KEY (product_key, co_item)
    );
    """)


def downgrade() -> None:
    op.execute("""
    DROP TABLE co_purchase;
    DROP TABLE smart_query_cache;
    DROP TABLE user_profile;
    DROP TABLE product_catalog;
    DROP TABLE taxonomy;
    """)
