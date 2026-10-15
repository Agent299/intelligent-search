"""Per-query statistics from log aggregation (doc p.149: 'the system explicitly counts the query frequency ...
used later to determine which queries to pre-process and where to cache them')."""
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.execute("""
    CREATE TABLE query_stats (
        normalized_query TEXT PRIMARY KEY,
        corrected_query  TEXT,                 -- after spell correction (enrichment)
        frequency        INTEGER NOT NULL,
        freq_rank        INTEGER NOT NULL,     -- 1 = most frequent
        tier             TEXT NOT NULL,        -- l1 (top 5%) | l2 (top 20%) | warm (rest)
        sessions         INTEGER NOT NULL,
        users            INTEGER NOT NULL,
        clicks           INTEGER NOT NULL,
        carts            INTEGER NOT NULL,
        purchases        INTEGER NOT NULL,
        top_purchased    TEXT[] NOT NULL DEFAULT '{}',
        related_searches TEXT[] NOT NULL DEFAULT '{}',   -- mined from session co-occurrence (Tier 0)
        fast_confidence  REAL,                 -- offline fast-path gate (Tier 0)
        status           TEXT NOT NULL DEFAULT 'aggregated',
            -- aggregated | fast_ok | skipped_rare | segmented | expanded | failed
        updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX query_stats_rank ON query_stats (freq_rank);
    CREATE INDEX query_stats_status ON query_stats (status);
    """)


def downgrade() -> None:
    op.execute("DROP TABLE query_stats;")
