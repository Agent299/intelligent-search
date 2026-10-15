"""LLM enrichment fields on the catalog (doc p.146: 'Uses an LLM to read product info and extract tags')."""
from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.execute("""
    ALTER TABLE product_catalog
        ADD COLUMN product_type    TEXT,
        ADD COLUMN search_tags     TEXT[] NOT NULL DEFAULT '{}',
        ADD COLUMN llm_enriched_at TIMESTAMPTZ;
    """)


def downgrade() -> None:
    op.execute("""
    ALTER TABLE product_catalog
        DROP COLUMN llm_enriched_at,
        DROP COLUMN search_tags,
        DROP COLUMN product_type;
    """)
