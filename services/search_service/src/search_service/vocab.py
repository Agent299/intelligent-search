"""Spell-correction dictionary: English + words from the catalog (names, product types, tags, brands)."""

import re
from collections import Counter

from common.db import connect
from common.normalize import SpellCorrector

_WORD = re.compile(r"[a-z][a-z'\-]{2,}")


def catalog_vocabulary() -> Counter:
    counts: Counter = Counter()
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT name, product_type, search_tags, brand FROM product_catalog")
        for row in cur.fetchall():
            text = " ".join([row["name"] or "", row["product_type"] or "", " ".join(row["search_tags"] or []),
                             row["brand"] or ""]).lower()
            counts.update(_WORD.findall(text))
    return counts


def build_spell_corrector() -> SpellCorrector:
    sc = SpellCorrector()
    sc.load_english()
    sc.add_words(dict(catalog_vocabulary()))
    return sc
