"""Query normalization (doc p.150): fix typos, standardize case, strip irrelevant characters."""

import hashlib
import re
import unicodedata
from importlib import resources

from symspellpy import SymSpell, Verbosity

_NON_WORD = re.compile(r"[^\w\s\-&']+")
_SPACES = re.compile(r"\s+")


def normalize_query(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).lower()
    q = _NON_WORD.sub(" ", q)
    return _SPACES.sub(" ", q).strip()


def query_hash(normalized: str) -> str:
    """Key for the L1/L2 caches (doc p.171)."""
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


class SpellCorrector:
    """Per-token correction (Tier 0, replaces the LLM spell-check in the doc), so 'basmatti' -> 'basmati'.

    The dictionary is general English plus the catalog vocabulary. English keeps ordinary words
    ('dorm', 'party') from being "corrected" into catalog terms; catalog words get a frequency boost so
    product and brand spellings win ties. Short tokens get at most one edit to avoid over-correction.
    """

    CATALOG_BOOST = 1_000

    def __init__(self, max_edit_distance: int = 2):
        self._sym = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        self._max = max_edit_distance
        self._catalog: set[str] = set()

    def load_english(self) -> None:
        path = resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        self._sym.load_dictionary(str(path), term_index=0, count_index=1)

    def add_words(self, counts: dict[str, int], boost: bool = True) -> None:
        for word, count in counts.items():
            self._sym.create_dictionary_entry(word, count * (self.CATALOG_BOOST if boost else 1))
        if boost:
            self._catalog.update(counts)

    def correct(self, normalized: str) -> str:
        out = []
        for token in normalized.split():
            if len(token) < 4 or any(c.isdigit() for c in token):
                out.append(token)
                continue
            distance = self._max if len(token) >= 8 else 1
            hits = self._sym.lookup(token, Verbosity.CLOSEST, max_edit_distance=distance, include_unknown=True)
            if not hits:
                out.append(token)
                continue
            # Among equally close suggestions prefer a word the catalog actually contains
            # ('jekry' -> 'jerky', not the more common English 'jerry'); otherwise the most frequent.
            best = min(hits, key=lambda h: (h.distance, h.term not in self._catalog, -h.count))
            out.append(best.term)
        return " ".join(out)
