"""Negation handling (Tier 0). Retrieval treats every query word as something to match, so
'snacks without nuts' pulls in walnuts and 'dress not too formal' pulls in formal dresses. We strip
negated terms from what is sent to retrieval and drop results that contain them, unless the product
explicitly declares their absence ('nut-free', 'no added sugar').

'X-free' phrases stay in the query: 'gluten free pasta' should still match products labelled gluten-free.
"""

import re
from dataclasses import dataclass, field

_NEGATION = re.compile(r"\b(?:without|no|not|non)\s+(?:(?:too|any|added|very)\s+)*([a-z][a-z\-]{2,})")
_FREE = re.compile(r"\b([a-z]{3,})[\s\-]free\b")
_FILLER = re.compile(r"\b(?:that's|that is|which is|that are|with|and|but)\s*$")


@dataclass
class ParsedQuery:
    positive: str                                 # what retrieval searches for
    negated: list[str] = field(default_factory=list)

    def excludes(self, text: str) -> bool:
        """True when `text` (product name/type) mentions a negated term without declaring its absence."""
        low = text.lower()
        for term in self.negated:
            stem = term[:-1] if term.endswith("s") and len(term) > 3 else term
            if not re.search(rf"\b\w*{re.escape(stem)}s?\b", low):
                continue
            declared_absent = re.search(
                rf"\b(?:{re.escape(stem)}s?[\s\-]free|no (?:added )?{re.escape(stem)}s?|without {re.escape(stem)}s?)\b",
                low)
            if not declared_absent:
                return True
        return False


def parse_negations(normalized: str) -> ParsedQuery:
    negated = [m.group(1) for m in _NEGATION.finditer(normalized)]
    negated += [m.group(1) for m in _FREE.finditer(normalized)]
    if not negated:
        return ParsedQuery(normalized)
    positive = _NEGATION.sub(" ", normalized)
    positive = _FILLER.sub("", re.sub(r"\s+", " ", positive).strip()).strip()
    return ParsedQuery(positive or normalized, list(dict.fromkeys(negated)))
