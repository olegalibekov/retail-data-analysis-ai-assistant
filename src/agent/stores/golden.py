"""Golden Knowledge retrieval: representative (question, SQL, report) trios.

The prototype ranks a small local file by word overlap, which costs no model
calls. In production the trios live in a bucket and are retrieved by embedding
similarity (see docs/ARCHITECTURE.md); the interface stays the same.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_STOP = frozenset(
    "a an and are as at be by did do does for from how i in is it me my of on or our show tell that the their they "
    "this to was we were what which who why with you your give get can could please".split()
)
_SYNONYMS = {
    "user": "customer",
    "client": "customer",
    "buyer": "customer",
    "shopper": "customer",
    "sal": "revenu",
    "incom": "revenu",
}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return _SYNONYMS.get(word, word)


def _tokens(text: str) -> frozenset[str]:
    """The meaningful words of a text, stemmed, for overlap matching."""
    words = re.findall(r"[a-z]+", text.lower())
    return frozenset(_stem(word) for word in words if word not in _STOP and len(word) > 2)


def _similarity(question: frozenset[str], example: frozenset[str]) -> float:
    """Share of common words, between 0 and 1 (cosine similarity of the two word sets)."""
    return len(question & example) / math.sqrt(len(question) * len(example))


class GoldenStore:
    """Sample trios contain id, provenance, question, sql, report and tags."""

    def __init__(self, path: Path):
        self.trios: list[dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        self._words = [_tokens(trio["question"] + " " + " ".join(trio.get("tags", ()))) for trio in self.trios]

    def retrieve(self, question: str, k: int = 2, min_score: float = 0.2) -> list[dict]:
        """The `k` trios most similar to the question, best first, leaving out weak matches."""
        question_words = _tokens(question)
        if not question_words:
            return []
        scored = []
        for trio, trio_words in zip(self.trios, self._words, strict=True):
            if trio_words:
                scored.append((_similarity(question_words, trio_words), trio))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [trio for score, trio in scored[:k] if score >= min_score]
