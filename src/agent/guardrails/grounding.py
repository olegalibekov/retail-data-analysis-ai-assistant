"""Groundedness checks for percentages and explicit measured counts.

A model that does arithmetic in its head gets percentages slightly wrong. This
check finds percentages in an answer or report that do not occur in any query
result of the conversation, either as written or as a fraction (0.181 for 18.1%).
The user's question and the model's own earlier answers are not evidence.

Explicit customer, order, item, user and purchase counts are also matched exactly
to result numbers; dates, inactivity windows and top-N suggestions are excluded.
This checks numeric presence, not metric attribution or SQL correctness.

The model gets one chance per turn to compute such a figure in SQL or to leave it
out. What is still unsupported after that is taken out by code and marked, so unsupported matched figures are not shown as verified facts.
"""

from __future__ import annotations

import re

MARKER = "[figure removed]"
REMOVAL_NOTE = (
    "_Some figures could not be verified against query results and were removed; they are marked [figure removed]._"
)

# A percentage as written: "18.1%", "18.1 %", "1,234.5%", "18.1 percent". A number with thousands separators
# is one figure, not "234.5%" behind a comma. The sign in front is matched so that it goes with the figure
# when the figure is removed.
_PERCENT = re.compile(
    r"(?<![\w.])[-+−]?(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?:%|(?i:percent|per cent)\b)"
)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    """Every number in a text.

    A comma between digits separates thousands in prose (1,234) and columns in a query result
    (7,120). The text is read both ways: missing a number of a query result would take a correct
    figure away from the user.
    """
    without_thousands_separators = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    numbers = _NUMBER.findall(text) + _NUMBER.findall(without_thousands_separators)
    return {float(number) for number in numbers}


def _is_supported(shown: str, known: set[float]) -> bool:
    """Does a percentage as written ("18.1") match a known number, as a percentage or as a fraction (0.181)?"""
    value = float(shown.replace(",", ""))
    decimals = len(shown.partition(".")[2])
    tolerance = 0.5 * 10**-decimals + 1e-9  # allow rounding to the precision shown
    for number in known:
        as_percentage, as_fraction = abs(number), abs(number) * 100
        if abs(as_percentage - value) <= tolerance or abs(as_fraction - value) <= tolerance:
            return True
    return False


def ungrounded_percentages(text: str, sources: list[str]) -> list[str]:
    """Percentages in `text` that cannot be traced to a number in `sources`, each as "18.1%"."""
    known = _numbers("\n".join(sources)) | {0.0, 100.0}
    written = dict.fromkeys(match.group("number") for match in _PERCENT.finditer(text))  # unique, in order
    return [f"{number}%" for number in written if not _is_supported(number, known)]


# Explicit measured counts, not dates, inactivity windows, IDs or requested top-N lists.
_COUNT = re.compile(
    r"(?<![\w.,])(?P<number>\d{1,3}(?:,\d{3})+|\d+)\s+"
    r"(?:(?:total|currently|active|churned|unique|distinct|returning|new|repeat|paying)\s+){0,3}"
    r"(?:customers?|orders?|items?|users?|purchases?)\b",
    re.I,
)


def _count_matches(text: str):
    for match in _COUNT.finditer(text):
        if not re.search(r"\b(?:top|best|first|last|next)\s+$", text[max(0, match.start() - 16) : match.start()], re.I):
            yield match


def ungrounded_figures(text: str, sources: list[str]) -> list[str]:
    """Check percentages and explicit customer/order/item counts against tool evidence."""
    known = _numbers("\n".join(sources))
    counts = [
        match.group() for match in _count_matches(text) if float(match.group("number").replace(",", "")) not in known
    ]
    return ungrounded_percentages(text, sources) + list(dict.fromkeys(counts))


def replace_with_marker(text: str, figures: list[str]) -> str:
    """`text` with every occurrence of the given checked figures replaced by the marker.

    `figures` are percentages as ungrounded_percentages reports them. They are looked up with the
    pattern that found them, so removing "5%" leaves "15%" alone.
    """
    unsupported = set(figures)

    def marker_or_figure(match: re.Match[str]) -> str:
        if f"{match.group('number')}%" in unsupported:
            return MARKER
        return match.group()

    text = _PERCENT.sub(marker_or_figure, text)
    for match in reversed(list(_count_matches(text))):
        if match.group() in unsupported:
            start, end = match.span("number")
            text = text[:start] + MARKER + text[end:]
    return text


def remove_unsupported(text: str, figures: list[str]) -> str:
    """`text` without the given checked figures, followed by a note that says what the marker stands for."""
    return f"{replace_with_marker(text, figures).rstrip()}\n\n{REMOVAL_NOTE}"
