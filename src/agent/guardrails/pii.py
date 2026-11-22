"""PII rules: which columns are sensitive, and scrubbing of results and final text.

Layer 1 (sql_guard) makes PII columns unreachable. This module is layers 2 and 3:
defence in depth for anything that slips through, e.g. an email inside a free-text
column or a value the model was told by the user.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import partial

import pandas as pd

# Columns of `users` that identify a person. Never selected, never shown.
PII_COLUMNS = frozenset(
    {"first_name", "last_name", "email", "street_address", "postal_code", "latitude", "longitude", "user_geom"}
)

# Result-column names that look like PII even if aliased (e.g. `u.email AS contact_email`).
_PII_NAME = re.compile(
    r"e[-_]?mail|first[-_]?name|last[-_]?name|full[-_]?name|address|postal|zip|phone|latitude|longitude", re.I
)

# What introduces a value ("postal code:", "CA", "Customer name:", "Dr.") is not personal data. Patterns
# capture it as `label` and it stays in the text, so the reader can see what kind of value was removed.
_PAD = r"[ \t*\"'`]*"  # spaces, Markdown emphasis and quotes: **ZIP:** 94105 or "zip": "94105"
_IS_OR_COLON = rf"(?:is\b(?:{_PAD}[:=])?|[:=])"  # "is", ":", "=" and "is:"
_SEPARATOR = rf"{_PAD}{_IS_OR_COLON}{_PAD}"
_OPTIONAL_SEPARATOR = rf"{_PAD}(?:{_IS_OR_COLON}{_PAD})?"

_NAME_LABEL = r"(?:first|last|full|customer(?:['’]s)?|buyer['’]s|user['’]s)[ _-]name"
_TITLE = r"(?:Mr|Mrs|Ms|Miss|Dr)\.? "
_LETTER = r"[^\W\d_]"  # of any alphabet, so "García" is taken whole
_CAPITALISED = rf"[A-ZÀ-ÖØ-Þ]{_LETTER}+"
_NAME_WORD = rf"(?:[A-ZÀ-ÖØ-Þ]['’])?{_CAPITALISED}(?:-{_CAPITALISED})*"  # Doe, O'Neil, Smith-Jones
# Initials and particles belong to the name word they precede: "J. K. Rowling", "van Gogh", "de la Cruz".
_NAME_PREFIX = r"(?:[A-Z]\. ?|(?:van|von|der|den|de|del|della|di|da|dos|du|la|le) )*"
# First, middle and last name. All of it is taken, so that no surname is left standing, up to the title
# of a next name ("Mr. Smith Mrs. Jones").
_NAME = rf"{_NAME_PREFIX}{_NAME_WORD}(?: (?!{_TITLE}){_NAME_PREFIX}{_NAME_WORD}){{0,2}}"
# What stands in for a name that is not shown: "Customer name: Not available", "Customer name: User 37775".
_NOT_A_NAME = "(?:Not|No|None|Unknown|Unavailable|Redacted|Hidden|Withheld|Private|Personal|User|Customer|ID)"

_SIGN = "[-−]?"  # the minus sign as typed or as typeset
_US_ZIP = r"\d{5}(?:-\d{4})?(?!\d|[.,]\d)"
# Digits in most countries; "SW1A 1AA" in the United Kingdom and "K1A 0B1" in Canada.
_ANY_POSTAL_CODE = r"(?:\d{2,5}-\d{3,4}|\d{4,10}|[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z\d]{2})(?![A-Za-z\d])"
# "ID" is left out: "User ID 37775" is how this assistant refers to customers, and it is not Idaho.
_US_STATES = (
    "AL AK AZ AR CA CO CT DC DE FL GA HI IA IL IN KS KY LA MA MD ME MI MN MO MS MT "
    "NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY"
).split()

# Applied in this order, wherever they occur. Each has to stay quiet on ordinary business text.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(?<![\d.])(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")),
    (
        "address",
        re.compile(
            r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b\.?"
        ),
    ),
    # A postal code needs an introduction: a bare five-digit number is a user id, a count or money far more
    # often. A bare "zip" only counts before a US ZIP, because product names have zips too ("Full Zip 2024").
    (
        "postal_code",
        re.compile(rf"(?P<label>(?i:\b(?:postal|post|zip)[ _-]?code\b){_OPTIONAL_SEPARATOR}){_ANY_POSTAL_CODE}"),
    ),
    (
        "postal_code",
        re.compile(rf"(?P<label>(?i:\bzip\b){_OPTIONAL_SEPARATOR}|\b(?:{'|'.join(_US_STATES)})[ \t]+){_US_ZIP}"),
    ),
    # A pair of decimals with three or more places, standing alone: a longer comma-separated run is a row
    # of figures. Latitudes that start with "0." are a strip along the equator; leaving them out keeps
    # fractions ("0.181, 0.238") from being read as a position.
    (
        "coordinates",
        re.compile(
            rf"(?<![\w.,])(?<!,\s){_SIGN}[1-9]\d?\.\d{{3,}}\s*,\s*{_SIGN}\d{{1,3}}\.\d{{3,}}(?!\d|\s*,\s*{_SIGN}\d)"
        ),
    ),
    (
        "coordinates",
        re.compile(
            rf"(?P<label>(?i:\b(?:latitude|longitude|lat|lng|lon)\b){_OPTIONAL_SEPARATOR}){_SIGN}\d{{1,3}}(?:\.\d+)?(?!\d)"
        ),
    ),
    # "name" alone is not a label: categories, brands and reports have names too.
    (
        "name",
        re.compile(rf"(?P<label>(?i:\b{_NAME_LABEL}\b){_SEPARATOR})(?!{_NOT_A_NAME}\b)(?:{_TITLE})?{_NAME}"),
    ),
)

# Common given names that are not also ordinary words, months, places or clothing terms (so no Will,
# Mark, Grace, May, Jordan, Virginia, Madison, Chase, Jean or Heather), to keep the name heuristic quiet.
_GIVEN_NAMES = (
    "James John Robert Michael William David Richard Joseph Thomas Charles Christopher Daniel Matthew Anthony "
    "Donald Steven Paul Andrew Joshua Kenneth Kevin Brian George Edward Ronald Timothy Jason Jeffrey Ryan Jacob "
    "Nicholas Eric Jonathan Stephen Larry Justin Scott Benjamin Samuel Gregory Alexander Patrick "
    "Raymond Dennis Jerry Aaron Adam Nathan Henry Douglas Zachary Kyle Walter Ethan Jeremy Harold Keith Roger "
    "Gerald Arthur Albert Bruce Philip Ralph Vincent "
    "Mary Patricia Jennifer Linda Barbara Susan Jessica Sarah Karen Lisa Nancy Betty Margaret Sandra "
    "Ashley Kimberly Emily Donna Michelle Carol Amanda Dorothy Melissa Deborah Stephanie Rebecca Sharon Laura "
    "Cynthia Kathleen Amy Angela Shirley Anna Brenda Pamela Emma Nicole Helen Samantha Katherine Christine Debra "
    "Rachel Carolyn Janet Catherine Maria Diane Julie Joyce Evelyn Megan Andrea Hannah Jacqueline Martha Gloria "
    "Teresa Sara Janice Julia Judy Alice Jane Olivia Sophia Natalie Lauren Christina Danielle Diana Theresa "
    "Dmitry Sergey Alexei Andrei Mikhail Vladimir Olga Natalia Tatiana Svetlana Ekaterina Anastasia "
    "Mohammed Ahmed Omar Fatima Aisha Juan Carlos Luis Miguel Pedro Sofia Camila Giovanni Pierre Hans "
    "Priya Rahul Amit Wei Hiroshi Yuki"
).split()
_GIVEN_NAME = rf"(?:{'|'.join(_GIVEN_NAMES)})"

# Shapes a person's name shares with brands and products ("Dr. Martens", "Miss Me", "Michael Kors"), so
# they are judged by where they come from: see scan_text.
_TITLED_NAME = re.compile(rf"(?P<label>\b{_TITLE}){_NAME}")  # Mr. Smith, Dr. Priya K. Patel
# A given name and a surname, with a middle name, initials or particles between them: Jane Doe, John Q. Public.
# The capitalised word before them is their label: it tells "Sean John Jeans" from "Customer John Smith". It
# is one plain word, because it is tried at every capitalised word of a text and has to fail fast.
_FULL_NAME = re.compile(
    rf"(?P<label>\b(?!{_GIVEN_NAME}\b){_CAPITALISED}\.? )?\b{_GIVEN_NAME} (?:{_GIVEN_NAME} )?{_NAME_PREFIX}{_NAME_WORD}"
)


# Two capitalised words that the sentence itself presents as a person, whatever the given name is:
# "customer Dmitry Ivanov", "Dmitry Ivanov is your top customer". The role word is lower-case on
# purpose: in a heading ("Customer Lifetime Value") the capitalised words are a term, not a name.
_ROLE = "(?:customer|buyer|user|client|shopper)"
# Only words that rank a person may stand between: "the largest market by customer count" is about a place.
_RANK = r"(?:top|best|biggest|largest|leading|main|key|first|second|third|only|new|loyal|number one|most [\w-]+|highest[- ][\w-]+)"
_TWO_NAME_WORDS = rf"(?!{_NOT_A_NAME}\b){_NAME_WORD} {_NAME_PREFIX}{_NAME_WORD}"
_NAME_AFTER_ROLE = re.compile(rf"(?P<label>\b{_ROLE}(?: named| called)?,?[ :]+){_TWO_NAME_WORDS}")
_NAME_BEFORE_ROLE = re.compile(
    rf"\b{_TWO_NAME_WORDS}(?=,? (?:is|was|remains|became) (?:your|our|the|a|an) (?:{_RANK} ){{0,2}}{_ROLE}\b)"
)


def _redaction(kind: str, match: re.Match[str]) -> str:
    """What a match is replaced with: the label that introduced the value, if any, and the marker."""
    label = match.groupdict().get("label") or ""
    return f"{label}[REDACTED {kind.replace('_', ' ').upper()}]"


def _occurs_in(name: str, sources: list[str]) -> bool:
    """Do the first two words of a name occur, as whole words, in one of the case-folded `sources`?

    Two words are enough to tell where a name comes from, and they keep "Ralph Lauren Polo" known when
    the data says "Polo Ralph Lauren". The full stop of a title or an initial and the amount of white
    space are left out of the comparison: "Dr Martens" in an answer is "Dr. Martens" in the data.
    """
    first_two_words = name.casefold().split()[:2]
    first, second = [re.escape(word.rstrip(".")) for word in first_two_words]
    as_whole_words = rf"(?<!{_LETTER}){first}\.?\s+{second}(?!{_LETTER})"
    return any(re.search(as_whole_words, source) for source in sources)


def _is_known(match: re.Match[str], sources: list[str]) -> bool:
    """Does a name occur in one of `sources`, read with its label or without it?

    With the label, "Dr. Martens" is known and so is "Sean John Jeans" when the data has the brand Sean
    John. Without it, "Customer Michael Kors" is known when the data has Michael Kors.
    """
    label = match.groupdict().get("label") or ""
    name = match.group()[len(label) :]
    return _occurs_in(match.group(), sources) or (" " in name and _occurs_in(name, sources))


def _redact_unsourced(shape: re.Pattern[str], text: str, sources: list[str]) -> tuple[str, int]:
    """Redact the names of one shape that occur in none of `sources`. Returns (text, names redacted)."""
    redacted: list[str] = []

    def keep_or_redact(match: re.Match[str]) -> str:
        if _is_known(match, sources):
            return match.group()
        redacted.append(match.group())
        return _redaction("name", match)

    return shape.sub(keep_or_redact, text), len(redacted)


def scan_text(text: str, known: Iterable[str] | None = None) -> tuple[str, list[str]]:
    """Redact PII-looking substrings. Returns (clean_text, kinds_found).

    Contact details, postal codes, coordinates and a name behind a label ("Customer name: ...") are
    redacted wherever they occur. So is a name behind a title ("Mr. Smith"), unless a known text has it.

    A full name on its own ("Jane Doe") has the same shape as many brands, so it is judged by where it
    comes from. `known` holds the texts a name may legitimately come from: query results and the brand
    names of the user's scope. Personal data never leaves the warehouse, so a person-looking name in
    model-written text that is in neither was invented, injected or typed by the user, and it is
    redacted. A name is person-looking when it starts with a common given name, or when the sentence
    presents it as a person ("customer Dmitry Ivanov", "Dmitry Ivanov is your top customer"). Brands named after people
    (Michael Kors, Jessica Simpson) come from query results and therefore stay, and so do brands with a
    title (Dr. Martens, Miss Me).

    `known=()` is the strictest setting, for logs: every person-looking name is redacted. `known=None`
    skips the check of full names, because without sources they cannot be told from brands.
    """
    found: list[str] = []
    for kind, pattern in _PATTERNS:
        text, replacements = pattern.subn(partial(_redaction, kind), text)
        found.extend([kind] * replacements)
    sources = [] if known is None else [source.casefold() for source in known]
    name_shapes = [_TITLED_NAME] if known is None else [_TITLED_NAME, _FULL_NAME, _NAME_AFTER_ROLE, _NAME_BEFORE_ROLE]
    for shape in name_shapes:
        text, replacements = _redact_unsourced(shape, text, sources)
        found.extend(["name"] * replacements)
    return text, found


def mask_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop PII-named columns and redact PII-looking values. Returns (df, events)."""
    events: list[str] = []
    bad = [c for c in df.columns if c.lower() in PII_COLUMNS or _PII_NAME.search(c)]
    if bad:
        df = df.drop(columns=bad)
        events.append(f"dropped columns: {', '.join(bad)}")
    for col in df.columns:
        if df[col].dtype != object and not pd.api.types.is_string_dtype(df[col]):
            continue
        hits, values = 0, []
        for value in df[col]:
            if isinstance(value, str):
                # A cell is its own source: a person-looking name in a query result is a brand or a product.
                value, found = scan_text(value, known=[value])
                hits += len(found)
            values.append(value)
        if hits:
            df = df.assign(**{col: values})
            events.append(f"redacted {hits} value(s) in column {col}")
    return df, events
