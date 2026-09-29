"""A tiny, safe builder for openFDA's Lucene-flavoured ``search`` expressions.

openFDA search strings are assembled by string concatenation in most client code
in the wild, which goes wrong in two ways that both produce *silently incorrect
numbers* rather than errors:

1. An unescaped double quote in a term ends the phrase early. The query still
   parses, matches something else, and returns a plausible count.
2. A range written with the wrong spacing parses as a term query and matches
   nothing, which openFDA reports as HTTP 404 -- indistinguishable from a
   legitimate "no reports".

Both failure modes are invisible in the output, so the fix belongs in a single
audited place rather than at each call site.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable

from prodrome.timeframe import as_openfda_date

#: Characters Lucene treats as syntax. Terms are always wrapped in double
#: quotes, so only the quote and the backslash actually need escaping inside one.
_ESCAPE = re.compile(r'([\\"])')


def phrase(value: str) -> str:
    """Quote and escape a term for use as an exact phrase match.

    Reaction preferred terms legitimately contain apostrophes ("Crohn's
    disease"), hyphens, commas and parentheses. All are safe inside double
    quotes; the quote character itself is not, and neither is a backslash.
    """
    if not value or not value.strip():
        raise ValueError("cannot build a phrase from an empty term")
    return '"' + _ESCAPE.sub(r"\\\1", value.strip()) + '"'


def field_is(field: str, value: str) -> str:
    """``field:"value"`` with the value escaped."""
    return f"{field}:{phrase(value)}"


def date_range(field: str, start: dt.date, end: dt.date) -> str:
    """``field:[YYYYMMDD TO YYYYMMDD]``, inclusive at both ends.

    Raises:
        ValueError: if the range runs backwards, which is always a bug in the
            caller rather than an empty result to be tolerated.
    """
    if end < start:
        raise ValueError(f"date range runs backwards: {start} .. {end}")
    return f"{field}:[{as_openfda_date(start)} TO {as_openfda_date(end)}]"


def all_of(*clauses: str) -> str:
    """Conjoin clauses with ``AND``, parenthesising to keep precedence explicit.

    Empty clauses are dropped so callers can pass a conditional filter without
    branching at the call site.
    """
    kept = [c for c in clauses if c and c.strip()]
    if not kept:
        raise ValueError("at least one clause is required")
    if len(kept) == 1:
        return kept[0]
    return " AND ".join(f"({c})" for c in kept)


def any_of(field: str, values: Iterable[str]) -> str:
    """``field:("a" OR "b")`` for a set of alternatives."""
    quoted = [phrase(v) for v in values]
    if not quoted:
        raise ValueError("at least one value is required")
    if len(quoted) == 1:
        return f"{field}:{quoted[0]}"
    return f"{field}:({' OR '.join(quoted)})"
