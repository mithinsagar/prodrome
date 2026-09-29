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

#: Characters that make a term unusable as a *search value* on openFDA, even quoted
#: and even backslash-escaped.
#:
#: There is exactly one, and it is an upstream defect worth stating plainly: openFDA
#: encodes the apostrophe in eponymous reaction terms as a caret, so its own ``count``
#: aggregation returns ``CROHN^S DISEASE``, ``PARKINSON^S DISEASE`` and
#: ``FOURNIER^S GANGRENE`` -- and then rejects those exact strings as search values
#: with ``BAD_REQUEST: Search not supported``, because ``^`` is Lucene's boost
#: operator. Measured: the raw term, a backslash-escaped caret, a percent-encoded
#: caret, and substituting an apostrophe or a space all fail.
#:
#: These are not obscure terms. Fournier's gangrene is an FDA-warned adverse event for
#: SGLT2 inhibitors, and empagliflozin -- an SGLT2 inhibitor -- is in this cohort. The
#: first version of this pipeline lost the entire drug to it.
#:
#: :func:`prodrome.clients.openfda.OpenFdaClient.reaction_reports` routes such terms
#: through a count aggregation instead, which returns the exact bucket. See
#: ``docs/METHODS.md``.
UNQUERYABLE_CHARACTERS = frozenset("^")

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")


def is_searchable(value: str) -> bool:
    """Whether a term can be used as an openFDA search value at all."""
    return not (UNQUERYABLE_CHARACTERS & set(value))


def tokens_of(value: str) -> tuple[str, ...]:
    """Alphanumeric tokens of a term, for narrowing an analysed-field query.

    Single characters are dropped: the caret encoding leaves a stray "S" token from
    a possessive, which matches enormous numbers of unrelated reports.
    """
    return tuple(t for t in _TOKEN_SPLIT.split(value.upper()) if len(t) > 1)


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
