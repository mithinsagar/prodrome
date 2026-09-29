"""Quarters, and the point-in-time discipline the whole project depends on.

Everything prodrome computes is computed *as of* a quarter boundary, using only
reports FDA had received by that date. This module is the single place that
decides what "as of" means, so the rule cannot drift between the ingest, stats
and latency layers.

Why `receivedate` and not `receiptdate`
---------------------------------------
openFDA exposes both. ``receiptdate`` is the most recent version of a report,
which moves forward every time a follow-up is filed -- so filtering on it would
let a 2019 report that was amended in 2024 appear in a 2024 window and vanish
from the 2019 one. ``receivedate`` is the date FDA first received the report and
never changes. Only the second one supports a stable point-in-time cut, so
prodrome uses it everywhere and never offers the other as an option.

The residual caveat, stated because it bounds what this project can claim: FAERS
reports reach the public database some weeks after FDA receives them, so a
quarter-end cut reconstructs "reports with receivedate <= T", not "what a
reviewer could literally have seen on date T". The lag is short relative to the
quarter grain and it biases lead-time estimates *downward* -- against the
project's own headline -- so it is a conservative approximation rather than a
flattering one.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Self

_QUARTER_PATTERN = re.compile(r"^(?P<year>\d{4})Q(?P<quarter>[1-4])$")

#: Last month/day of each quarter, indexed by quarter number.
_QUARTER_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
_QUARTER_START = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}


@dataclass(frozen=True, order=True, slots=True)
class Quarter:
    """A calendar quarter, ordered and arithmetic-capable.

    Ordering is by (year, quarter) via ``order=True``, which is what makes
    "earliest quarter satisfying a criterion" a plain ``min()``.
    """

    year: int
    quarter: int

    def __post_init__(self) -> None:
        if not 1 <= self.quarter <= 4:
            raise ValueError(f"quarter must be 1-4, got {self.quarter}")
        if not 1900 <= self.year <= 2999:
            raise ValueError(f"implausible year {self.year}")

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse ``"2023Q3"``."""
        match = _QUARTER_PATTERN.match(text.strip().upper())
        if match is None:
            raise ValueError(f"not a quarter label: {text!r} (expected e.g. '2023Q3')")
        return cls(int(match["year"]), int(match["quarter"]))

    @classmethod
    def containing(cls, date: dt.date) -> Self:
        """The quarter a date falls in."""
        return cls(date.year, (date.month - 1) // 3 + 1)

    @property
    def label(self) -> str:
        """``"2023Q3"``. Used as the warehouse key and dashboard axis label."""
        return f"{self.year}Q{self.quarter}"

    @property
    def start_date(self) -> dt.date:
        month, day = _QUARTER_START[self.quarter]
        return dt.date(self.year, month, day)

    @property
    def end_date(self) -> dt.date:
        """Inclusive last day. This is the point-in-time cutoff."""
        month, day = _QUARTER_END[self.quarter]
        return dt.date(self.year, month, day)

    @property
    def index(self) -> int:
        """Absolute quarter index, for differencing two quarters."""
        return self.year * 4 + (self.quarter - 1)

    def shift(self, quarters: int) -> Quarter:
        """This quarter moved by `quarters` (may be negative)."""
        total = self.index + quarters
        return Quarter(total // 4, total % 4 + 1)

    def __sub__(self, other: Quarter) -> int:
        """Number of quarters from `other` to `self`; negative if earlier."""
        if not isinstance(other, Quarter):
            return NotImplemented
        return self.index - other.index

    def __str__(self) -> str:
        return self.label


def quarters_between(first: Quarter, last: Quarter) -> list[Quarter]:
    """Inclusive ascending list of quarters.

    Raises:
        ValueError: if `last` precedes `first`, which in practice means a config
            window was written backwards.
    """
    if last < first:
        raise ValueError(f"window runs backwards: {first} .. {last}")
    return [first.shift(i) for i in range(last - first + 1)]


def as_openfda_date(date: dt.date) -> str:
    """openFDA's date literal format, ``YYYYMMDD``."""
    return date.strftime("%Y%m%d")
