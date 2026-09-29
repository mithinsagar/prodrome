"""Tests for the quarter abstraction that defines point-in-time semantics."""

from __future__ import annotations

import datetime as dt

import pytest

from prodrome.timeframe import Quarter, as_openfda_date, quarters_between


@pytest.mark.parametrize(
    ("label", "year", "quarter", "start", "end"),
    [
        ("2023Q1", 2023, 1, dt.date(2023, 1, 1), dt.date(2023, 3, 31)),
        ("2023Q2", 2023, 2, dt.date(2023, 4, 1), dt.date(2023, 6, 30)),
        ("2023Q3", 2023, 3, dt.date(2023, 7, 1), dt.date(2023, 9, 30)),
        ("2020Q4", 2020, 4, dt.date(2020, 10, 1), dt.date(2020, 12, 31)),
    ],
)
def test_boundaries(label: str, year: int, quarter: int, start: dt.date, end: dt.date) -> None:
    q = Quarter.parse(label)
    assert (q.year, q.quarter) == (year, quarter)
    assert q.start_date == start
    assert q.end_date == end
    assert q.label == label


def test_leap_year_q1_ends_on_march_31_regardless() -> None:
    assert Quarter(2024, 1).end_date == dt.date(2024, 3, 31)


@pytest.mark.parametrize("bad", ["2023Q5", "2023Q0", "23Q1", "2023-Q1", "", "Q1"])
def test_rejects_malformed_labels(bad: str) -> None:
    with pytest.raises(ValueError, match=r"not a quarter label|quarter must be"):
        Quarter.parse(bad)


def test_rejects_out_of_range_quarter_on_construction() -> None:
    with pytest.raises(ValueError, match="quarter must be 1-4"):
        Quarter(2023, 7)


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        (dt.date(2023, 1, 1), "2023Q1"),
        (dt.date(2023, 3, 31), "2023Q1"),
        (dt.date(2023, 4, 1), "2023Q2"),
        (dt.date(2023, 12, 31), "2023Q4"),
    ],
)
def test_containing(date: dt.date, expected: str) -> None:
    assert Quarter.containing(date).label == expected


def test_shift_crosses_year_boundaries_in_both_directions() -> None:
    assert Quarter(2023, 4).shift(1).label == "2024Q1"
    assert Quarter(2023, 1).shift(-1).label == "2022Q4"
    assert Quarter(2023, 2).shift(9).label == "2025Q3"
    assert Quarter(2023, 2).shift(-9).label == "2021Q1"


def test_shift_round_trips() -> None:
    q = Quarter(2021, 3)
    for n in range(-20, 21):
        assert q.shift(n).shift(-n) == q


def test_subtraction_counts_quarters() -> None:
    assert Quarter(2024, 1) - Quarter(2023, 1) == 4
    assert Quarter(2023, 1) - Quarter(2024, 1) == -4
    assert Quarter(2023, 3) - Quarter(2023, 3) == 0


def test_ordering_enables_min_as_signal_onset() -> None:
    fired = [Quarter(2023, 2), Quarter(2022, 4), Quarter(2024, 1)]
    assert min(fired).label == "2022Q4"
    assert sorted(q.label for q in fired) == ["2022Q4", "2023Q2", "2024Q1"]


def test_quarters_between_is_inclusive() -> None:
    got = quarters_between(Quarter(2022, 3), Quarter(2023, 2))
    assert [q.label for q in got] == ["2022Q3", "2022Q4", "2023Q1", "2023Q2"]
    assert quarters_between(Quarter(2023, 1), Quarter(2023, 1)) == [Quarter(2023, 1)]


def test_quarters_between_rejects_a_backwards_window() -> None:
    with pytest.raises(ValueError, match="runs backwards"):
        quarters_between(Quarter(2023, 3), Quarter(2022, 1))


def test_openfda_date_format() -> None:
    assert as_openfda_date(dt.date(2023, 9, 30)) == "20230930"
    assert as_openfda_date(dt.date(2023, 1, 5)) == "20230105"
