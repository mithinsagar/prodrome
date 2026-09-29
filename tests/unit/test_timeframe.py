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


class TestVersionSampling:
    """Label histories are reduced to what the quarter grain can resolve.

    Pembrolizumab has 100 archived versions and each is a multi-megabyte download, so
    fetching versions the analysis cannot distinguish is pure cost.
    """

    def test_keeps_the_last_version_in_each_quarter(self) -> None:
        """A reaction added part-way through a quarter should be detected in that
        quarter, not the next one -- so the last version wins, not the first."""
        import datetime as dt

        from prodrome.clients.dailymed import SplVersion
        from prodrome.ingest.labels import sample_versions_to_quarters

        versions = [
            SplVersion(1, dt.date(2023, 1, 10)),
            SplVersion(2, dt.date(2023, 2, 20)),
            SplVersion(3, dt.date(2023, 3, 30)),  # last of 2023Q1
            SplVersion(4, dt.date(2023, 5, 5)),  # only one in 2023Q2
        ]
        kept = [v.version for v in sample_versions_to_quarters(versions)]
        assert kept == [1, 3, 4], "baseline plus the last of each quarter"

    def test_always_keeps_the_first_archived_version(self) -> None:
        """It defines the left-truncation baseline. Dropping it in favour of the
        quarter's last version could miss an addition made inside that quarter and
        wrongly mark the pair prevalent."""
        import datetime as dt

        from prodrome.clients.dailymed import SplVersion
        from prodrome.ingest.labels import sample_versions_to_quarters

        versions = [SplVersion(i, dt.date(2023, 1, i)) for i in range(1, 6)]
        kept = sample_versions_to_quarters(versions)
        assert kept[0].version == 1
        assert len(kept) == 2, "baseline plus the quarter's last version"

    def test_a_long_history_collapses_to_at_most_one_per_quarter_plus_baseline(self) -> None:
        import datetime as dt

        from prodrome.clients.dailymed import SplVersion
        from prodrome.ingest.labels import sample_versions_to_quarters

        # 100 versions spread over 5 years: at most 20 quarters plus the baseline.
        versions = [
            SplVersion(i, dt.date(2020, 1, 1) + dt.timedelta(days=18 * i)) for i in range(100)
        ]
        kept = sample_versions_to_quarters(versions)
        assert len(kept) <= 21, f"expected quarter-grain reduction, got {len(kept)}"
        quarters = {Quarter.containing(v.published).label for v in kept}
        assert len(kept) - len(quarters) <= 1, "at most the baseline may share a quarter"

    def test_empty_history_is_empty(self) -> None:
        from prodrome.ingest.labels import sample_versions_to_quarters

        assert sample_versions_to_quarters([]) == []
