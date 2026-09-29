"""Tests for the openFDA client's response handling.

The count-aggregation behaviour tested here was all discovered by probing the live
API; none of it is documented, and each item is something that silently produces
wrong numbers rather than an error.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
import respx

from prodrome.clients import query
from prodrome.clients.base import ApiClient, HttpCache, RateLimiter
from prodrome.clients.openfda import (
    ANALYSED_STRING_FIELDS,
    COUNT_LIMIT_ANONYMOUS,
    COUNT_LIMIT_WITH_KEY,
    EVENT_ENDPOINT,
    FIELD_COUNTRY,
    CountFieldError,
    OpenFdaClient,
    validate_count_field,
)
from prodrome.selector import DrugSelector
from prodrome.timeframe import Quarter

SELECTOR = DrugSelector(unii="53AXN4NNHX", substance_names=("SEMAGLUTIDE",))


def build(tmp_path: Path, *, has_key: bool = True) -> tuple[OpenFdaClient, ApiClient]:
    transport = ApiClient(
        base_url=EVENT_ENDPOINT,
        namespace="openfda-event",
        cache=HttpCache(tmp_path / "cache"),
        limiter=RateLimiter(240),
        max_retries=1,
        backoff_base_seconds=0.001,
        max_sleep_seconds=0.01,
        api_key="k" if has_key else None,
    )
    return OpenFdaClient(transport, has_api_key=has_key), transport


class TestCountKeyHandling:
    @respx.mock
    def test_term_aggregations_use_the_term_key(self, tmp_path: Path) -> None:
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": [{"term": "NAUSEA", "count": 537}]})
        )
        client, transport = build(tmp_path)
        with transport:
            assert client.drug_reaction_counts(SELECTOR, Quarter(2024, 1)).counts == {"NAUSEA": 537}

    @respx.mock
    def test_date_aggregations_use_the_time_key(self, tmp_path: Path) -> None:
        """openFDA returns "time", not "term", when counting a date field.

        Undocumented, and the reason a term-only reader raises KeyError on the
        quarterly-volume harvest.
        """
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"time": "20230115", "count": 4},
                        {"time": "20230520", "count": 7},
                    ]
                },
            )
        )
        client, transport = build(tmp_path)
        with transport:
            buckets = client.quarterly_new_reports(SELECTOR, "ILEUS")
        assert buckets == {"2023Q1": 4, "2023Q2": 7}

    @respx.mock
    def test_unparseable_date_buckets_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"time": "not-a-date", "count": 3},
                        {"time": "20230115", "count": 4},
                    ]
                },
            )
        )
        client, transport = build(tmp_path)
        with transport:
            assert client.quarterly_new_reports(SELECTOR, "ILEUS") == {"2023Q1": 4}


class TestTruncationSemantics:
    @respx.mock
    def test_short_response_is_exhaustive_so_absence_is_zero(self, tmp_path: Path) -> None:
        """The distinction that stops the pipeline writing false zeros."""
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": [{"term": "NAUSEA", "count": 5}]})
        )
        client, transport = build(tmp_path)
        with transport:
            response = client.drug_reaction_counts(SELECTOR, Quarter(2024, 1))
        assert not response.truncated
        assert response.get("NAUSEA") == 5
        assert response.get("ILEUS") == 0, "exhaustive response: absence is a real zero"

    @respx.mock
    def test_full_response_is_truncated_so_absence_is_unknown(self, tmp_path: Path) -> None:
        """Ileus for semaglutide has 21 reports and is not in the top 100.

        Reading its absence from a truncated aggregation as zero is how a pipeline
        silently loses its most important cells.
        """
        rows = [{"term": f"TERM{i}", "count": 1000 - i} for i in range(COUNT_LIMIT_WITH_KEY)]
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": rows})
        )
        client, transport = build(tmp_path)
        with transport:
            response = client.drug_reaction_counts(SELECTOR, Quarter(2024, 1))
        assert response.truncated
        assert response.get("ILEUS") is None, "truncated response: absence is unknown"

    def test_count_limit_depends_on_having_a_key(self, tmp_path: Path) -> None:
        with_key, t1 = build(tmp_path, has_key=True)
        without_key, t2 = build(tmp_path / "b", has_key=False)
        try:
            assert with_key.count_limit == COUNT_LIMIT_WITH_KEY
            assert without_key.count_limit == COUNT_LIMIT_ANONYMOUS
        finally:
            t1.close()
            t2.close()

    @respx.mock
    def test_limit_is_not_sent_without_a_key(self, tmp_path: Path) -> None:
        """Sending limit with count and no key is a 403, not a clamp."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"results": []})

        respx.get(url__startswith=EVENT_ENDPOINT).mock(side_effect=handler)
        client, transport = build(tmp_path, has_key=False)
        with transport:
            client.drug_reaction_counts(SELECTOR, Quarter(2024, 1))
        assert "limit=" not in str(captured[0].url)


class TestEmptyResults:
    @respx.mock
    def test_no_matches_is_zero_not_an_error(self, tmp_path: Path) -> None:
        respx.get(url__startswith=EVENT_ENDPOINT).mock(return_value=httpx.Response(404))
        client, transport = build(tmp_path)
        with transport:
            assert client.total_reports(Quarter(2024, 1)) == 0
            assert client.drug_reports(SELECTOR, Quarter(2024, 1)) == 0

    @respx.mock
    def test_malformed_envelope_raises_rather_than_returning_zero(self, tmp_path: Path) -> None:
        """A shape change upstream must not be read as 'no reports'."""
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"unexpected": "shape"})
        )
        client, transport = build(tmp_path)
        with transport, pytest.raises(ValueError, match="unexpected openFDA envelope"):
            client.total_reports(Quarter(2024, 1))


class TestWindowClause:
    def test_window_is_cumulative_and_uses_receivedate(self, tmp_path: Path) -> None:
        """Cumulative, and on receivedate: see prodrome.timeframe for why."""
        client, transport = build(tmp_path)
        with transport:
            clause = client.window_clause(Quarter(2023, 3))
        assert clause == "receivedate:[20040101 TO 20230930]"
        assert "receiptdate" not in clause


class TestLatestCompleteQuarter:
    @respx.mock
    def test_excludes_the_quarter_still_accruing(self, tmp_path: Path) -> None:
        """The quarter containing last_updated is partial; including it would read
        as a decline in every count series."""
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(
                200, json={"meta": {"last_updated": "2026-07-30"}, "results": []}
            )
        )
        client, transport = build(tmp_path)
        with transport:
            assert client.latest_complete_quarter().label == "2026Q2"


class TestAnalysedFieldGuard:
    """openFDA answers a count over an analysed string field with HTTP 500, not 400.

    This is the single most expensive bug encountered while building this project. A
    client error arriving as a server error is retried by any sane HTTP client, and the
    retries then hide it: `count=occurcountry` failed deterministically, was absorbed
    nine times per call, and the whole index was misdiagnosed as ~40% unreliable. The
    guard exists so the error surfaces before a request is spent.
    """

    @pytest.mark.parametrize(
        "field",
        [
            "occurcountry",
            "primarysourcecountry",
            "patient.reaction.reactionmeddrapt",
            "patient.drug.medicinalproduct",
            "patient.drug.openfda.unii",
            "patient.drug.activesubstance.activesubstancename",
        ],
    )
    def test_analysed_fields_are_rejected_with_the_remedy_in_the_message(self, field: str) -> None:
        with pytest.raises(CountFieldError, match=rf"Use {re.escape(field)}\.exact instead"):
            validate_count_field(field)

    @pytest.mark.parametrize(
        "field",
        [
            "occurcountry.exact",
            "primarysourcecountry.exact",
            "patient.reaction.reactionmeddrapt.exact",
            # Coded and date fields are NOT analysed and must not carry .exact.
            "primarysource.qualification",
            "patient.patientsex",
            "serious",
            "receivedate",
        ],
    )
    def test_countable_fields_pass_through_unchanged(self, field: str) -> None:
        assert validate_count_field(field) == field

    def test_the_country_constant_carries_exact(self) -> None:
        """A regression guard on the exact constant that caused the outage."""
        assert FIELD_COUNTRY == "occurcountry.exact"
        assert FIELD_COUNTRY not in ANALYSED_STRING_FIELDS

    @respx.mock
    def test_a_bad_count_field_never_reaches_the_network(self, tmp_path: Path) -> None:
        """The whole point: fail before spending quota on a guaranteed 500."""
        route = respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        client, transport = build(tmp_path)
        with transport, pytest.raises(CountFieldError):
            client.stratum_counts(SELECTOR, "NAUSEA", Quarter(2024, 1), "occurcountry")
        assert route.call_count == 0

    @respx.mock
    def test_the_diagnostics_helpers_use_countable_fields(self, tmp_path: Path) -> None:
        """Every stratum helper must already send a countable field."""
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": [{"term": "US", "count": 5}]})
        )
        client, transport = build(tmp_path)
        with transport:
            assert client.reporter_country_counts(SELECTOR, "NAUSEA", Quarter(2024, 1)).counts
            assert client.qualification_counts(SELECTOR, "NAUSEA", Quarter(2024, 1)).counts


class TestUnqueryableTerms:
    """openFDA returns reaction terms it will not accept back as search values.

    It encodes the apostrophe in eponymous terms as a caret, so its own count
    aggregation yields `CROHN^S DISEASE`, `PARKINSON^S DISEASE` and
    `FOURNIER^S GANGRENE` -- and then rejects those strings with BAD_REQUEST, because
    `^` is Lucene's boost operator. Raw, backslash-escaped, percent-encoded, and
    apostrophe- or space-substituted forms all fail.

    These are not obscure: Fournier's gangrene is an FDA-warned adverse event for SGLT2
    inhibitors, and the first version of this pipeline lost empagliflozin entirely to it.
    """

    @pytest.mark.parametrize(
        "term", ["FOURNIER^S GANGRENE", "CROHN^S DISEASE", "PARKINSON^S DISEASE"]
    )
    def test_caret_terms_are_recognised_as_unsearchable(self, term: str) -> None:
        assert not query.is_searchable(term)

    @pytest.mark.parametrize("term", ["NAUSEA", "COVID-19", "ILL-DEFINED DISORDER"])
    def test_ordinary_terms_including_hyphens_are_searchable(self, term: str) -> None:
        """Hyphens are fine -- only the caret is rejected."""
        assert query.is_searchable(term)

    def test_single_character_tokens_are_dropped(self) -> None:
        """The caret encoding leaves a stray "S" from the possessive, which on its own
        matches an enormous number of unrelated reports."""
        assert query.tokens_of("FOURNIER^S GANGRENE") == ("FOURNIER", "GANGRENE")

    @respx.mock
    def test_an_unsearchable_term_is_routed_through_a_count_bucket(self, tmp_path: Path) -> None:
        """The bucket is exact. The token narrowing is only a filter.

        Measured against the live API: AND-ing the tokens of `CROHN^S DISEASE` returns
        58,021 reports where the exact term has 57,974, because other terms share both
        tokens. So the narrowed query must never be used as the answer -- only to keep
        the aggregation small enough to read the right bucket out of.
        """
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"term": "CROHN^S DISEASE", "count": 57974},
                        {"term": "CROHN^S DISEASE COMPLICATION", "count": 47},
                    ]
                },
            )

        respx.get(url__startswith=EVENT_ENDPOINT).mock(side_effect=handler)
        client, transport = build(tmp_path)
        with transport:
            got = client.reaction_reports("CROHN^S DISEASE", Quarter(2026, 2))

        assert got == 57974, "the exact bucket, not the token-narrowed total"
        sent = str(captured[0].url)
        assert "count=patient.reaction.reactionmeddrapt.exact" in sent
        assert "CROHN%5ES" not in sent, "the caret must not reach the search clause"
        assert "CROHN" in sent and "DISEASE" in sent, "narrowed by tokens"

    @respx.mock
    def test_a_truncated_bucket_search_reports_zero_and_warns(self, tmp_path: Path) -> None:
        """If the aggregation truncated and the term is absent, its count is genuinely
        unrecoverable. Zero is recorded, and the log says so -- silently guessing would
        put a fabricated number into a published statistic."""
        rows = [{"term": f"OTHER TERM {i}", "count": 1000 - i} for i in range(1000)]
        respx.get(url__startswith=EVENT_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"results": rows})
        )
        client, transport = build(tmp_path)
        with transport:
            assert client.reaction_reports("CROHN^S DISEASE", Quarter(2026, 2)) == 0

    @respx.mock
    def test_a_searchable_term_still_uses_the_direct_path(self, tmp_path: Path) -> None:
        """The fallback must not become the default: it costs an aggregation."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"meta": {"results": {"total": 778508}}})

        respx.get(url__startswith=EVENT_ENDPOINT).mock(side_effect=handler)
        client, transport = build(tmp_path)
        with transport:
            assert client.reaction_reports("NAUSEA", Quarter(2026, 2)) == 778508
        assert "count=" not in str(captured[0].url)
