"""Tests against the live public APIs.

Marked ``network`` and excluded from CI. Not because the API is unreliable -- every
shape the pipeline sends succeeds 45/45 -- but because a test that reaches a
third-party service can fail for reasons unrelated to the change under review, and a
suite that fails for unrelated reasons is a suite people stop reading. These run on
the weekly schedule instead, where a failure is information.

What they exist to catch is narrow but important: an upstream *contract* change. The
count aggregation's key name, the UNII harmonisation fields, the SPL archive layout
and the version-history date format are all undocumented behaviours this project
depends on, and any of them changing would break the pipeline silently.

Run with:  pytest -m network
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from prodrome.clients.base import ApiClient, HttpCache, RateLimiter
from prodrome.clients.dailymed import (
    DailyMedClient,
    active_ingredient_uniis,
    labeller_from_title,
)
from prodrome.clients.openfda import EVENT_ENDPOINT, OpenFdaClient
from prodrome.labelmatch.sectioning import Tier, parse_label
from prodrome.selector import DrugSelector
from prodrome.timeframe import Quarter

pytestmark = pytest.mark.network

# Semaglutide, and the label FDA added ileus to in September 2023. Used as ground
# truth throughout: the transition is a dated, public regulatory event.
SEMAGLUTIDE_UNII = "53AXN4NNHX"
OZEMPIC_SET_ID = "adec4fd2-6858-4c99-91d4-531f5f2a2d79"
ILEUS_ABSENT_VERSION = 13  # published 2022-10-07
ILEUS_PRESENT_VERSION = 14  # published 2023-10-09

SELECTOR = DrugSelector(
    unii=SEMAGLUTIDE_UNII,
    substance_names=("SEMAGLUTIDE",),
    brand_names=("OZEMPIC",),
)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("live-cache")


@pytest.fixture(scope="module")
def openfda(cache_dir: Path) -> OpenFdaClient:
    transport = ApiClient(
        base_url=EVENT_ENDPOINT,
        namespace="openfda-event",
        cache=HttpCache(cache_dir),
        limiter=RateLimiter(60),
        max_retries=8,
        backoff_base_seconds=2.0,
        max_sleep_seconds=45.0,
    )
    return OpenFdaClient(transport, has_api_key=False)


@pytest.fixture(scope="module")
def dailymed(cache_dir: Path) -> DailyMedClient:
    transport = ApiClient(
        base_url="https://dailymed.nlm.nih.gov/dailymed",
        namespace="dailymed",
        cache=HttpCache(cache_dir),
        limiter=RateLimiter(30),
        timeout_seconds=180.0,
        max_retries=5,
        backoff_base_seconds=2.0,
    )
    return DailyMedClient(transport)


class TestCountFieldGuard:
    """openFDA answers a count over an analysed string field with HTTP 500.

    A client error dressed as a server error gets retried, and the retries hide it.
    This asserts both halves of the finding against the live API, so a change in
    openFDA's behaviour shows up here rather than as a mystery outage.
    """

    def test_the_analysed_form_really_does_fail(self, openfda: OpenFdaClient) -> None:
        from prodrome.clients.openfda import CountFieldError, validate_count_field

        with pytest.raises(CountFieldError, match="analysed string field"):
            validate_count_field("occurcountry")

    def test_the_exact_form_works(self, openfda: OpenFdaClient) -> None:
        response = openfda.reporter_country_counts(SELECTOR, "NAUSEA", Quarter(2024, 4))
        assert response.counts, "expected a country breakdown"
        assert "US" in response.counts, "FAERS is a US database; US should dominate"

    def test_coded_fields_must_not_carry_exact(self, openfda: OpenFdaClient) -> None:
        """qualification, sex and serious are coded, not analysed."""
        response = openfda.qualification_counts(SELECTOR, "NAUSEA", Quarter(2024, 4))
        assert response.counts
        assert set(response.counts) <= {"1", "2", "3", "4", "5"}


class TestOpenFdaContract:
    def test_metadata_reports_a_data_vintage(self, openfda: OpenFdaClient) -> None:
        updated = openfda.last_updated()
        assert isinstance(updated, dt.date)
        assert updated > dt.date(2020, 1, 1)

    def test_latest_complete_quarter_trails_the_vintage(self, openfda: OpenFdaClient) -> None:
        """The quarter containing last_updated is still accruing reports."""
        assert openfda.latest_complete_quarter() < Quarter.containing(openfda.last_updated())

    def test_count_value_equals_the_report_total_for_one_term(self, openfda: OpenFdaClient) -> None:
        """The property the whole request budget rests on.

        If this stops holding, one count request no longer substitutes for a row of
        targeted queries and every co-occurrence count in the warehouse is wrong.
        """
        quarter = Quarter(2024, 1)
        aggregated = openfda.drug_reaction_counts(SELECTOR, quarter)
        term = next(t for t in aggregated.counts if aggregated.counts[t] > 50)
        targeted = openfda.drug_reaction_reports(SELECTOR, term, quarter)
        assert aggregated.counts[term] == targeted

    def test_unii_harmonisation_coverage_is_measurable(self, openfda: OpenFdaClient) -> None:
        """Coverage is bimodal and must be measurable, not assumed."""
        coverage = openfda.measure_selector_coverage(SELECTOR)
        assert coverage.union_reports > 10_000
        assert 0.0 <= coverage.unii_coverage <= 1.0
        # The substance-name clause must recall at least as much as the UNII clause;
        # for several cohort drugs it is the only clause that recalls anything.
        assert coverage.substance_reports >= coverage.unii_reports

    def test_a_date_count_uses_the_time_key(self, openfda: OpenFdaClient) -> None:
        """Undocumented: date aggregations return "time", term aggregations "term"."""
        buckets = openfda.quarterly_new_reports(SELECTOR, "NAUSEA")
        assert buckets, "expected quarterly buckets for a high-volume pair"
        assert all(Quarter.parse(label) for label in buckets)


class TestDailyMedContract:
    def test_version_history_is_dated_and_ordered(self, dailymed: DailyMedClient) -> None:
        history = dailymed.version_history(OZEMPIC_SET_ID)
        assert len(history) >= 15, "the application-holder label has many revisions"
        assert history == sorted(history)
        assert history[0].published < history[-1].published

    def test_version_numbers_are_not_contiguous(self, dailymed: DailyMedClient) -> None:
        """Ozempic's history skips version 10.

        Iterating range(1, n+1) would request a version that does not exist -- and
        DailyMed answers that with HTTP 200 and an HTML error page.
        """
        versions = [entry.version for entry in dailymed.version_history(OZEMPIC_SET_ID)]
        assert versions != list(range(1, len(versions) + 1))

    def test_an_archived_version_yields_its_own_document(self, dailymed: DailyMedClient) -> None:
        document = dailymed.fetch_version(OZEMPIC_SET_ID, ILEUS_ABSENT_VERSION)
        assert document is not None
        assert document.effective_date is not None
        assert document.effective_date.year == 2022

    def test_active_ingredient_extraction_excludes_excipients(
        self, dailymed: DailyMedClient
    ) -> None:
        """The document lists water and propylene glycol alongside semaglutide."""
        document = dailymed.fetch_version(OZEMPIC_SET_ID, ILEUS_PRESENT_VERSION)
        assert document is not None
        uniis = active_ingredient_uniis(document.xml)
        assert set(uniis) == {SEMAGLUTIDE_UNII}

    def test_labeller_is_extractable_from_the_title(self, dailymed: DailyMedClient) -> None:
        title = dailymed.title(OZEMPIC_SET_ID) or ""
        assert "NOVO NORDISK" in labeller_from_title(title).upper()


class TestGroundTruthLabelTransition:
    """FDA added ileus to Ozempic's label in September 2023.

    This is the project's ground truth, and this test is the one that would catch a
    regression in sectioning, spelling normalisation or synonymy turning a real label
    change into a missed one.
    """

    @pytest.fixture(scope="class")
    def parsed(self, dailymed: DailyMedClient) -> dict[int, object]:
        out = {}
        for version in (ILEUS_ABSENT_VERSION, ILEUS_PRESENT_VERSION):
            document = dailymed.fetch_version(OZEMPIC_SET_ID, version)
            assert document is not None
            out[version] = parse_label(document.xml)
        return out

    def test_both_versions_parse_to_usable_safety_text(self, parsed: dict[int, object]) -> None:
        for version, label in parsed.items():
            assert label.is_usable, f"v{version} has too little core safety text"
            assert label.core_char_count > 5_000

    def test_clinical_studies_is_excluded(self, parsed: dict[int, object]) -> None:
        """It runs to ~18,000 characters and names adverse events as trial outcomes."""
        codes = {s.code for s in parsed[ILEUS_PRESENT_VERSION].sections}
        assert "34092-7" not in codes
        assert "34067-9" not in codes, "indications must be excluded too"

    def test_ileus_appears_between_the_two_versions(self, parsed: dict[int, object]) -> None:
        before = parsed[ILEUS_ABSENT_VERSION].text_for((Tier.CORE,)).lower()
        after = parsed[ILEUS_PRESENT_VERSION].text_for((Tier.CORE,)).lower()
        assert "ileus" not in before, "v13 (2022-10) should not mention ileus"
        assert "ileus" in after, "v14 (2023-10) should mention ileus"

    def test_pancreatitis_is_labelled_in_both(self, parsed: dict[int, object]) -> None:
        """An already-labelled control: it must not look like a new addition."""
        for label in parsed.values():
            assert "pancreatitis" in label.text_for((Tier.CORE,)).lower()
