"""Reconstructing a drug's label timeline and deciding what each version said.

This is the half of the pipeline that does not exist anywhere else. Everything that
compares adverse events to labels compares them to the label as it stands *today*,
because that is what openFDA's label endpoint serves. That answers "is this
reaction labelled" and cannot answer "since when", which is the only version of the
question that supports a latency measurement.

What this module builds, per drug:

1. The version history from DailyMed -- version numbers and publication dates.
2. Each archived version's document, unpacked from its ZIP.
3. Each document reduced to its safety sections by LOINC code.
4. A verdict per (reaction, version): does this version describe this reaction, in
   which tier, on what evidence.

From that, :mod:`prodrome.latency.onset` derives the first date each reaction
appeared, which is the outcome variable.

Two properties the implementation has to preserve
-------------------------------------------------
**A missing version is a gap, not a zero.** Some versions listed in history are not
retrievable, and some documents do not parse. Those versions are recorded as
unusable and excluded, because reading them as "the reaction was not mentioned"
would fabricate a label gap -- or, worse, fabricate an *un-labelling* between two
versions that both mention it, which would corrupt the first-mention date.

**Version numbers are not contiguous.** Ozempic's history runs 1-9 then 11-20,
skipping 10. Iterating ``range(1, n+1)`` would request a version that does not
exist and, because DailyMed answers that with an HTTP 200 carrying an HTML error
page, would appear to succeed.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from prodrome.clients.dailymed import DailyMedClient, LabelDocument
from prodrome.config import CohortDrug, Config
from prodrome.labelmatch.decide import LabelMatcher, MentionDecision
from prodrome.labelmatch.embed import Embedder
from prodrome.labelmatch.sectioning import ParsedLabel, parse_label

logger = logging.getLogger(__name__)

#: Seed for sampling the null-term pool. Fixed so that a label-match verdict is
#: reproducible: the empirical null depends on which terms were sampled, so an
#: unseeded sample would let two runs disagree about whether a reaction is labelled.
NULL_SAMPLE_SEED = 20240101


@dataclass(frozen=True, slots=True)
class LabelVersionRecord:
    """One archived label version, parsed and assessed."""

    drug_unii: str
    spl_set_id: str
    document: LabelDocument
    parsed: ParsedLabel

    @property
    def is_usable(self) -> bool:
        return self.parsed.is_usable

    def as_row(self, run_id: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "drug_unii": self.drug_unii,
            "spl_set_id": self.spl_set_id,
            "spl_version": self.document.version,
            "published_date": self.document.published,
            "effective_date": self.document.effective_date,
            "authoritative_date": self.document.authoritative_date,
            "core_char_count": self.parsed.core_char_count,
            "section_count": len(self.parsed.sections),
            "is_usable": self.is_usable,
        }


@dataclass(frozen=True, slots=True)
class MentionRecord:
    """A verdict on one (reaction, label version) pair."""

    drug_unii: str
    spl_version: int
    authoritative_date: object
    decision: MentionDecision

    def as_row(self, run_id: str) -> dict[str, object]:
        row: dict[str, object] = {
            "run_id": run_id,
            "drug_unii": self.drug_unii,
            "spl_version": self.spl_version,
            "authoritative_date": self.authoritative_date,
            "reaction": self.decision.reaction,
            "verdict": self.decision.verdict.value,
            "is_labelled": self.decision.is_labelled,
            "best_semantic_score": self.decision.best_semantic_score,
            "best_semantic_z": self.decision.best_semantic_z,
            "evidence_method": None,
            "evidence_score": None,
            "evidence_section_code": None,
            "evidence_section_name": None,
            "evidence_tier": None,
            "evidence_snippet": None,
            "evidence_via": None,
        }
        evidence = self.decision.evidence
        if evidence is not None:
            row.update(
                {
                    "evidence_method": evidence.method,
                    "evidence_score": evidence.score,
                    "evidence_section_code": evidence.section_code,
                    "evidence_section_name": evidence.section_name,
                    "evidence_tier": evidence.tier.value,
                    "evidence_snippet": evidence.snippet[:2000],
                    "evidence_via": evidence.via,
                }
            )
        return row


def build_null_pool(
    reactions_by_drug: Mapping[str, Sequence[str]], *, size: int = 400
) -> list[str]:
    """Assemble the null-term pool for the empirical-null label matcher.

    Terms are drawn from *other* drugs in the cohort, which makes the null what it
    needs to be: reaction terms that are plausible MedDRA vocabulary but have no
    particular reason to appear on this drug's label. Drawing from a generic English
    word list would make the null far too easy to beat, and every weak semantic
    match would clear it.

    The sample is shuffled with a fixed seed so verdicts are reproducible.
    """
    pool: list[str] = []
    for terms in reactions_by_drug.values():
        pool.extend(terms)
    unique = sorted(set(pool))
    # Seeded deliberately: the empirical null depends on which terms were sampled,
    # so an unseeded shuffle would let two runs disagree about whether a reaction
    # is labelled. Reproducibility is the requirement, not unpredictability.
    random.Random(NULL_SAMPLE_SEED).shuffle(unique)  # noqa: S311
    return unique[:size]


class LabelTimelineBuilder:
    """Reconstructs label timelines and assesses reaction mentions."""

    def __init__(
        self,
        client: DailyMedClient,
        embedder: Embedder,
        config: Config,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._config = config

    def build_timeline(self, drug: CohortDrug) -> list[LabelVersionRecord]:
        """Fetch and parse every retrievable version of a drug's label.

        Returns them oldest first. A drug with no pinned ``spl_set_id`` yields
        nothing and is logged: it can still contribute disproportionality series,
        but not a label outcome, so it is excluded from the latency analysis rather
        than treated as never-labelled.
        """
        if not drug.spl_set_id:
            logger.warning(
                "%s has no spl_set_id pinned; it will have no label timeline and is "
                "excluded from latency analysis",
                drug.name,
            )
            return []

        records: list[LabelVersionRecord] = []
        history = self._client.version_history(drug.spl_set_id)
        if not history:
            logger.warning("no label history for %s (%s)", drug.name, drug.spl_set_id)
            return []

        # Each archive is a multi-megabyte ZIP and a cold backfill fetches every
        # version of every drug, so this is the longest-running phase of the
        # pipeline by a wide margin. It logs progress rather than going silent for
        # tens of minutes, because a silent process is indistinguishable from a
        # hung one.
        logger.info(
            "%s: fetching %d label versions (%s .. %s)",
            drug.name,
            len(history),
            history[0].published,
            history[-1].published,
        )
        for position, entry in enumerate(history, start=1):
            document = self._client.fetch_version(drug.spl_set_id, entry.version)
            if position % 10 == 0 or position == len(history):
                logger.info("%s: %d/%d versions fetched", drug.name, position, len(history))
            if document is None:
                continue
            parsed = parse_label(document.xml)
            if not parsed.is_usable:
                logger.info(
                    "%s SPL v%s has only %d characters of core safety text; recorded "
                    "as unusable rather than as 'nothing labelled'",
                    drug.name,
                    entry.version,
                    parsed.core_char_count,
                )
            records.append(
                LabelVersionRecord(
                    drug_unii=drug.unii,
                    spl_set_id=drug.spl_set_id,
                    document=document,
                    parsed=parsed,
                )
            )
        logger.info(
            "%s: %d/%d label versions retrievable, %d usable",
            drug.name,
            len(records),
            len(history),
            sum(1 for r in records if r.is_usable),
        )
        return records

    def assess_mentions(
        self,
        records: Sequence[LabelVersionRecord],
        reactions: Sequence[str],
        null_terms: Sequence[str],
    ) -> Iterator[MentionRecord]:
        """Decide, for every version and reaction, whether the label describes it.

        One :class:`LabelMatcher` per version, because the passage embeddings and the
        empirical null are properties of that version's text. Reactions are then
        queried against it, so cost scales with versions rather than with
        versions x reactions.
        """
        settings = self._config.labelmatch
        for record in records:
            matcher = LabelMatcher(
                record.parsed,
                self._embedder,
                null_terms=null_terms,
                sentences_per_chunk=settings.chunk_sentences,
                stride=settings.chunk_stride,
                top_k=settings.top_k,
            )
            logger.debug(
                "%s v%s: %d chunks, index=%s, null mean=%.3f sd=%.3f",
                record.drug_unii,
                record.document.version,
                matcher.chunk_count,
                matcher.index_backend,
                matcher.null.mean,
                matcher.null.std,
            )
            for reaction in reactions:
                yield MentionRecord(
                    drug_unii=record.drug_unii,
                    spl_version=record.document.version,
                    authoritative_date=record.document.authoritative_date,
                    decision=matcher.decide(reaction),
                )
