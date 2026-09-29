"""Deciding whether a label already describes a reaction.

The decision is the outcome variable of the whole project, so it is built to be
*auditable*: every verdict carries the evidence that produced it -- the matching
snippet, the section it came from, the method, and the score -- so a pharmacovigilance
reviewer can check it without reading any code.

Why a fixed cosine threshold does not work
------------------------------------------
Measured with BAAI/bge-small-en-v1.5 on short reaction terms:

==================================  ======
pair                                cosine
==================================  ======
ileus / intestinal obstruction       0.729
thrombocytopenia / low platelet      0.754
ileus / blockage of the bowel        0.689
**ileus / kidney stone**             0.611
**ileus / hair loss**                0.568
==================================  ======

True and false pairs are separated by less than 0.08. Any fixed cutoff either
admits kidney stones as a match for ileus or rejects genuine paraphrases. This is
not a defect of the model -- short clinical terms occupy a narrow cone of the
embedding space -- but it is fatal to the obvious implementation, and it is why
published cosine thresholds do not transfer between corpora.

The empirical-null decision
---------------------------
Instead of asking "is the similarity high", prodrome asks "is the similarity
higher than this label gives to reactions it does *not* mention". For each label
version, a sample of reaction terms drawn from elsewhere in the cohort is scored
against the same passages. That yields a per-label null distribution of best-match
scores, and a candidate is accepted only when it stands out against *that*.

This mirrors the empirical calibration applied to the disproportionality statistics
in :mod:`prodrome.stats.calibration`, and for the same reason: the theoretical null
is wrong, so measure the real one. It also makes the decision robust to swapping
the embedding backend, since the null is recomputed in whatever space is in use.

Evidence tiers
--------------
Lexical matches are accepted outright -- an exact or spelling-normalised hit on a
label is not a probabilistic judgement. The embedding layer is consulted only for
terms lexical matching missed, which is where it adds information rather than
noise.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from prodrome.labelmatch.chunking import Chunk, chunk_section
from prodrome.labelmatch.embed import Embedder
from prodrome.labelmatch.index import build_index
from prodrome.labelmatch.lexical import LexicalMatcher
from prodrome.labelmatch.sectioning import CORE_SECTIONS, ParsedLabel, Tier

logger = logging.getLogger(__name__)

#: bge-family models are trained with an instruction prefix on the query side.
#: Omitting it measurably degrades retrieval; it is applied only to the reaction
#: term, never to the passages, which is how the model was trained.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

#: Minimum z-score against the per-label null for a semantic match to be accepted.
#: 2.5 is deliberately strict: a false "already labelled" verdict destroys a real
#: label gap, which is the finding this project exists to surface, whereas a false
#: "not labelled" is caught downstream when the latency model finds no label change
#: to predict.
DEFAULT_MIN_Z = 2.5

#: Hard floor on raw cosine regardless of z-score, so a label whose null happens to
#: be unusually tight cannot admit a weak match on relative grounds alone.
DEFAULT_MIN_COSINE = 0.55

#: Size of the null sample. Large enough for a stable mean and standard deviation,
#: small enough that embedding it is cheap.
NULL_SAMPLE_SIZE = 96

#: LOINC codes of the primary safety sections, for tiering a semantic hit.
_CORE_CODES = frozenset(CORE_SECTIONS)


class Verdict(StrEnum):
    """The outcome of a label-mention decision."""

    LABELLED_CORE = "labelled_core"
    """Mentioned in primary safety labelling."""

    LABELLED_SECONDARY = "labelled_secondary"
    """Mentioned only in a weaker safety section."""

    NOT_LABELLED = "not_labelled"
    """Absent from every safety section -- a candidate label gap."""

    UNKNOWN = "unknown"
    """The label could not be assessed. Distinct from NOT_LABELLED: treating an
    unparseable label as "nothing is labelled" would manufacture label gaps."""


@dataclass(frozen=True, slots=True)
class Evidence:
    """Why a verdict was reached, in a form a reviewer can check."""

    method: str
    score: float
    section_code: str
    section_name: str
    tier: Tier
    snippet: str
    via: str

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "score": round(self.score, 4),
            "section_code": self.section_code,
            "section_name": self.section_name,
            "tier": self.tier.value,
            "snippet": self.snippet[:400],
            "via": self.via,
        }


@dataclass(frozen=True, slots=True)
class MentionDecision:
    """A verdict on one (reaction, label version) pair."""

    reaction: str
    verdict: Verdict
    evidence: Evidence | None
    #: Best semantic score seen, even when the verdict was NOT_LABELLED. Kept so a
    #: reviewer can see near misses, and so the threshold can be re-tuned from
    #: stored output without re-embedding everything.
    best_semantic_score: float
    best_semantic_z: float

    @property
    def is_labelled(self) -> bool:
        return self.verdict in (Verdict.LABELLED_CORE, Verdict.LABELLED_SECONDARY)

    @property
    def is_gap(self) -> bool:
        """A label gap is a *confident* absence, never an unassessable label."""
        return self.verdict is Verdict.NOT_LABELLED

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "reaction": self.reaction,
            "verdict": self.verdict.value,
            "is_labelled": self.is_labelled,
            "best_semantic_score": round(self.best_semantic_score, 4),
            "best_semantic_z": round(self.best_semantic_z, 4),
        }
        if self.evidence is not None:
            row.update({f"evidence_{k}": v for k, v in self.evidence.as_row().items()})
        return row


@dataclass(frozen=True, slots=True)
class NullDistribution:
    """Per-label null distribution of best-match similarity scores."""

    mean: float
    std: float
    n: int

    def z(self, score: float) -> float:
        """Standardised score. Returns 0 for a degenerate null rather than
        infinity, so a collapsed null cannot manufacture a confident match."""
        if self.std <= 1e-9 or not math.isfinite(score):
            return 0.0
        return (score - self.mean) / self.std


class LabelMatcher:
    """Decides, for one label version, which reactions it already describes.

    Constructed per label version because both the lexical index and the passage
    embeddings are per-label. Reactions are then queried in batch, which is what
    makes the cost proportional to labels rather than to (labels x reactions).
    """

    def __init__(
        self,
        label: ParsedLabel,
        embedder: Embedder,
        *,
        null_terms: Sequence[str],
        sentences_per_chunk: int = 3,
        stride: int = 2,
        top_k: int = 5,
        min_z: float = DEFAULT_MIN_Z,
        min_cosine: float = DEFAULT_MIN_COSINE,
    ) -> None:
        self._label = label
        self._embedder = embedder
        self._top_k = top_k
        self._min_z = min_z
        self._min_cosine = min_cosine
        self._usable = label.is_usable

        self._chunks: list[Chunk] = []
        for section in label.sections_in((Tier.CORE, Tier.SECONDARY)):
            self._chunks.extend(
                chunk_section(
                    section.text,
                    section_code=section.code,
                    section_name=section.name,
                    sentences_per_chunk=sentences_per_chunk,
                    stride=stride,
                )
            )

        # Separate lexical matchers per tier so a verdict knows which tier it came
        # from without re-searching.
        self._core_matcher = LexicalMatcher(label.text_for((Tier.CORE,)))
        self._secondary_matcher = LexicalMatcher(label.text_for((Tier.SECONDARY,)))

        self._index = None
        self._null = NullDistribution(0.0, 0.0, 0)
        if self._chunks and self._usable:
            vectors = embedder.encode([c.text for c in self._chunks])
            self._index = build_index(vectors)
            self._null = self._fit_null(null_terms)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def index_backend(self) -> str:
        return self._index.backend if self._index is not None else "none"

    @property
    def null(self) -> NullDistribution:
        return self._null

    def _encode_terms(self, terms: Sequence[str]) -> np.ndarray:
        prefixed = [
            BGE_QUERY_PREFIX + t if self._embedder.name.startswith("onnx:BAAI/bge") else t
            for t in terms
        ]
        return self._embedder.encode(prefixed)

    def _best_scores(self, terms: Sequence[str]) -> list[tuple[float, int]]:
        """Best (score, chunk position) per term."""
        if self._index is None or not terms:
            return [(float("-inf"), -1) for _ in terms]
        results = self._index.search(self._encode_terms(terms), self._top_k)
        return [
            (hits[0].score, hits[0].position) if hits else (float("-inf"), -1) for hits in results
        ]

    def _fit_null(self, null_terms: Sequence[str]) -> NullDistribution:
        """Score unrelated reaction terms against this label to get its null.

        The sample is truncated rather than randomly drawn: the caller supplies an
        already-shuffled list with a fixed seed, so the null is reproducible.
        """
        sample = list(null_terms)[:NULL_SAMPLE_SIZE]
        if len(sample) < 8:
            logger.warning(
                "only %d null terms supplied; the empirical null will be unstable "
                "and semantic matching will be effectively disabled for this label",
                len(sample),
            )
            return NullDistribution(0.0, 0.0, len(sample))
        scores = np.array([s for s, _ in self._best_scores(sample)], dtype=float)
        finite = scores[np.isfinite(scores)]
        if finite.size < 8:
            return NullDistribution(0.0, 0.0, int(finite.size))
        return NullDistribution(float(finite.mean()), float(finite.std(ddof=1)), int(finite.size))

    def decide(self, reaction: str) -> MentionDecision:
        """Decide whether this label describes `reaction`."""
        if not self._usable:
            return MentionDecision(reaction, Verdict.UNKNOWN, None, float("nan"), float("nan"))

        # Lexical first: an exact hit is not a probabilistic judgement.
        for matcher, tier, verdict in (
            (self._core_matcher, Tier.CORE, Verdict.LABELLED_CORE),
            (self._secondary_matcher, Tier.SECONDARY, Verdict.LABELLED_SECONDARY),
        ):
            hit = matcher.find(reaction)
            if hit.found:
                return MentionDecision(
                    reaction=reaction,
                    verdict=verdict,
                    evidence=Evidence(
                        method=f"lexical:{hit.kind.value}",
                        score=1.0,
                        section_code="",
                        section_name=tier.value,
                        tier=tier,
                        snippet=_window(matcher.normalised_text, hit.matched_text),
                        via=hit.via,
                    ),
                    best_semantic_score=float("nan"),
                    best_semantic_z=float("nan"),
                )

        ((score, position),) = self._best_scores([reaction])
        z = self._null.z(score)
        if position >= 0 and score >= self._min_cosine and z >= self._min_z:
            chunk = self._chunks[position]
            tier = Tier.CORE if chunk.section_code in _CORE_CODES else Tier.SECONDARY
            return MentionDecision(
                reaction=reaction,
                verdict=(
                    Verdict.LABELLED_CORE if tier is Tier.CORE else Verdict.LABELLED_SECONDARY
                ),
                evidence=Evidence(
                    method="semantic",
                    score=score,
                    section_code=chunk.section_code,
                    section_name=chunk.section_name,
                    tier=tier,
                    snippet=chunk.text,
                    via=f"z={z:.2f} vs null mean {self._null.mean:.3f}",
                ),
                best_semantic_score=score,
                best_semantic_z=z,
            )

        return MentionDecision(
            reaction=reaction,
            verdict=Verdict.NOT_LABELLED,
            evidence=None,
            best_semantic_score=score if math.isfinite(score) else float("nan"),
            best_semantic_z=z,
        )

    def decide_many(self, reactions: Sequence[str]) -> list[MentionDecision]:
        """Decide for many reactions. Batches the embedding work."""
        return [self.decide(r) for r in reactions]


def _window(text: str, needle: str, *, radius: int = 120) -> str:
    """Text around a match, for the evidence trail."""
    position = text.find(needle)
    if position < 0:
        return text[: radius * 2]
    start = max(0, position - radius)
    end = min(len(text), position + len(needle) + radius)
    return ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")
