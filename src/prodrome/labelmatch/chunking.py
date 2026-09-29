"""Splitting label safety text into retrievable passages.

Chunk size is a real trade-off here rather than a hyperparameter to shrug at.
A reaction is usually named in a single clause, so a *small* chunk maximises the
cosine similarity between the reaction term and the passage that mentions it --
embedding a 4,000-character section dilutes one mention of ileus into noise. But
a chunk of one sentence loses the context that makes a mention interpretable to a
human reviewing the evidence, and it fragments enumerated adverse-reaction lists
that run across sentence boundaries.

prodrome uses overlapping windows of a few sentences: small enough to keep the
signal concentrated, overlapping so a mention spanning a boundary is never split
away from its context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Sentence boundary: terminal punctuation followed by whitespace and a capital or
#: digit. Label prose is dense with abbreviations ("5.7", "e.g.", "vs.", "mg/dL",
#: "U.S.") and numbered cross-references, so a naive split on "." shatters it.
_SENTENCE_BOUNDARY = re.compile(
    r"""
    (?<![A-Z])          # not after a single capital (initials, "U.S.")
    (?<!\be\.g)(?<!\bi\.e)(?<!\bvs)(?<!\bno)(?<!\bapprox)
    (?<!\bDr)(?<!\bMr)(?<!\bSt)(?<!\bFig)(?<!\bTab)
    (?<!\d)              # not after a digit ("5.7", "0.5 mg")
    [.!?]
    \s+
    (?=[A-Z0-9•])   # before a capital, digit or bullet
    """,
    re.VERBOSE,
)

#: Bullets and enumerations act as boundaries too: adverse-reaction sections are
#: often lists with no terminal punctuation at all.
_BULLET = re.compile(r"\s*[•·▪‣⁃]\s*|\s*\n\s*[-*]\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable passage of a label, with provenance.

    Attributes:
        text: the passage.
        section_code: LOINC code of the section it came from, so a hit can be
            attributed to a tier without a second lookup.
        section_name: human-readable section name for the evidence trail.
        ordinal: position within the section, for stable ordering.
    """

    text: str
    section_code: str
    section_name: str
    ordinal: int

    @property
    def is_substantive(self) -> bool:
        """Whether the passage carries enough text to be worth embedding.

        Very short fragments -- a stray heading, a table cell -- produce unstable
        embeddings that match almost anything, so they are dropped rather than
        indexed.
        """
        return len(self.text) >= MIN_CHUNK_CHARS


#: Below this a passage is a fragment, not a statement.
MIN_CHUNK_CHARS = 40

#: Above this a passage is truncated: a single enormous "sentence" is almost
#: always a table flattened into prose, and embedding it whole is useless.
MAX_CHUNK_CHARS = 1200


def split_sentences(text: str) -> list[str]:
    """Split label prose into sentence-like units."""
    if not text.strip():
        return []
    pieces: list[str] = []
    for bulleted in _BULLET.split(text):
        if not bulleted or not bulleted.strip():
            continue
        pieces.extend(p.strip() for p in _SENTENCE_BOUNDARY.split(bulleted) if p.strip())
    return pieces


def chunk_section(
    text: str,
    *,
    section_code: str,
    section_name: str,
    sentences_per_chunk: int = 3,
    stride: int = 2,
) -> list[Chunk]:
    """Split one section into overlapping sentence windows.

    Args:
        sentences_per_chunk: window size.
        stride: how far the window advances. A stride below the window size gives
            the overlap that keeps a boundary-spanning mention intact.

    Raises:
        ValueError: if `stride` exceeds `sentences_per_chunk`, which would skip
            text entirely -- silently dropping label content, and with it any
            mention that happened to fall in the gap.
    """
    if stride > sentences_per_chunk:
        raise ValueError(
            f"stride {stride} exceeds window {sentences_per_chunk}: text would be skipped"
        )
    if stride < 1 or sentences_per_chunk < 1:
        raise ValueError("stride and window must both be at least 1")

    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    ordinal = 0
    for start in range(0, max(len(sentences), 1), stride):
        window = sentences[start : start + sentences_per_chunk]
        if not window:
            break
        joined = " ".join(window)[:MAX_CHUNK_CHARS]
        chunk = Chunk(
            text=joined,
            section_code=section_code,
            section_name=section_name,
            ordinal=ordinal,
        )
        if chunk.is_substantive:
            chunks.append(chunk)
            ordinal += 1
        if start + sentences_per_chunk >= len(sentences):
            break
    return chunks
