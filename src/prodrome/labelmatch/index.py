"""Vector search over a label's passages.

Two implementations behind one protocol. FAISS is used when installed, because it
is the right tool once the corpus is every version of every label in the cohort --
tens of thousands of passages, re-queried for hundreds of reaction terms each.
A numpy brute-force index is the fallback, and for a single label version it is
genuinely faster: exact cosine over a few hundred normalised rows is one matrix
multiply, and building a FAISS index costs more than the search saves.

Choosing per corpus size rather than always reaching for the library is the point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

#: Above this many passages, FAISS's indexed search beats a dense matmul enough
#: to be worth the build cost.
FAISS_THRESHOLD = 2_000


@dataclass(frozen=True, slots=True)
class Neighbour:
    """One retrieved passage."""

    position: int
    score: float


class VectorIndex(Protocol):
    """Exact cosine search over L2-normalised vectors."""

    @property
    def backend(self) -> str: ...

    @property
    def size(self) -> int: ...

    def search(self, queries: np.ndarray, k: int) -> list[list[Neighbour]]: ...


class NumpyIndex:
    """Exact cosine search by dense matrix product.

    Vectors are already L2-normalised by the embedders, so an inner product *is*
    the cosine similarity and no renormalisation is needed at query time.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    @property
    def backend(self) -> str:
        return "numpy"

    @property
    def size(self) -> int:
        return int(self._vectors.shape[0])

    def search(self, queries: np.ndarray, k: int) -> list[list[Neighbour]]:
        if self.size == 0:
            return [[] for _ in range(len(queries))]
        scores = np.asarray(queries, dtype=np.float32) @ self._vectors.T
        k = min(k, self.size)
        # argpartition finds the top k without sorting the whole row, then only
        # those k are sorted. On a 400-passage index the saving is small; on the
        # full-cohort corpus it is not.
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        out: list[list[Neighbour]] = []
        for row, positions in enumerate(top):
            ordered = positions[np.argsort(-scores[row, positions])]
            out.append([Neighbour(int(p), float(scores[row, p])) for p in ordered])
        return out


class FaissIndex:
    """FAISS flat inner-product index.

    ``IndexFlatIP`` rather than an approximate index: the corpus is small enough
    that exact search is affordable, and an approximate index would make label-match
    decisions depend on the FAISS build and its random seed -- so two runs could
    disagree about whether a reaction is labelled. Reproducibility outweighs the
    speed here.
    """

    def __init__(self, vectors: np.ndarray) -> None:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ImportError(
                "FaissIndex needs the optional extra: pip install -e '.[embed]'"
            ) from exc
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index = faiss.IndexFlatIP(vectors.shape[1]) if vectors.size else None
        if self._index is not None:
            self._index.add(vectors)
        self._size = int(vectors.shape[0])

    @property
    def backend(self) -> str:
        return "faiss-flat-ip"

    @property
    def size(self) -> int:
        return self._size

    def search(self, queries: np.ndarray, k: int) -> list[list[Neighbour]]:
        if self._index is None or self._size == 0:
            return [[] for _ in range(len(queries))]
        scores, positions = self._index.search(
            np.ascontiguousarray(queries, dtype=np.float32), min(k, self._size)
        )
        return [
            [Neighbour(int(p), float(s)) for p, s in zip(row_p, row_s, strict=True) if p >= 0]
            for row_p, row_s in zip(positions, scores, strict=True)
        ]


def build_index(vectors: np.ndarray, *, prefer_faiss: bool | None = None) -> VectorIndex:
    """Build the appropriate index for this corpus size.

    Args:
        prefer_faiss: force the choice. None selects by corpus size, falling back
            to numpy if FAISS is not installed.
    """
    want_faiss = prefer_faiss if prefer_faiss is not None else vectors.shape[0] >= FAISS_THRESHOLD
    if want_faiss:
        try:
            return FaissIndex(vectors)
        except ImportError:
            logger.info("FAISS not installed; using exact numpy search instead")
    return NumpyIndex(vectors)
