"""Embedding backends for semantic label matching.

Two backends, and the reason there are two is reproducibility rather than
convenience.

``hashed``
    A deterministic hashing vectoriser over word and character n-grams. No model
    download, no dependency beyond numpy, identical output on every machine and
    every Python version. This is what CI uses, so the test suite never depends on
    a model registry being reachable and never silently changes behaviour when an
    upstream checkpoint is re-uploaded. It is a real vector space -- character
    n-grams capture morphological similarity well -- but it has no semantics: it
    will not connect "intestinal obstruction" to "blockage of the bowel".

``onnx``
    A local sentence-transformer (BAAI/bge-small-en-v1.5) through fastembed, which
    runs on ONNX Runtime with no PyTorch. This is what the published figures use.
    Roughly 130 MB of weights, downloaded once, then fully offline. Chosen over
    an API embedding service because the pipeline has to be free to run and
    reproducible years from now.

Both are exposed through the same protocol, the backend is recorded in the run
manifest, and the similarity threshold is calibrated per backend -- a threshold
tuned on one is meaningless on the other.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalised row vectors."""

    @property
    def name(self) -> str:
        """Identifier recorded in the run manifest."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch. Returns shape ``(len(texts), dimension)``, L2-normalised."""


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so a dot product is a cosine similarity.

    Zero rows are left as zero rather than divided by zero; they then score 0
    against everything, which is the correct behaviour for an empty passage.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised: np.ndarray = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return normalised


class HashedEmbedder:
    """Deterministic hashing vectoriser over word and character n-grams.

    Uses BLAKE2b rather than Python's built-in ``hash``, which is randomised per
    process by PYTHONHASHSEED -- with the built-in, two runs on the same machine
    would produce different vectors, and the whole point of this backend is that
    they do not.

    Character n-grams do most of the work: they make "thrombocytopenia" and
    "thrombocytopenic" near-identical without any linguistic knowledge, which
    covers the morphological variation the lexical layer's stemmer misses.
    """

    def __init__(self, dimension: int = 512, *, char_ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dimension < 16:
            raise ValueError(f"dimension {dimension} is too small to be useful")
        low, high = char_ngram_range
        if low < 1 or high < low:
            raise ValueError(f"invalid char_ngram_range {char_ngram_range}")
        self._dimension = dimension
        self._char_range = char_ngram_range

    @property
    def name(self) -> str:
        return f"hashed-{self._dimension}d-char{self._char_range[0]}-{self._char_range[1]}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimension

    def _features(self, text: str) -> list[str]:
        lowered = text.lower()
        words = lowered.split()
        features = list(words)
        features.extend(f"{a}_{b}" for a, b in pairwise(words))
        padded = f" {lowered} "
        low, high = self._char_range
        for size in range(low, high + 1):
            features.extend(padded[i : i + size] for i in range(len(padded) - size + 1))
        return features

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                # Signed hashing: a second hash bit decides the sign, which makes
                # collisions cancel on average instead of always accumulating.
                bucket = self._bucket(feature)
                sign = 1.0 if self._bucket(feature + "\x00") % 2 else -1.0
                matrix[row, bucket] += sign
        return _l2_normalise(matrix)


class OnnxEmbedder:
    """Local sentence-transformer embeddings through fastembed.

    Instantiation downloads the model on first use and caches it. Import of
    fastembed is deferred to construction so that the package remains importable,
    and the whole test suite runnable, without the optional extra installed.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", *, cache_dir: str | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised by the extra
            raise ImportError(
                "the 'onnx' embedding backend needs the optional extra: pip install -e '.[embed]'"
            ) from exc
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        probe = next(iter(self._model.embed(["dimension probe"])))
        self._dimension = int(np.asarray(probe).shape[-1])

    @property
    def name(self) -> str:
        return f"onnx:{self._model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = np.asarray(list(self._model.embed(list(texts))), dtype=np.float32)
        return _l2_normalise(vectors)


def build_embedder(backend: str, *, model_name: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    """Construct the requested backend.

    Raises:
        ValueError: on an unknown backend name. Deliberately not falling back to
            `hashed`: a silent downgrade would change what the similarity threshold
            means and quietly invalidate every label-match decision in the run.
    """
    if backend == "hashed":
        return HashedEmbedder()
    if backend == "onnx":
        return OnnxEmbedder(model_name=model_name)
    raise ValueError(f"unknown embedding backend {backend!r} (expected 'hashed' or 'onnx')")
