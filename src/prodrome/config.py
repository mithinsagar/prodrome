"""Configuration: environment for secrets, YAML for the analysis plan.

The split is deliberate. Anything that changes what the numbers *mean* -- the
cohort, the window, the thresholds, the embedding model -- lives in version-
controlled YAML so a result can be traced to the configuration that produced it.
Anything that is a credential or a machine-local path lives in the environment
and is never written to the warehouse.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from prodrome.selector import DrugSelector
from prodrome.timeframe import Quarter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "prodrome.yml"


class Settings(BaseSettings):
    """Secrets and local paths, from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="PRODROME_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openfda_api_key: str | None = None
    llm_provider: Literal["none", "gemini", "groq"] = "none"
    llm_api_key: str | None = None
    llm_model: str | None = None
    embed_backend: Literal["hashed", "onnx"] = "hashed"

    cache_dir: Path = REPO_ROOT / "data" / "cache"
    warehouse_path: Path = REPO_ROOT / "data" / "warehouse" / "prodrome.duckdb"
    export_dir: Path = REPO_ROOT / "data" / "exports"

    @property
    def llm_enabled(self) -> bool:
        """True only when a provider *and* a key are present.

        A provider set without a key is a misconfiguration that would otherwise
        surface as an authentication error deep in a weekly job, so the brief
        layer checks this and falls back to the deterministic template instead.
        """
        return self.llm_provider != "none" and bool(self.llm_api_key)

    @property
    def openfda_daily_quota(self) -> int:
        """Requests per day available to us. Drives the budget guard in ingest."""
        return 120_000 if self.openfda_api_key else 1_000


class WindowConfig(BaseModel):
    """The quarter range every point-in-time series is computed over."""

    first_quarter: str
    last_quarter: str | Literal["auto"] = "auto"

    @field_validator("first_quarter")
    @classmethod
    def _valid_quarter(cls, v: str) -> str:
        Quarter.parse(v)
        return v

    @field_validator("last_quarter")
    @classmethod
    def _valid_last(cls, v: str) -> str:
        if v != "auto":
            Quarter.parse(v)
        return v

    def resolve_last(self, data_last_updated: Quarter) -> Quarter:
        """Resolve ``auto`` against the quarter openFDA says it is current to.

        Using the API's own ``meta.last_updated`` rather than today's date keeps
        the final quarter from being a partially-populated stub, which would show
        up as a spurious drop in every series.
        """
        return (
            data_last_updated if self.last_quarter == "auto" else Quarter.parse(self.last_quarter)
        )


class CohortDrug(BaseModel):
    """One tracked drug and the pinned rules for finding its reports and label.

    Every field here exists because a simpler version of it was measured against
    the live API and found to fail silently. See :mod:`prodrome.selector` for the
    coverage data behind ``substance_names`` and ``brand_names``, and
    :mod:`prodrome.ingest.cohort` for why ``spl_set_id`` cannot be discovered at
    runtime.
    """

    unii: str = Field(min_length=10, max_length=10)
    name: str
    spl_set_id: str | None = Field(
        default=None,
        description=(
            "DailyMed SPL set id of the *application holder's* label. Repackager "
            "set ids carry a single version and would make the drug look as though "
            "its label never changed, so this is pinned rather than discovered."
        ),
    )
    substance_names: list[str] = Field(
        default_factory=list,
        description=(
            "Active substance names as reporters code them. Essential, not "
            "optional: openFDA's UNII harmonisation has zero coverage for some "
            "drugs, and for those this is the only way their reports are found."
        ),
    )
    brand_names: list[str] = Field(default_factory=list)
    unii_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Measured share of this drug's reports that openFDA harmonised to a "
            "UNII. Recorded so every figure carries the provenance of its "
            "population; 0.0 means UNII matching alone would have found nothing."
        ),
    )
    approval_date: str | None = None
    notes: str | None = None

    @field_validator("unii")
    @classmethod
    def _upper_alnum(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.isalnum():
            raise ValueError(f"UNII must be 10 alphanumeric characters, got {v!r}")
        return v

    @field_validator("substance_names", "brand_names")
    @classmethod
    def _upper_strip(cls, v: list[str]) -> list[str]:
        """Uppercase, because openFDA's ``.exact`` indexes are case-sensitive.

        A lowercase name in this list matches nothing and reports no error, which
        would look exactly like a drug with no adverse events.
        """
        return [name.strip().upper() for name in v if name and name.strip()]

    @property
    def selector(self) -> DrugSelector:
        """The search clause identifying this drug's reports."""
        return DrugSelector(
            unii=self.unii,
            substance_names=tuple(self.substance_names),
            brand_names=tuple(self.brand_names),
        )

    @property
    def relies_on_name_matching(self) -> bool:
        """True when UNII harmonisation does not cover this drug well.

        Surfaced on the dashboard so a reader can see which drugs' numbers rest on
        reporter-supplied names rather than FDA's substance harmonisation.
        """
        return self.unii_coverage is not None and self.unii_coverage < 0.95


class LabelMatchConfig(BaseModel):
    """Thresholds for deciding whether a reaction is already on a label."""

    embed_backend: Literal["hashed", "onnx"] | None = None
    onnx_model: str = "BAAI/bge-small-en-v1.5"
    #: Cosine similarity above which an embedding match counts as a mention.
    #: Calibrated on the hand-adjudicated set in ``conf/adjudicated.yml``; see
    #: ``prodrome labelmatch calibrate``.
    similarity_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    #: Sentences per chunk when splitting a label section for retrieval.
    chunk_sentences: int = Field(default=3, ge=1, le=20)
    chunk_stride: int = Field(default=2, ge=1, le=20)
    top_k: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def _stride_fits(self) -> LabelMatchConfig:
        if self.chunk_stride > self.chunk_sentences:
            raise ValueError("chunk_stride cannot exceed chunk_sentences (it would skip text)")
        return self


class LatencyConfig(BaseModel):
    """Settings for the survival and lead-time analysis."""

    #: Horizon in quarters used to define the supervised outcome
    #: "labelled within H quarters of signal onset".
    horizon_quarters: int = Field(default=8, ge=1, le=40)
    #: Quarter at which the training/evaluation split is made. Everything at or
    #: after this is held out, so the evaluation is prospective by construction.
    holdout_from_quarter: str = "2023Q1"
    #: Minimum co-occurrence count for a pair to enter the modelling set at all.
    min_case_count: int = Field(default=3, ge=1)

    @field_validator("holdout_from_quarter")
    @classmethod
    def _valid(cls, v: str) -> str:
        Quarter.parse(v)
        return v


class HttpConfig(BaseModel):
    """Client behaviour. Defaults are tuned to stay inside the free quota."""

    #: openFDA allows 240 requests/minute. 120 leaves headroom for a second
    #: concurrent consumer (a local run while CI is going) without a 429 storm.
    requests_per_minute: int = Field(default=120, ge=1, le=240)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=5, ge=0, le=10)
    backoff_base_seconds: float = Field(default=1.5, gt=0)
    #: Cap on any single sleep, including one a Retry-After header asks for. A
    #: server can legitimately ask us to wait an hour; a weekly job must not.
    max_sleep_seconds: float = Field(default=60.0, gt=0)
    user_agent: str = "prodrome/0.1 (+https://github.com/mithinsagar/prodrome)"


class Config(BaseModel):
    """The whole analysis plan, loaded from ``conf/prodrome.yml``."""

    window: WindowConfig
    http: HttpConfig = HttpConfig()
    labelmatch: LabelMatchConfig = LabelMatchConfig()
    latency: LatencyConfig = LatencyConfig()
    #: Reaction terms are discovered per drug, but capped so one enormous drug
    #: cannot consume the whole request budget.
    max_reactions_per_drug: int = Field(default=60, ge=1, le=1000)
    #: Reaction must reach this many reports for the tracked drug, cumulatively,
    #: to be worth spending point-in-time queries on.
    min_reaction_reports: int = Field(default=5, ge=1)
    cohort: list[CohortDrug] = Field(default_factory=list)
    negative_controls: list[tuple[str, str]] = Field(
        default_factory=list,
        description="(UNII, reaction PT) pairs treated as known non-associations.",
    )

    @model_validator(mode="after")
    def _unique_cohort(self) -> Config:
        seen: set[str] = set()
        duplicates = {d.unii for d in self.cohort if d.unii in seen or seen.add(d.unii)}  # type: ignore[func-returns-value]
        if duplicates:
            raise ValueError(f"cohort contains duplicate UNIIs: {sorted(duplicates)}")
        return self

    def drug_by_unii(self, unii: str) -> CohortDrug:
        for drug in self.cohort:
            if drug.unii == unii.upper():
                return drug
        raise KeyError(f"{unii} is not in the configured cohort")


def _expand_includes(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    """Resolve ``include:`` entries so the cohort can live in its own file.

    A 100-drug cohort inline would make the main config unreadable, and the
    cohort is the part most likely to be edited by hand.
    """
    includes = raw.pop("include", [])
    merged: dict[str, Any] = {}
    for relative in includes:
        path = (base / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"included config not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"included config {path} must be a mapping at the top level")
        merged.update(loaded)
    merged.update(raw)
    return merged


def load_config(path: Path | str | None = None) -> Config:
    """Load and validate the analysis plan."""
    resolved = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not resolved.is_file():
        raise FileNotFoundError(f"config not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{resolved} must be a mapping at the top level")
    return Config.model_validate(_expand_includes(raw, resolved.parent))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, read once.

    Cached because reading ``.env`` repeatedly inside a tight ingest loop shows
    up in profiles. Tests that manipulate the environment must call
    ``get_settings.cache_clear()``.
    """
    return Settings()


def settings_from_env(**overrides: object) -> Settings:
    """Build settings without the cache, for tests and explicit overrides."""
    return Settings(**overrides)  # type: ignore[arg-type]


def redact(value: str | None, keep: int = 4) -> str:
    """Render a secret safely for logs and run manifests."""
    if not value:
        return "<unset>"
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep)


def env_flag(name: str, *, default: bool = False) -> bool:
    """Read a boolean environment flag with the usual truthy spellings."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
