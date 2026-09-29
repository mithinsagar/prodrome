"""Construction of the two configured clients.

Kept separate from the client classes so that tests can build a client with an
arbitrary transport without dragging configuration in, and so the request budget
is applied in exactly one place.
"""

from __future__ import annotations

from prodrome.clients.base import ApiClient, HttpCache, RateLimiter
from prodrome.clients.dailymed import SERVICE_ROOT, DailyMedClient
from prodrome.clients.openfda import EVENT_ENDPOINT, LABEL_ENDPOINT, OpenFdaClient
from prodrome.config import Config, Settings

#: Share of the daily quota a single run may spend, leaving room for a rerun on
#: the same day after a failure. A weekly job that burns the whole allowance
#: cannot be retried until tomorrow, which in practice means a missed week.
RUN_QUOTA_SHARE = 0.6


def _cache(settings: Settings, *, enabled: bool) -> HttpCache:
    return HttpCache(settings.cache_dir, enabled=enabled)


def build_openfda(
    config: Config,
    settings: Settings,
    *,
    endpoint: str = EVENT_ENDPOINT,
    cache_enabled: bool = True,
    request_budget: int | None = None,
) -> tuple[OpenFdaClient, ApiClient]:
    """Build an openFDA client and return it alongside the raw transport.

    The transport is returned too because its :class:`RequestStats` go into the
    run manifest, and the caller owns closing it.
    """
    budget = (
        request_budget
        if request_budget is not None
        else int(settings.openfda_daily_quota * RUN_QUOTA_SHARE)
    )
    transport = ApiClient(
        base_url=endpoint,
        namespace="openfda-label" if endpoint == LABEL_ENDPOINT else "openfda-event",
        cache=_cache(settings, enabled=cache_enabled),
        limiter=RateLimiter(config.http.requests_per_minute),
        timeout_seconds=config.http.timeout_seconds,
        max_retries=config.http.max_retries,
        backoff_base_seconds=config.http.backoff_base_seconds,
        max_sleep_seconds=config.http.max_sleep_seconds,
        user_agent=config.http.user_agent,
        api_key=settings.openfda_api_key,
        request_budget=budget,
    )
    return OpenFdaClient(transport, has_api_key=bool(settings.openfda_api_key)), transport


def build_dailymed(
    config: Config,
    settings: Settings,
    *,
    cache_enabled: bool = True,
    request_budget: int | None = None,
) -> tuple[DailyMedClient, ApiClient]:
    """Build a DailyMed client.

    DailyMed publishes no documented rate limit. prodrome self-imposes one
    anyway, at half the openFDA rate: it is a free public service and a label
    backfill pulls multi-megabyte archives.
    """
    transport = ApiClient(
        base_url=SERVICE_ROOT,
        namespace="dailymed",
        cache=_cache(settings, enabled=cache_enabled),
        limiter=RateLimiter(max(config.http.requests_per_minute // 2, 1)),
        timeout_seconds=max(config.http.timeout_seconds, 120.0),
        max_retries=config.http.max_retries,
        backoff_base_seconds=config.http.backoff_base_seconds,
        max_sleep_seconds=config.http.max_sleep_seconds,
        user_agent=config.http.user_agent,
        request_budget=request_budget,
    )
    return DailyMedClient(transport), transport
