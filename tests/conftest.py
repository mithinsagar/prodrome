"""Shared fixtures."""

from __future__ import annotations

import pytest

from prodrome.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    """Settings are process-cached; a test that changes the environment must not
    leak that into the next one."""
    get_settings.cache_clear()
