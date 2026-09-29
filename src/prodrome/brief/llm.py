"""Provider-agnostic text generation for the weekly brief.

Two free-tier providers, both optional. The brief is a presentation layer: the
analysis is complete and correct without it, and the deterministic template is
always available. Nothing in the pipeline's findings depends on a model being
reachable, which is why this module is allowed to fail softly where the rest of the
codebase is not.

Requests are made with the standard library rather than an SDK. The payloads are
two small JSON bodies, an SDK per provider would add dependencies that the optional
feature does not justify, and both providers' REST shapes are stable.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

#: Deliberately low: the brief must be a faithful restatement of the evidence, not
#: creative writing, and sampling variety is a liability when the content is facts.
TEMPERATURE = 0.2

#: A weekly brief covering eight gaps has no reason to exceed this.
MAX_OUTPUT_TOKENS = 1400

REQUEST_TIMEOUT_SECONDS = 90


class LlmError(RuntimeError):
    """Generation failed. Always recoverable: the caller falls back to the template."""


class TextGenerator(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


#: Only these hosts are ever contacted. Checked rather than assumed, so a
#: mistyped or injected provider URL cannot turn into a request to an arbitrary
#: scheme (file:, ftp:) or host.
_ALLOWED_HOSTS = frozenset({"generativelanguage.googleapis.com", "api.groq.com"})


def _post_json(
    url: str, payload: Mapping[str, object], headers: Mapping[str, str]
) -> dict[str, object]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise LlmError(f"refusing to call a non-allowlisted endpoint: {url}")
    # Scheme and host are allowlisted immediately above.
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        # Scheme and host are allowlisted above.
        # Scheme and host are allowlisted at the top of this function.
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            loaded = json.load(response)
            if not isinstance(loaded, dict):
                raise LlmError(f"expected a JSON object from {url}, got {type(loaded).__name__}")
            return loaded
    except urllib.error.HTTPError as exc:
        body = exc.read()[:300].decode("utf-8", errors="replace")
        raise LlmError(f"HTTP {exc.code} from {url}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LlmError(f"request to {url} failed: {exc}") from exc


def _dig(payload: object, *path: str | int) -> object:
    """Walk a nested JSON structure, raising LlmError on any shape mismatch.

    Provider responses are untrusted input in the shape sense: a schema change or an
    error envelope returned with a 200 would otherwise surface as a TypeError from
    inside an index chain, with no indication of which provider or which field.
    """
    current = payload
    for step in path:
        if isinstance(step, int):
            if not isinstance(current, list) or len(current) <= step:
                raise LlmError(f"expected a list of at least {step + 1} at {path}")
            current = current[step]
        else:
            if not isinstance(current, dict) or step not in current:
                raise LlmError(f"expected key {step!r} at {path}")
            current = current[step]
    return current


@dataclass(frozen=True, slots=True)
class GeminiGenerator:
    """Google AI Studio (Gemini) free tier."""

    api_key: str
    model: str = "gemini-2.0-flash"

    @property
    def name(self) -> str:
        return f"gemini:{self.model}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
        body = _post_json(url, payload, {"x-goog-api-key": self.api_key})
        parts = _dig(body, "candidates", 0, "content", "parts")
        if not isinstance(parts, list):
            raise LlmError(f"unexpected Gemini response shape: {str(body)[:200]}")
        return "".join(
            str(part.get("text", "")) for part in parts if isinstance(part, dict)
        ).strip()


@dataclass(frozen=True, slots=True)
class GroqGenerator:
    """Groq free tier, OpenAI-compatible chat completions."""

    api_key: str
    model: str = "llama-3.3-70b-versatile"

    @property
    def name(self) -> str:
        return f"groq:{self.model}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            payload,
            {"Authorization": f"Bearer {self.api_key}"},
        )
        return str(_dig(body, "choices", 0, "message", "content")).strip()


def build_generator(provider: str, api_key: str | None, model: str | None) -> TextGenerator | None:
    """Construct a generator, or None when generation is disabled.

    Returns None rather than raising for ``provider == "none"`` or a missing key:
    a brief without a model is the documented default, not an error state.
    """
    if provider == "none" or not api_key:
        return None
    if provider == "gemini":
        return GeminiGenerator(api_key=api_key, model=model or "gemini-2.0-flash")
    if provider == "groq":
        return GroqGenerator(api_key=api_key, model=model or "llama-3.3-70b-versatile")
    logger.warning("unknown LLM provider %r; the brief will use the template", provider)
    return None
