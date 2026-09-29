"""Verifying that a generated brief states only facts from its evidence pack.

Why this is mechanical rather than a prompt instruction
------------------------------------------------------
"Only use the numbers provided" is a request, not a guarantee. The specific risk
with a quantitative summary is not a hallucinated sentence -- that is obvious on
reading -- but a hallucinated *digit*: "a reporting odds ratio of 4.2" where the
evidence says 2.4. It is in the right units, the right magnitude and the right
place, and nothing about the prose signals a problem.

So every number in the generated text is extracted and matched against the closed
set of quantities in the evidence pack. A brief containing an unmatched number is
rejected and the deterministic template is published instead. The model gets to
improve the writing; it does not get to be trusted with the arithmetic.

Matching rules
--------------
A number matches if it is within a relative tolerance of any allowed value, which
lets a writer round 2.413 to "2.4" without failing. Integers below a small bound,
years, and quarter labels are exempt: "3 signals", "two of the eight", "2023Q4" and
ordinary prose numerals are not quantitative claims about the data, and treating
them as such would reject every well-written brief.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Numbers in the text, including decimals, percentages and thousands separators.
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*%?")

#: Quarter labels are identifiers, not measurements.
_QUARTER = re.compile(r"\b(19|20)\d{2}Q[1-4]\b")

#: Four-digit years, likewise.
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

#: Small integers are counting words in prose ("the three drugs", "section 5.7"),
#: not claims about the data. Anything above this must be traceable.
SMALL_INTEGER_BOUND = 20

#: Relative tolerance for a match, so rounding a quoted figure is allowed.
RELATIVE_TOLERANCE = 0.02


@dataclass
class VerificationReport:
    """The outcome of verifying a brief against its evidence."""

    checked: int = 0
    unverified: list[str] = field(default_factory=list)
    exempt: int = 0

    @property
    def passed(self) -> bool:
        return not self.unverified

    def summary(self) -> str:
        if self.passed:
            return (
                f"verified: {self.checked} quantitative claims all trace to the "
                f"evidence pack ({self.exempt} prose numerals exempt)"
            )
        return (
            f"REJECTED: {len(self.unverified)} of {self.checked} numbers do not appear "
            f"in the evidence pack: {', '.join(self.unverified[:8])}"
        )


def _matches_any(value: float, allowed: set[float]) -> bool:
    for candidate in allowed:
        if candidate == 0:
            if abs(value) < 1e-9:
                return True
            continue
        if abs(value - candidate) <= abs(candidate) * RELATIVE_TOLERANCE:
            return True
        # A percentage in the text against a proportion in the evidence, e.g.
        # "38%" for 0.38 -- the same fact in different units.
        if abs(value / 100.0 - candidate) <= abs(candidate) * RELATIVE_TOLERANCE:
            return True
    return False


def verify_numbers(text: str, allowed: set[float]) -> VerificationReport:
    """Check every quantitative claim in `text` against `allowed`.

    Args:
        text: the generated brief.
        allowed: every quantity the evidence pack contains.
    """
    report = VerificationReport()
    masked = _YEAR.sub("", _QUARTER.sub("", text))

    for match in _NUMBER.finditer(masked):
        raw = match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        # Prose numerals: small integers with no decimal point.
        if "." not in raw and value <= SMALL_INTEGER_BOUND:
            report.exempt += 1
            continue
        report.checked += 1
        if not _matches_any(value, allowed):
            report.unverified.append(match.group(0).strip())
    return report


def check_forbidden_claims(text: str) -> list[str]:
    """Flag causal or clinical language a pharmacovigilance brief must not use.

    Disproportionality is a reporting association and nothing more. FAERS has no
    denominator, no control group and no verification of causality, so a brief that
    says a drug "causes" a reaction has overstated the evidence regardless of how
    large the ratio is -- and doing so in a document that looks like a regulatory
    artefact is the most consequential error this project could make.

    The check is on the generated prose only; the deterministic template is written
    not to need it.
    """
    forbidden = {
        r"\bcauses?\b": "asserts causation; disproportionality is an association in reporting",
        r"\bcaused by\b": "asserts causation",
        r"\bproves?\b": "overstates the strength of spontaneous-report evidence",
        r"\bconfirms?\b": "overstates; a signal is a hypothesis, not a confirmation",
        r"\bsafe\b": "makes a safety claim the data cannot support",
        r"\bshould (?:stop|discontinue|avoid)\b": "gives clinical advice",
        r"\brisk of \d": "converts a reporting ratio into an absolute risk",
        r"\bincidence\b": "FAERS has no denominator, so incidence is not estimable",
    }
    found: list[str] = []
    for pattern, reason in forbidden.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(f"{pattern.strip(chr(92) + 'b')}: {reason}")
    return found
