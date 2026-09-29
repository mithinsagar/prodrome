"""The weekly brief: a narrative summary that cannot invent numbers."""

from prodrome.brief.evidence import EvidencePack, GapEvidence, build_evidence
from prodrome.brief.verify import VerificationReport, verify_numbers

__all__ = [
    "EvidencePack",
    "GapEvidence",
    "VerificationReport",
    "build_evidence",
    "verify_numbers",
]
