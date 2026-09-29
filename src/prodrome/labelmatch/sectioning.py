"""Parsing an SPL document into safety sections, by LOINC code.

Which sections count as "the label says this" is the single most consequential
judgement in this project, because it defines the outcome variable. Getting it
wrong does not produce an error -- it produces a confidently wrong latency.

The problem with "search the whole label"
-----------------------------------------
A US prescribing information document is mostly *not* safety labelling. Ozempic's
2023 label devotes 18,761 characters to CLINICAL STUDIES, which names dozens of
adverse events as trial outcomes -- including events FDA has never warned about.
Matching against it would mark almost every reaction as already labelled, and the
label-gap analysis would return nothing.

Two subtler traps:

*Indications look like adverse events.* "Weight decreased" is an adverse event in
FAERS and a therapeutic goal for semaglutide. A whole-document match cannot tell
those apart; excluding INDICATIONS & USAGE can.

*Animal findings are not human warnings.* NONCLINICAL TOXICOLOGY and
CARCINOGENESIS describe rodent results. Treating "thyroid C-cell tumours in mice"
as a labelled human adverse reaction would be wrong in exactly the direction that
flatters the pipeline.

Tiers instead of a boolean
--------------------------
"Labelled" is not binary in regulatory practice -- a boxed warning and a passing
mention under drug interactions are not the same claim. prodrome therefore sorts
sections into tiers and records *which tier* matched, so a reviewer can apply
their own bar. The headline gap metric uses :data:`CORE_SECTIONS` only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

SPL_NAMESPACE = "urn:hl7-org:v3"
_SECTION_TAG = f"{{{SPL_NAMESPACE}}}section"
_CODE_TAG = f"{{{SPL_NAMESPACE}}}code"
_TITLE_TAG = f"{{{SPL_NAMESPACE}}}title"


class Tier(StrEnum):
    """How strong a claim a mention in this section represents."""

    CORE = "core"
    """Primary safety labelling. A mention here is unambiguously "FDA-warned"."""

    SECONDARY = "secondary"
    """Safety-relevant, but a mention is a weaker claim -- a population-specific
    caveat or an interaction rather than a recognised adverse reaction."""

    EXCLUDED = "excluded"
    """Present in the document but not evidence that a reaction is labelled."""


#: Primary safety labelling, in descending order of regulatory force.
CORE_SECTIONS: dict[str, str] = {
    "34066-1": "Boxed warning",
    "34070-3": "Contraindications",
    "43685-7": "Warnings and precautions",
    "34084-4": "Adverse reactions",
}

#: Safety-relevant but weaker. Reported separately so a reviewer can decide
#: whether a pregnancy-only caveat counts as the reaction being labelled.
SECONDARY_SECTIONS: dict[str, str] = {
    "34073-7": "Drug interactions",
    "43684-0": "Use in specific populations",
    "42228-7": "Pregnancy",
    "77290-5": "Lactation",
    "77291-3": "Females and males of reproductive potential",
    "34081-0": "Pediatric use",
    "34082-8": "Geriatric use",
    "34088-5": "Overdosage",
    "42231-1": "Medication guide",
    "34076-0": "Information for patients",
}

#: Explicitly excluded, each with the reason, because "we ignored this section"
#: is a claim that needs defending rather than a detail.
EXCLUDED_SECTIONS: dict[str, str] = {
    "34092-7": "Clinical studies -- names adverse events as trial outcomes, not warnings",
    "34090-1": "Clinical pharmacology -- mechanistic, not a safety claim",
    "43679-0": "Mechanism of action -- mechanistic",
    "43681-6": "Pharmacodynamics -- mechanistic",
    "43682-4": "Pharmacokinetics -- mechanistic",
    "43680-8": "Nonclinical toxicology -- animal findings, not human warnings",
    "34083-6": "Carcinogenesis and mutagenesis -- animal findings",
    "34067-9": "Indications and usage -- therapeutic goals collide with adverse events",
    "34068-7": "Dosage and administration -- procedural",
    "43678-2": "Dosage forms and strengths -- procedural",
    "34089-3": "Description -- chemistry",
    "34069-5": "How supplied -- logistics",
    "59845-8": "Instructions for use -- device handling",
    "51945-4": "Package display panel -- carton artwork text",
    "48780-1": "SPL product data elements -- ingredient list",
    "42229-5": "SPL unclassified section -- heterogeneous, no consistent meaning",
}


def tier_for(code: str) -> Tier:
    """Tier a LOINC section code falls into.

    An unrecognised code is treated as EXCLUDED, which is the conservative
    direction: a new section type cannot silently start counting as evidence that
    a reaction is labelled. Unrecognised codes are logged so the map can be
    extended deliberately.
    """
    if code in CORE_SECTIONS:
        return Tier.CORE
    if code in SECONDARY_SECTIONS:
        return Tier.SECONDARY
    if code not in EXCLUDED_SECTIONS:
        logger.debug("unmapped SPL section code %s; treating as excluded", code)
    return Tier.EXCLUDED


def section_name(code: str) -> str:
    return CORE_SECTIONS.get(code) or SECONDARY_SECTIONS.get(code) or code


@dataclass(frozen=True, slots=True)
class LabelSection:
    """One coded section's own text, excluding that of nested subsections."""

    code: str
    name: str
    tier: Tier
    title: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class ParsedLabel:
    """A label document reduced to the sections that matter."""

    sections: tuple[LabelSection, ...]

    def text_for(self, tiers: tuple[Tier, ...] = (Tier.CORE,)) -> str:
        """Concatenated text of every section in the given tiers."""
        return "\n\n".join(s.text for s in self.sections if s.tier in tiers and s.text)

    def sections_in(self, tiers: tuple[Tier, ...]) -> tuple[LabelSection, ...]:
        return tuple(s for s in self.sections if s.tier in tiers)

    @property
    def core_char_count(self) -> int:
        return sum(s.char_count for s in self.sections if s.tier is Tier.CORE)

    @property
    def is_usable(self) -> bool:
        """Whether this label has enough safety text to match against.

        A label with no core safety sections is not evidence of absence -- it is a
        parse failure or an unusual document type (a kit, a bulk ingredient). The
        label-match layer must record it as unknown rather than as "not labelled",
        or every such drug becomes a spurious label gap.
        """
        return self.core_char_count >= MIN_CORE_CHARS


#: Below this, the document almost certainly is not a full prescribing
#: information -- real core safety sections run to thousands of characters.
MIN_CORE_CHARS = 200

_WHITESPACE = re.compile(r"[ \t\r\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _own_text(section: ET.Element) -> str:
    """Text belonging to this section but not to any nested subsection.

    SPL nests sections -- "5.7 Severe Gastrointestinal Adverse Reactions" sits
    inside "5 WARNINGS AND PRECAUTIONS". Taking a parent's full subtree text would
    attribute a child's content to the parent as well, double-counting it and
    making tier attribution meaningless. Walking only the non-section descendants
    gives each piece of text exactly one owner; a parent's full text is then
    recoverable by rolling up its children.
    """
    parts: list[str] = []

    def walk(element: ET.Element, *, is_root: bool) -> None:
        if not is_root and element.tag == _SECTION_TAG:
            return  # belongs to the nested section, not this one
        if element.tag != _TITLE_TAG and element.text:
            parts.append(element.text)
        for child in element:
            walk(child, is_root=False)
            if child.tail:
                parts.append(child.tail)

    walk(section, is_root=True)
    joined = " ".join(p.strip() for p in parts if p and p.strip())
    return _BLANK_LINES.sub("\n\n", _WHITESPACE.sub(" ", joined)).strip()


def parse_label(xml: str) -> ParsedLabel:
    """Parse an SPL document into coded sections.

    Args:
        xml: the full SPL document text.

    Returns:
        The parsed label. A document that fails to parse yields an empty
        :class:`ParsedLabel`, which ``is_usable`` reports as unusable -- the
        caller must treat that as unknown rather than as "nothing is labelled".
    """
    try:
        # SPL documents are well-formed XML from a regulated submission pipeline,
        # but archived ones occasionally carry encoding damage. Parsing from bytes
        # lets ElementTree honour the declared encoding.
        # Parsed with the standard library rather than defusedxml. The threat model
        # is narrow: documents come from NLM's DailyMed over HTTPS, and Python's
        # ElementTree does not resolve external entities or DTDs, so XXE and
        # external-reference attacks do not apply. The residual risk is
        # entity-expansion denial of service on a document already downloaded,
        # which is bounded by the archive size guard in DailyMedClient.
        root = ET.fromstring(xml.encode("utf-8", errors="replace"))  # noqa: S314
    except ET.ParseError as exc:
        logger.warning("SPL document did not parse: %s", exc)
        return ParsedLabel(())

    sections: list[LabelSection] = []
    for element in root.iter(_SECTION_TAG):
        code_element = element.find(_CODE_TAG)
        if code_element is None:
            continue
        code = code_element.get("code") or ""
        if not code:
            continue
        tier = tier_for(code)
        if tier is Tier.EXCLUDED:
            continue
        title_element = element.find(_TITLE_TAG)
        title = " ".join(title_element.itertext() if title_element is not None else []).strip()
        text = _own_text(element)
        if not text:
            continue
        sections.append(
            LabelSection(
                code=code,
                name=section_name(code),
                tier=tier,
                title=_WHITESPACE.sub(" ", title),
                text=text,
            )
        )
    return ParsedLabel(tuple(sections))
