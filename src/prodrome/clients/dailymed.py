"""DailyMed client: the label *history* that makes this project possible.

Why this source exists in the design
------------------------------------
openFDA's label endpoint indexes only the current version of each label. That is
sufficient to answer "is this reaction on the label today", which is what every
existing label-comparison tool does -- and it is the wrong question. Whether a
reaction is on the label today tells you nothing about *when* it got there, so it
cannot distinguish a reaction FDA has warned about for a decade from one added
last quarter in response to the very reports you are analysing.

DailyMed, run by the National Library of Medicine, publishes the full version
history of every Structured Product Label: a version number and publication date
per revision, and the complete document for each. That turns "is it labelled"
into "as of which date was it labelled", which is the outcome variable this
entire project is built to predict.

Verified against the live service while designing this: Ozempic's application-
holder label has 19 archived versions running from 2017-12-06 to 2026-06-10, and
the ileus warning FDA added in 2023 is absent from version 13 (2022-10-07) and
present in version 14 (2023-10-09). That transition is the ground truth the
latency model is trained against.

Two operational hazards, both handled here
------------------------------------------
*Repackager labels.* Searching by drug name returns many set ids. Most belong to
repackagers and relabellers, which republish a snapshot once and never revise it
-- Ozempic has several such set ids with exactly one version each. Using one
would make a drug look as though its label never changed. The application
holder's set id is therefore pinned in configuration, and
:meth:`DailyMedClient.rank_candidate_set_ids` exists to help a human choose it
rather than to guess at runtime.

*Success-shaped failures.* Asking for a version that does not exist returns HTTP
200 with an HTML error page. Only a content check distinguishes that from a real
archive, which is why every archive fetch asserts the ZIP magic bytes.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from prodrome.clients.base import ApiClient

logger = logging.getLogger(__name__)

SERVICE_ROOT = "https://dailymed.nlm.nih.gov/dailymed"
ZIP_MAGIC = b"PK"

#: DailyMed renders history dates as "Oct 09, 2023".
_HISTORY_DATE = "%b %d, %Y"

#: Repackager and relabeller names, matched against the labeller in an SPL title.
#: Used only to *rank* candidates for a human, never to silently exclude one.
_REPACKAGER_HINTS = re.compile(
    r"\b(A-S MEDICATION|AIDAREX|PROFICIENT RX|QUALITY CARE|NUCARE|REDPHARM|"
    r"BRYANT RANCH|DIRECT[_ ]RX|PD-RX|RPK PHARMACEUTICALS|ASCLEMED|"
    r"PREFERRED PHARMACEUTICALS|DENTON PHARMA|NORTHWIND|LAKE ERIE MEDICAL|"
    r"MEDSOURCE|CARDINAL HEALTH|MCKESSON|HF ACQUISITION|HENRY SCHEIN)\b",
    re.IGNORECASE,
)


class DailyMedError(RuntimeError):
    """A DailyMed response could not be interpreted."""


@dataclass(frozen=True, slots=True)
class SplVersion:
    """One archived revision of a label."""

    version: int
    published: dt.date

    def __lt__(self, other: SplVersion) -> bool:
        return self.version < other.version


@dataclass(frozen=True, slots=True)
class SplCandidate:
    """A set id found by search, with the evidence for choosing it."""

    set_id: str
    title: str
    version_count: int
    latest_published: dt.date | None

    @property
    def looks_like_repackager(self) -> bool:
        return bool(_REPACKAGER_HINTS.search(self.title))

    @property
    def rank_key(self) -> tuple[int, int]:
        """Sort key: application holders first, then by revision count.

        A label with many revisions is almost certainly the real one, because
        only the application holder revises a label.
        """
        return (0 if self.looks_like_repackager else 1, self.version_count)


@dataclass(frozen=True, slots=True)
class LabelDocument:
    """An archived label version and its extracted XML."""

    set_id: str
    version: int
    published: dt.date
    xml: str
    #: ``effectiveTime`` declared inside the document. Usually a few days before
    #: the DailyMed publication date; prodrome prefers it when present because it
    #: is the date the labeller asserts the content took effect.
    effective_date: dt.date | None

    @property
    def authoritative_date(self) -> dt.date:
        """The date to treat this version as taking effect."""
        return self.effective_date or self.published


class DailyMedClient:
    """Version history and archived documents for SPL set ids."""

    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def version_history(self, set_id: str) -> list[SplVersion]:
        """All archived revisions of a label, oldest first.

        Note that version numbers are not always contiguous -- Ozempic's history
        skips version 10 -- so callers must iterate the returned list rather than
        ``range(1, n + 1)``.
        """
        payload = self._client.get_json(f"services/v2/spls/{set_id}/history.json")
        if payload is None:
            return []
        try:
            history = payload["data"]["history"]
        except (KeyError, TypeError) as exc:
            raise DailyMedError(f"unexpected history envelope for set id {set_id}") from exc

        versions: list[SplVersion] = []
        for row in history:
            try:
                versions.append(
                    SplVersion(
                        version=int(row["spl_version"]),
                        published=dt.datetime.strptime(
                            str(row["published_date"]).strip(), _HISTORY_DATE
                        ).date(),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping malformed history row for %s: %r", set_id, row)
        return sorted(versions)

    def title(self, set_id: str) -> str | None:
        payload = self._client.get_json(f"services/v2/spls/{set_id}/history.json")
        if payload is None:
            return None
        return str(payload.get("data", {}).get("spl", {}).get("title", "")) or None

    def search_set_ids(self, drug_name: str, *, page_size: int = 50) -> list[SplCandidate]:
        """Find candidate set ids for a drug name, ranked for human review.

        Deliberately not automatic: choosing the wrong set id silently converts a
        drug with a rich revision history into one that appears never to have been
        revised, and no downstream check would catch it.
        """
        payload = self._client.get_json(
            "services/v2/spls.json", {"drug_name": drug_name, "pagesize": page_size}
        )
        if payload is None:
            return []
        candidates: list[SplCandidate] = []
        for row in payload.get("data") or []:
            set_id = str(row.get("setid", ""))
            if not set_id:
                continue
            published: dt.date | None
            try:
                published = dt.datetime.strptime(
                    str(row.get("published_date", "")).strip(), _HISTORY_DATE
                ).date()
            except ValueError:
                published = None
            candidates.append(
                SplCandidate(
                    set_id=set_id,
                    title=str(row.get("title", "")),
                    version_count=int(row.get("spl_version") or 0),
                    latest_published=published,
                )
            )
        return candidates

    def rank_candidate_set_ids(self, drug_name: str) -> list[SplCandidate]:
        """Candidates ordered with the most plausible application holder first."""
        return sorted(self.search_set_ids(drug_name), key=lambda c: c.rank_key, reverse=True)

    def fetch_version(self, set_id: str, version: int) -> LabelDocument | None:
        """Download and unpack one archived label version.

        Returns None when the version is not retrievable, which happens for
        versions listed in history but withdrawn from the archive. A missing
        version is a gap to be recorded, not a run-ending error: the label
        timeline is still usable, just coarser around that date.
        """
        body = self._client.get_bytes(
            "getFile.cfm",
            {"setid": set_id, "type": "zip", "version": version},
            suffix=".zip",
            expect_prefix=ZIP_MAGIC,
        )
        if body is None:
            logger.info("archive for set id %s version %s is not retrievable", set_id, version)
            return None

        try:
            xml = _extract_spl_xml(body)
        except DailyMedError as exc:
            logger.warning("archive for %s v%s is unusable: %s", set_id, version, exc)
            return None

        published = self._published_date(set_id, version)
        return LabelDocument(
            set_id=set_id,
            version=version,
            published=published or dt.date.min,
            xml=xml,
            effective_date=_effective_time(xml),
        )

    def _published_date(self, set_id: str, version: int) -> dt.date | None:
        for entry in self.version_history(set_id):
            if entry.version == version:
                return entry.published
        return None

    def iter_label_timeline(self, set_id: str) -> list[LabelDocument]:
        """Every retrievable version of a label, oldest first.

        This is the full backfill for one drug. Versions that cannot be retrieved
        are skipped with a log line rather than aborting the drug.
        """
        documents: list[LabelDocument] = []
        for entry in self.version_history(set_id):
            document = self.fetch_version(set_id, entry.version)
            if document is not None:
                documents.append(document)
        return documents


def _extract_spl_xml(archive: bytes) -> str:
    """Pull the single SPL document out of a DailyMed archive.

    An archive contains the label XML plus any images it references. Exactly one
    ``.xml`` member is expected; anything else means the archive layout changed
    and the caller should not guess.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            names = [n for n in bundle.namelist() if n.lower().endswith(".xml")]
            if len(names) != 1:
                raise DailyMedError(f"expected exactly one XML member, found {len(names)}")
            return bundle.read(names[0]).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        raise DailyMedError(f"not a readable ZIP archive: {exc}") from exc


_EFFECTIVE_TIME = re.compile(r'<effectiveTime\s+value="(\d{8})')


def _effective_time(xml: str) -> dt.date | None:
    """First ``effectiveTime`` in the document, which is the label's own date.

    Parsed with a regex rather than by walking the tree because this runs on
    every version of every drug and only needs the first few kilobytes; a full
    parse here would triple the backfill time for no gain.
    """
    match = _EFFECTIVE_TIME.search(xml)
    if match is None:
        return None
    try:
        return dt.datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


#: FDA's UNII code system OID, as it appears in SPL ingredient blocks.
UNII_CODE_SYSTEM = "2.16.840.1.113883.4.9"

#: SPL ingredient class codes that denote an *active* ingredient. Everything else
#: (IACT) is an excipient: Ozempic's document carries UNIIs for disodium
#: phosphate, propylene glycol, phenol and water alongside semaglutide's, and
#: keying a cohort on one of those would be a silent, total failure.
_ACTIVE_INGREDIENT_CLASSES = frozenset({"ACTIB", "ACTIM", "ACTIR"})

_INGREDIENT_TAG = f"{{{'urn:hl7-org:v3'}}}ingredient"
_SUBSTANCE_TAG = f"{{{'urn:hl7-org:v3'}}}ingredientSubstance"
_CODE_TAG_V3 = f"{{{'urn:hl7-org:v3'}}}code"
_NAME_TAG = f"{{{'urn:hl7-org:v3'}}}name"


def active_ingredient_uniis(xml: str) -> dict[str, str]:
    """Map UNII -> substance name for the *active* ingredients of a label.

    This is what lets DailyMed serve as the primary identity source. openFDA's
    label index is incomplete -- osimertinib, esketamine and ubrogepant are all
    absent from it under both brand and generic name -- so resolving identity
    through it silently drops real drugs from a cohort. The SPL document itself
    always carries the UNII, because FDA requires it.

    Returns an empty mapping when the document has no parseable ingredient block,
    which the caller must treat as a failure to resolve rather than as a drug with
    no active ingredient.
    """
    try:
        # See the note in labelmatch/sectioning.py on the XML threat model.
        root = ET.fromstring(xml.encode("utf-8", errors="replace"))  # noqa: S314
    except ET.ParseError as exc:
        logger.warning("could not parse SPL for ingredient extraction: %s", exc)
        return {}

    found: dict[str, str] = {}
    for ingredient in root.iter(_INGREDIENT_TAG):
        if (ingredient.get("classCode") or "") not in _ACTIVE_INGREDIENT_CLASSES:
            continue
        substance = ingredient.find(_SUBSTANCE_TAG)
        if substance is None:
            continue
        code = substance.find(_CODE_TAG_V3)
        if code is None or code.get("codeSystem") != UNII_CODE_SYSTEM:
            continue
        unii = (code.get("code") or "").strip().upper()
        if len(unii) != 10 or not unii.isalnum():
            continue
        name_element = substance.find(_NAME_TAG)
        name = (name_element.text or "").strip() if name_element is not None else unii
        found.setdefault(unii, name)
    return found


_LABELLER_IN_TITLE = re.compile(r"\[([^\]]+)\]\s*$")


def labeller_from_title(title: str) -> str:
    """Extract the labeller from a DailyMed title.

    Titles end with the labeller in square brackets:
    ``"OZEMPIC (SEMAGLUTIDE) INJECTION, SOLUTION [NOVO NORDISK ...]"``.
    """
    match = _LABELLER_IN_TITLE.search(title.strip())
    return match.group(1).strip() if match else ""
