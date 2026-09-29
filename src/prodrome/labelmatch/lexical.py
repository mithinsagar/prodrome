"""Lexical matching between MedDRA reaction terms and US label prose.

This module exists because the naive version of this comparison -- embed the
reaction term, embed the label, threshold the cosine similarity -- fails on a
large and *systematic* class of cases that have nothing to do with semantics.

Three distinct gaps
-------------------
**Orthography.** MedDRA is maintained to British spelling conventions; US
prescribing information is written in American English. So the FAERS term is
"Diarrhoea" and the label says "diarrhea"; "Oesophagitis" against
"esophagitis"; "Ischaemic stroke" against "ischemic stroke". These are the *same
word*. Any pipeline that misses them systematically under-reports how much of a
label it matched, which inflates the apparent label gap -- in the direction that
makes the project's headline look better. That is the worst kind of bug.

**Register.** MedDRA prefers the Latinate clinical term where a label often uses
the colloquial one: "Pyrexia" against "fever", "Pruritus" against "itching",
"Dyspnoea" against "shortness of breath", "Asthenia" against "weakness". These
are genuine synonyms a clinician treats as interchangeable, and an embedding model
usually -- but not reliably -- scores them as similar.

**Morphology.** "Seizures" against "seizure"; "Hepatic failure" against "hepatic
failure occurred"; adjectival forms like "thrombocytopenic" for
"Thrombocytopenia".

Why rules and not just a bigger model
-------------------------------------
These three gaps are *deterministic*. A rule that maps "oe" to "e" inside
``oesophag`` is right every time, costs nothing, and is auditable by a reviewer
who does not trust neural similarity. Spending model capacity on them would be
both slower and less reliable. The embedding layer is reserved for the genuinely
semantic residue -- "Intestinal obstruction" against "blockage of the bowel" --
which is what it is actually good at.

A note on MedDRA licensing
--------------------------
MedDRA's own synonym and hierarchy files are licensed and cannot be redistributed,
so :data:`CLINICAL_SYNONYMS` is a hand-built table, not a MedDRA extract. It is
consequently incomplete, which is a stated limitation rather than a hidden one:
the calibration report in ``docs/METHODS.md`` measures what it costs. Reaction
*term strings* themselves come from FAERS records, which are public.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

#: British-to-American normalisations, applied to lowercased text.
#:
#: Deliberately a curated stem list rather than a blanket ``oe -> e`` rule: the
#: general rule mangles ordinary English ("toe", "shoe", "does") and would corrupt
#: the label text it is meant to be searching. Each entry is a medical stem where
#: the substitution is unambiguous.
SPELLING_NORMALISATIONS: tuple[tuple[str, str], ...] = (
    # oe -> e
    ("oesophag", "esophag"),
    ("oedema", "edema"),
    ("oedemat", "edemat"),
    ("diarrhoe", "diarrhe"),
    ("gonorrhoe", "gonorrhe"),
    ("dyspnoe", "dyspne"),
    ("apnoe", "apne"),
    ("tachypnoe", "tachypne"),
    ("hypopnoe", "hypopne"),
    ("orthopnoe", "orthopne"),
    ("amenorrhoe", "amenorrhe"),
    ("menorrhoe", "menorrhe"),
    ("rhinorrhoe", "rhinorrhe"),
    ("seborrhoe", "seborrhe"),
    ("coeliac", "celiac"),
    ("amoeb", "ameb"),
    ("foet", "fet"),
    ("oestr", "estr"),
    ("oesophageal", "esophageal"),
    # ae -> e
    ("anaemi", "anemi"),
    ("haem", "hem"),
    ("ischaemi", "ischemi"),
    ("leukaemi", "leukemi"),
    ("septicaemi", "septicemi"),
    ("bacteraemi", "bacteremi"),
    ("viraemi", "viremi"),
    ("uraemi", "uremi"),
    ("hypoxaemi", "hypoxemi"),
    ("toxaemi", "toxemi"),
    ("azotaemi", "azotemi"),
    ("acidaemi", "acidemi"),
    ("alkalaemi", "alkalemi"),
    ("lipaemi", "lipemi"),
    ("glycaemi", "glycemi"),
    ("kalaemi", "kalemi"),
    ("natraemi", "natremi"),
    ("calcaemi", "calcemi"),
    ("magnesaemi", "magnesemi"),
    ("phosphataemi", "phosphatemi"),
    ("uricaemi", "uricemi"),
    ("cholesterolaemi", "cholesterolemi"),
    ("proteinaemi", "proteinemi"),
    ("bilirubinaemi", "bilirubinemi"),
    ("paediatr", "pediatr"),
    ("anaesth", "anesth"),
    ("orthopaedi", "orthopedi"),
    ("gynaecolog", "gynecolog"),
    ("caec", "cec"),
    ("aetiolog", "etiolog"),
    ("oligaemi", "oligemi"),
    ("hypovolaemi", "hypovolemi"),
    ("hypervolaemi", "hypervolemi"),
    # leuc -> leuk
    ("leucocy", "leukocy"),
    ("leucopeni", "leukopeni"),
    ("leucocytosis", "leukocytosis"),
    # misc
    ("tumour", "tumor"),
    ("behaviour", "behavior"),
    ("sulph", "sulf"),
    ("oedipal", "edipal"),
    ("catheterisation", "catheterization"),
    ("hospitalisation", "hospitalization"),
    ("visualisation", "visualization"),
    ("normalisation", "normalization"),
    ("localisation", "localization"),
)

#: Clinical synonyms: canonical MedDRA-style term -> phrasings a label may use.
#:
#: Curated by hand, because MedDRA's own synonym files are licensed. Entries are
#: restricted to pairs a clinician would treat as interchangeable in a safety
#: context -- not merely related. "Hypertension" and "blood pressure increased"
#: are here; "hypertension" and "cardiovascular risk" are not.
CLINICAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "pyrexia": ("fever", "febrile", "elevated temperature"),
    "pruritus": ("itching", "itch", "itchiness"),
    "dyspnea": ("shortness of breath", "breathlessness", "difficulty breathing"),
    "asthenia": ("weakness", "lack of energy", "loss of strength"),
    "malaise": ("feeling unwell", "general discomfort"),
    "somnolence": ("drowsiness", "sleepiness", "sedation"),
    "hypoesthesia": ("numbness", "reduced sensation", "decreased sensation"),
    "paresthesia": ("tingling", "pins and needles"),
    "emesis": ("vomiting", "throwing up"),
    "nausea": ("feeling sick", "queasiness"),
    "epistaxis": ("nosebleed", "nose bleeding"),
    "syncope": ("fainting", "loss of consciousness", "passing out"),
    "presyncope": ("near fainting", "lightheadedness"),
    "vertigo": ("spinning sensation", "dizziness"),
    "dizziness": ("light headedness", "lightheadedness"),
    "myalgia": ("muscle pain", "muscle ache", "aching muscles"),
    "arthralgia": ("joint pain", "joint ache", "aching joints"),
    "cephalalgia": ("headache",),
    "alopecia": ("hair loss", "loss of hair"),
    "urticaria": ("hives", "welts"),
    "erythema": ("redness of the skin", "skin redness"),
    "ecchymosis": ("bruising", "bruise"),
    "petechiae": ("pinpoint bruising", "small red spots"),
    "jaundice": ("yellowing of the skin", "icterus", "yellow eyes"),
    "hepatotoxicity": ("liver injury", "liver damage", "hepatic injury"),
    "nephrotoxicity": ("kidney injury", "renal injury", "kidney damage"),
    "thrombocytopenia": ("low platelet", "decreased platelet", "platelet count decreased"),
    "neutropenia": ("low neutrophil", "decreased neutrophil"),
    "leukopenia": ("low white blood cell", "decreased white blood cell"),
    "anemia": ("low hemoglobin", "low red blood cell"),
    "pancytopenia": ("reduction in all blood cell",),
    "hyperglycemia": ("high blood sugar", "elevated blood glucose", "increased blood glucose"),
    "hypoglycemia": ("low blood sugar", "decreased blood glucose"),
    "hypertension": ("high blood pressure", "blood pressure increased", "elevated blood pressure"),
    "hypotension": ("low blood pressure", "blood pressure decreased"),
    "tachycardia": ("fast heart rate", "rapid heart rate", "increased heart rate"),
    "bradycardia": ("slow heart rate", "decreased heart rate"),
    "palpitations": ("pounding heartbeat", "racing heart"),
    "dysgeusia": ("altered taste", "taste disturbance", "change in taste"),
    "anosmia": ("loss of smell",),
    "xerostomia": ("dry mouth",),
    "dysphagia": ("difficulty swallowing", "trouble swallowing"),
    "odynophagia": ("painful swallowing",),
    "dyspepsia": ("indigestion", "upset stomach"),
    "flatulence": ("gas", "bloating"),
    "constipation": ("difficulty passing stool", "hard stools"),
    "ileus": (
        "intestinal obstruction",
        "bowel obstruction",
        "paralytic ileus",
        "gastrointestinal obstruction",
        "obstruction of the intestine",
    ),
    "gastroparesis": ("delayed gastric emptying", "slow stomach emptying"),
    "cholelithiasis": ("gallstones", "gallstone"),
    "cholecystitis": ("gallbladder inflammation", "inflammation of the gallbladder"),
    "nephrolithiasis": ("kidney stones", "kidney stone"),
    "dysuria": ("painful urination", "burning on urination"),
    "hematuria": ("blood in urine", "blood in the urine"),
    "polyuria": ("increased urination", "frequent urination"),
    "oliguria": ("decreased urine output", "reduced urine output"),
    "edema": ("swelling", "fluid retention"),
    "angioedema": ("swelling of the face", "swelling of the lips", "deep swelling"),
    "anaphylaxis": (
        "severe allergic reaction",
        "anaphylactic reaction",
        "serious allergic reaction",
    ),
    "erythema multiforme": ("target lesions",),
    "photosensitivity reaction": ("sun sensitivity", "sensitivity to sunlight"),
    "myelosuppression": ("bone marrow suppression",),
    "seizure": ("convulsion", "fit", "epileptic"),
    "tremor": ("shaking", "trembling"),
    "akathisia": ("inner restlessness", "motor restlessness"),
    "dyskinesia": ("involuntary movement", "abnormal movement"),
    "insomnia": ("difficulty sleeping", "trouble sleeping", "sleeplessness"),
    "confusional state": ("confusion", "disorientation"),
    "suicidal ideation": ("thoughts of suicide", "suicidal thoughts", "thinking about suicide"),
    "depressed mood": ("depression", "low mood"),
    "agitation": ("restlessness",),
    "blurred vision": ("vision blurred", "visual blurring"),
    "diplopia": ("double vision",),
    "photophobia": ("sensitivity to light",),
    "tinnitus": ("ringing in the ears",),
    "vision loss": ("blindness", "loss of vision"),
    "weight decreased": ("weight loss", "loss of weight"),
    "weight increased": ("weight gain",),
    "decreased appetite": ("loss of appetite", "reduced appetite", "anorexia"),
    "dehydration": ("loss of body fluid", "fluid loss"),
    "pneumonitis": ("lung inflammation", "inflammation of the lung"),
    "interstitial lung disease": ("pulmonary fibrosis", "lung scarring"),
    "pulmonary embolism": ("blood clot in the lung",),
    "deep vein thrombosis": ("blood clot in the leg", "venous thrombosis"),
    "myocardial infarction": ("heart attack",),
    "cerebrovascular accident": ("stroke",),
    "cardiac failure": ("heart failure",),
    "atrial fibrillation": ("irregular heartbeat", "irregular heart rhythm"),
    "pancreatitis": ("inflammation of the pancreas",),
    "hypersensitivity": ("allergic reaction", "allergy"),
    "infusion related reaction": ("infusion reaction",),
    "injection site reaction": ("reaction at the injection site",),
    "rhabdomyolysis": ("muscle breakdown", "muscle destruction"),
    "osteonecrosis": ("bone death",),
    "diabetic ketoacidosis": ("ketoacidosis",),
    "thyroid neoplasm": ("thyroid tumor", "thyroid c-cell tumor", "medullary thyroid carcinoma"),
    "sepsis": ("blood infection", "septic shock", "serious infection"),
    "herpes zoster": ("shingles",),
    "oral candidiasis": ("thrush", "oral thrush"),
    "urinary tract infection": ("bladder infection",),
    "nasopharyngitis": ("common cold", "cold symptoms"),
    "upper respiratory tract infection": ("upper respiratory infection",),
}


def _build_reverse_synonyms() -> dict[str, tuple[str, ...]]:
    """Invert :data:`CLINICAL_SYNONYMS` so lookups work in both directions.

    Synonymy is symmetric but the table is written one way round, which without
    this would make matching asymmetric: "Ileus" would find "intestinal
    obstruction" in a label, but the reaction term "Intestinal obstruction" would
    not find "ileus". Both directions occur in FAERS, so both must work.
    """
    reverse: dict[str, set[str]] = {}
    for canonical, variants in CLINICAL_SYNONYMS.items():
        for variant in variants:
            key = normalise_spelling(normalise(variant))
            reverse.setdefault(key, set()).add(canonical)
            # Sibling synonyms are equivalent to each other, not just to the
            # canonical form.
            reverse[key].update(v for v in variants if v != variant)
    return {key: tuple(sorted(values)) for key, values in reverse.items()}


#: Populated at import time, after `normalise` is defined. See `synonyms_for`.
_REVERSE_SYNONYMS: dict[str, tuple[str, ...]] = {}


def synonyms_for(normalised_term: str) -> tuple[str, ...]:
    """Every curated synonym of a normalised term, in both directions."""
    forward = CLINICAL_SYNONYMS.get(normalised_term, ())
    return tuple(dict.fromkeys(forward + _REVERSE_SYNONYMS.get(normalised_term, ())))


#: Suffixes stripped when reducing a word to a comparison stem. Ordered longest
#: first so "-ations" is tried before "-s".
_SUFFIXES: tuple[str, ...] = (
    "ations",
    "ation",
    "ities",
    "ity",
    "ings",
    "ing",
    "ives",
    "ive",
    "ously",
    "ous",
    "ally",
    "als",
    "al",
    "ies",
    "ed",
    "es",
    "s",
)

_NON_WORD = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")

#: Words carrying no discriminating power in either a MedDRA term or label prose.
#: Removed before stem comparison so "Hepatic failure" matches "failure, hepatic".
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "to",
        "and",
        "or",
        "with",
        "without",
        "for",
        "at",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "not",
        "no",
        "nos",
        "unspecified",
        "other",
        "due",
        "related",
    }
)


class MatchKind(StrEnum):
    """How a match was established, ordered from strongest evidence to weakest."""

    EXACT = "exact"
    """The normalised term appears verbatim."""

    SPELLING = "spelling"
    """Matched after British-to-American normalisation."""

    SYNONYM = "synonym"
    """Matched a curated clinical synonym."""

    STEM = "stem"
    """Every content word matched by stem, in any order."""

    NONE = "none"


@dataclass(frozen=True, slots=True)
class LexicalMatch:
    """A lexical hit, with enough context for a human to check it."""

    kind: MatchKind
    matched_text: str
    #: The variant of the reaction term that hit, which for a synonym match is the
    #: synonym rather than the original term. Surfaced in the dashboard so a
    #: reviewer can see *why* a pair was called already-labelled.
    via: str

    @property
    def found(self) -> bool:
        return self.kind is not MatchKind.NONE


NO_MATCH = LexicalMatch(MatchKind.NONE, "", "")


def strip_accents(text: str) -> str:
    """Remove diacritics, which appear in imported reaction terms."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalise(text: str) -> str:
    """Lowercase, de-accent, strip punctuation and collapse whitespace.

    Applied identically to reaction terms and to label text, which is what makes
    the comparison symmetric. Spelling normalisation is *not* applied here -- it is
    a separate step so that :class:`MatchKind` can distinguish a verbatim hit from
    one that needed orthographic help, and a reviewer can see which happened.
    """
    lowered = strip_accents(text).lower()
    return _WHITESPACE.sub(" ", _NON_WORD.sub(" ", lowered)).strip()


def normalise_spelling(text: str) -> str:
    """Apply the British-to-American substitutions to already-normalised text."""
    for british, american in SPELLING_NORMALISATIONS:
        if british in text:
            text = text.replace(british, american)
    return text


def stem(word: str) -> str:
    """Crude suffix-stripping stem.

    Not a linguistic stemmer: a Porter stemmer would conflate terms this domain
    needs kept apart, and a lemmatiser would pull in a model dependency for a job
    that a suffix list does adequately. Words of four characters or fewer are left
    alone, since stripping them produces collisions.
    """
    if len(word) <= 4:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_stems(text: str) -> tuple[str, ...]:
    """Stems of the content words of a normalised phrase."""
    return tuple(stem(word) for word in text.split() if word and word not in _STOPWORDS)


def term_variants(term: str) -> tuple[str, ...]:
    """Every surface form of a reaction term worth searching for.

    The original, its spelling-normalised form, and any curated synonyms of
    either. Deduplicated and ordered longest first, so that a search reports the
    most specific hit rather than an incidental substring of it.
    """
    base = normalise(term)
    normalised = normalise_spelling(base)
    variants = {base, normalised}
    for key in (base, normalised):
        variants.update(synonyms_for(key))
    return tuple(sorted((v for v in variants if v), key=len, reverse=True))


class LexicalMatcher:
    """Finds a reaction term in label text by exact, spelling and stem matching.

    Constructed once per label version: it precomputes the normalised forms of the
    text, which would otherwise be redone for every one of the hundreds of
    reactions checked against that label.
    """

    def __init__(self, label_text: str) -> None:
        self._normalised = normalise(label_text)
        self._spelling_normalised = normalise_spelling(self._normalised)
        self._stems = content_stems(self._spelling_normalised)
        self._stem_set = set(self._stems)

    @property
    def normalised_text(self) -> str:
        return self._spelling_normalised

    def find(self, term: str) -> LexicalMatch:
        """Look for `term`, returning the strongest kind of match found."""
        base = normalise(term)
        if not base:
            return NO_MATCH

        if base and base in self._normalised:
            return LexicalMatch(MatchKind.EXACT, base, base)

        spelled = normalise_spelling(base)
        if spelled != base and spelled in self._spelling_normalised:
            return LexicalMatch(MatchKind.SPELLING, spelled, spelled)

        for synonym in synonyms_for(base) + synonyms_for(spelled):
            candidate = normalise_spelling(normalise(synonym))
            if candidate and candidate in self._spelling_normalised:
                return LexicalMatch(MatchKind.SYNONYM, candidate, synonym)

        # Stem match: every content word of the term is present somewhere in the
        # label. Word order is not required, so "Hepatic failure" matches
        # "failure of hepatic function". A single-word term is not accepted this
        # way -- a lone common stem matches almost any label.
        wanted = content_stems(spelled)
        if len(wanted) >= 2 and all(w in self._stem_set for w in wanted):
            return LexicalMatch(MatchKind.STEM, " ".join(wanted), spelled)

        return NO_MATCH


# Built here rather than at the definition site because inverting the table needs
# `normalise` and `normalise_spelling`, which are defined further down the module.
_REVERSE_SYNONYMS.update(_build_reverse_synonyms())
