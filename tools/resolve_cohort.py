"""Resolve a list of drug names into a reviewable ``conf/cohort.yml``.

Run this when the cohort changes, review the diff, and commit it. It is not part
of the pipeline: the cohort is an input to the study, not a derived artefact, and
regenerating it silently on every run would let an upstream change rewrite the
study population underneath a published figure.

Usage::

    python tools/resolve_cohort.py                  # resolve the default list
    python tools/resolve_cohort.py --names A B C     # resolve specific drugs
    python tools/resolve_cohort.py --show-rejected   # explain every choice
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from prodrome.clients.factory import build_dailymed, build_openfda
from prodrome.clients.openfda import LABEL_ENDPOINT
from prodrome.config import get_settings, load_config
from prodrome.ingest.cohort import CohortResolver

# A cohort chosen for one property above all: these drugs have *moved*. Each is
# either a product whose safety labelling demonstrably changed between 2016 and
# now, or a recent approval under active post-marketing surveillance -- so the
# outcome variable has both positives and genuine right-censored observations.
# Breadth across therapeutic areas is deliberate: a cohort of nothing but GLP-1s
# would let the model learn "is it a GLP-1" instead of "is this signal real".
DEFAULT_NAMES: tuple[str, ...] = (
    # Incretin and metabolic -- high volume, rapidly evolving labels.
    "semaglutide",
    "tirzepatide",
    "dulaglutide",
    "liraglutide",
    "exenatide",
    "empagliflozin",
    "dapagliflozin",
    "canagliflozin",
    # JAK inhibitors -- class-wide boxed warnings added 2021.
    "tofacitinib",
    "baricitinib",
    "upadacitinib",
    # Checkpoint inhibitors -- immune-related adverse reactions accumulated
    # across many label revisions.
    "pembrolizumab",
    "nivolumab",
    "atezolizumab",
    "ipilimumab",
    "durvalumab",
    # Direct oral anticoagulants.
    "apixaban",
    "rivaroxaban",
    "dabigatran",
    "edoxaban",
    # CGRP migraine agents -- all post-2018, still accruing label history.
    "erenumab",
    "galcanezumab",
    "rimegepant",
    "ubrogepant",
    # Multiple sclerosis.
    "ocrelizumab",
    "natalizumab",
    "fingolimod",
    "dimethyl fumarate",
    "alemtuzumab",
    # Psychiatry and neurology.
    "aripiprazole",
    "brexpiprazole",
    "lurasidone",
    "valbenazine",
    "esketamine",
    # Canonical cases where a FAERS signal preceded a labelling change by years.
    "montelukast",  # neuropsychiatric boxed warning, March 2020
    "febuxostat",  # cardiovascular mortality boxed warning, 2019
    "varenicline",
    "hydroxychloroquine",
    "tramadol",
    "gabapentin",
    "pregabalin",
    # Oncology small molecules.
    "ibrutinib",
    "osimertinib",
    "palbociclib",
    "olaparib",
    "lenvatinib",
    # Immunology biologics.
    "adalimumab",
    "ustekinumab",
    "secukinumab",
    "dupilumab",
    # Cardiorenal and other recent approvals.
    "finerenone",
    "sacubitril",
    "tafamidis",
    "teriflunomide",
    "siponimod",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="*", default=list(DEFAULT_NAMES))
    parser.add_argument("--out", type=Path, default=Path("conf/cohort.yml"))
    parser.add_argument("--show-rejected", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config()
    settings = get_settings()
    events, events_transport = build_openfda(config, settings)
    labels, labels_transport = build_openfda(config, settings, endpoint=LABEL_ENDPOINT)
    dailymed, dailymed_transport = build_dailymed(config, settings)

    try:
        resolver = CohortResolver(dailymed, events, labels)
        resolutions = resolver.resolve_all(args.names)
    finally:
        for transport in (events_transport, labels_transport, dailymed_transport):
            transport.close()

    accepted = [r for r in resolutions if r.ok]
    rejected = [r for r in resolutions if not r.ok]

    print(f"\nResolved {len(accepted)}/{len(resolutions)} requested drugs.\n")
    print(f"{'drug':20s} {'UNII':11s} {'ver':>5s} {'reports':>9s} {'UNII cov':>9s}  labeller")
    print("-" * 100)
    for r in accepted:
        c = r.chosen
        assert c is not None and r.unii is not None
        cov = f"{c.coverage.unii_coverage:8.1%}" if c.coverage else "        ?"
        flag = " !" if c.coverage and c.coverage.relies_on_names else "  "
        print(
            f"{r.requested[:20]:20s} {r.unii:11s} {c.latest_version:5d} "
            f"{c.event_report_count:9,d} {cov}{flag} {c.labeller[:34]}"
        )
    weak = [
        r for r in accepted if r.chosen and r.chosen.coverage and r.chosen.coverage.relies_on_names
    ]
    if weak:
        print(
            f"\n  ! {len(weak)} of {len(accepted)} drugs rely on reporter-supplied name "
            f"matching because openFDA's UNII harmonisation does not cover them.\n"
            f"    Their unii_coverage is recorded in the cohort and carried into every result."
        )

    if rejected:
        print(f"\n{len(rejected)} not included:")
        for r in rejected:
            print(f"  {r.requested:24s} {r.problem}")

    if args.show_rejected:
        print("\nPer-drug candidate detail:")
        for r in resolutions:
            print(f"\n  {r.requested}")
            for c in sorted(r.candidates, key=lambda x: x.score, reverse=True)[:6]:
                mark = "->" if r.chosen is c else "  "
                actives = ",".join(sorted(c.uniis)) or "-"
                print(
                    f"   {mark} {c.set_id[:8]} revs={c.version_count:3d} "
                    f"repack={c.looks_like_repackager!s:5s} actives={actives:24s} "
                    f"{c.labeller[:32]}"
                )

    payload = {"cohort": [r.to_cohort_entry() for r in accepted]}
    header = (
        "# Tracked cohort, resolved by tools/resolve_cohort.py and committed for review.\n"
        "#\n"
        "# `unii` is FDA's substance identifier and is the join key for adverse events:\n"
        "# openFDA's own name harmonisation is unreliable (Ozempic's reports carry\n"
        '# generic_name = "ORAL SEMAGLUTIDE" for a subcutaneous injection), so names are\n'
        "# never used to join.\n"
        "#\n"
        "# `spl_set_id` must be the *application holder's* label. Repackager set ids carry\n"
        "# a single version and would make a drug look as though its label never changed,\n"
        "# turning every real label change into a false negative in the outcome variable.\n"
        "# The resolver ranks by revision count precisely because only the application\n"
        "# holder revises a label.\n"
        "#\n"
        "# Regenerate with:  python tools/resolve_cohort.py --show-rejected\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(f"\nWrote {len(accepted)} entries to {args.out}")
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
