"""Composing the weekly brief.

The brief is always generated deterministically first. That version is complete,
publishable and reproducible: same warehouse, same bytes. If a model is configured,
it is asked to rewrite the same evidence more readably, and its output is accepted
only if it passes numeric verification and the forbidden-claims check. Otherwise the
deterministic version is published and the reason recorded.

The ordering is the point. The template is not a fallback bolted on for robustness;
it is the reference output, and the model is an optional improvement to its prose.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from prodrome.brief.evidence import EvidencePack
from prodrome.brief.llm import LlmError, TextGenerator
from prodrome.brief.verify import VerificationReport, check_forbidden_claims, verify_numbers

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are writing an internal weekly pharmacovigilance signal brief for a drug-safety
team. You will be given a JSON evidence pack of already-computed findings.

Hard rules:
1. Use ONLY numbers that appear in the evidence pack. Never compute, infer, combine
   or estimate a number. If a figure is not in the pack, do not state it.
2. Never assert causation. These are disproportionate *reporting* associations in a
   spontaneous-report database with no denominator and no control group. Write
   "reported disproportionately", never "causes", "confirms", "proves" or
   "incidence".
3. Reproduce every caveat listed for a gap. If a gap is flagged for a reporting
   artefact, say so in the same sentence that reports its statistic.
4. Give no clinical or prescribing advice.
5. Be concise and factual. No preamble, no marketing tone, no speculation about
   mechanism.

Structure your output as GitHub-flavoured Markdown:
- a two-sentence summary of what changed this week
- a "## Signals without label coverage" section with one short paragraph per gap
- a "## Method notes" section of at most three bullets

Write in British English. Prefer plain words over jargon where both are exact.
"""


@dataclass(frozen=True, slots=True)
class Brief:
    """A composed brief and how it was produced."""

    markdown: str
    source: str
    verification: VerificationReport | None
    rejection_reason: str | None = None

    @property
    def was_model_generated(self) -> bool:
        return self.source not in ("template",)


def render_template(pack: EvidencePack) -> str:  # noqa: PLR0912
    """The deterministic brief.

    Complete and publishable on its own. Written to be honest rather than fluent:
    every statement is a direct restatement of a stored number, with its caveats
    attached in the same sentence.
    """
    lines: list[str] = [
        f"# Weekly signal brief -- {pack.as_of_quarter}",
        "",
        f"Generated {pack.generated_on.isoformat()} from run `{pack.run_id}`. "
        f"openFDA data current to {pack.data_vintage or 'unknown'}. "
        f"Cohort: {pack.cohort_size} drugs.",
        "",
        f"**{pack.total_open_gaps} drug-reaction pairs are reported disproportionately "
        f"and are not described in the drug's current label.** "
        f"The {len(pack.gaps)} highest-priority are listed below, ranked by the "
        f"calibrated probability of a label change within the next eight quarters.",
        "",
    ]

    inflation = pack.median_leakage_inflation
    if inflation is not None:
        # The direction matters. Asserting that a retrospective statistic is inflated
        # while displaying a ratio below 1.0 would contradict itself in the same
        # sentence -- and a ratio at or below 1 is a real finding for a cohort whose
        # labelled pairs were all labelled long before the observation window.
        if inflation >= 1.05:
            lines += [
                f"For pairs that were eventually labelled, the reporting odds ratio "
                f"computed on the full database is a median {inflation}x the value "
                f"available when the label actually changed. Reporting of a reaction "
                f"rises once it is labelled, so a retrospective statistic is inflated "
                f"by the outcome it would be used to predict.",
                "",
            ]
        else:
            lines += [
                f"For pairs that were eventually labelled, the reporting odds ratio "
                f"computed on the full database is a median {inflation}x the value "
                f"available when the label changed -- no material retrospective "
                f"inflation in this cohort. That is expected when the labelled pairs "
                f"were labelled well before the observation window, leaving no "
                f"post-labelling reporting surge inside it.",
                "",
            ]

    lines += ["## Signals without label coverage", ""]
    if not pack.gaps:
        lines += ["No open label gaps met the reporting threshold this quarter.", ""]
    for index, gap in enumerate(pack.gaps, start=1):
        parts = [f"{gap.reports} reports against {gap.expected:.1f} expected"]
        if gap.ror is not None and gap.ror_ci_lower is not None:
            parts.append(f"ROR {gap.ror:.2f} (95% lower bound {gap.ror_ci_lower:.2f})")
        if gap.eb05 is not None:
            parts.append(f"EB05 {gap.eb05:.2f}")
        if gap.signal_quarter:
            since = (
                f", {gap.quarters_since_signal} quarters ago"
                if gap.quarters_since_signal is not None
                else ""
            )
            parts.append(f"first met a signalling criterion in {gap.signal_quarter}{since}")
        lines.append(f"**{index}. {gap.headline}** -- " + "; ".join(parts) + ".")
        if gap.caveats:
            lines.append("")
            for caveat in gap.caveats:
                lines.append(f"  - Caveat: {caveat}.")
        lines.append("")

    lines += ["## Method notes", ""]
    reached = {
        criterion: median
        for criterion, median in pack.median_lead_quarters_by_criterion.items()
        if median is not None
    }
    if reached:
        pairs = ", ".join(f"{c} {m:.1f}" for c, m in sorted(reached.items()))
        lines.append(
            f"- Median quarters from a criterion first firing to the label change, "
            f"by criterion: {pairs}."
        )
    elif pack.median_lead_quarters_by_criterion:
        lines.append(
            "- No criterion reached a median time-to-label in this run: no tracked pair "
            "acquired a label mention after its signal fired within the observation "
            "window. Pairs already labelled at the earliest archived label version are "
            "left-truncated and excluded by design."
        )
    # Report model metrics only when a model was actually fitted. A precision of 0.0%
    # against a base rate of 0.00% is not a result, it is the absence of one, and
    # printing it as though it were a measurement is worse than saying nothing.
    if (
        pack.model_precision_at_50 is not None
        and pack.model_base_rate is not None
        and pack.model_base_rate > 0
    ):
        lines.append(
            f"- Of the 50 pairs the model ranked highest in the held-out period, "
            f"{pack.model_precision_at_50:.1%} were labelled within the horizon, against "
            f"a base rate of {pack.model_base_rate:.2%}"
            + (
                f" -- a lift of {pack.model_lift_at_50:.1f}x."
                if pack.model_lift_at_50 is not None
                else "."
            )
        )
    else:
        lines.append(
            "- No prioritisation model was fitted for this run: too few observed label "
            "changes in the training period. Ranking falls back to the shrinkage "
            "observed-to-expected ratio, which is the most conservative single measure "
            "available."
        )
    lines.append(
        "- These are disproportionate reporting associations in FAERS, not measured "
        "risks. FAERS has no denominator and no control group, reporting is voluntary "
        "and incomplete, and a disproportionate ratio is a hypothesis to evaluate, not "
        "evidence of causation."
    )
    lines.append("")
    return "\n".join(lines)


def compose_brief(pack: EvidencePack, generator: TextGenerator | None) -> Brief:
    """Compose the brief, using a model only if its output verifies."""
    template = render_template(pack)
    if generator is None:
        return Brief(markdown=template, source="template", verification=None)

    user_prompt = "Evidence pack (JSON). Every number you may use is in here:\n\n" + json.dumps(
        pack.as_prompt_payload(), indent=2
    )
    try:
        generated = generator.generate(SYSTEM_PROMPT, user_prompt)
    except LlmError as exc:
        logger.warning("generation failed (%s); publishing the deterministic brief", exc)
        return Brief(
            markdown=template,
            source="template",
            verification=None,
            rejection_reason=f"generation failed: {exc}",
        )

    if not generated.strip():
        return Brief(template, "template", None, "model returned an empty response")

    verification = verify_numbers(generated, pack.allowed_numbers)
    forbidden = check_forbidden_claims(generated)

    if not verification.passed:
        logger.warning("%s; publishing the deterministic brief", verification.summary())
        return Brief(template, "template", verification, verification.summary())
    if forbidden:
        reason = "forbidden claims: " + "; ".join(forbidden)
        logger.warning("%s; publishing the deterministic brief", reason)
        return Brief(template, "template", verification, reason)

    footer = (
        f"\n\n---\n*Narrative generated by {generator.name} from a fixed evidence pack; "
        f"{verification.checked} quantitative claims were checked against it and all "
        f"matched. Numbers are computed by the pipeline, not by the model.*\n"
    )
    return Brief(generated.strip() + footer, generator.name, verification)
