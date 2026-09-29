-- Artefact indicators per pair.
select
    run_id,
    drug_unii,
    reaction,
    reporter_concentration,
    top_country,
    top_country_share,
    consumer_share,
    lawyer_share,
    spike_ratio,
    spike_quarter,
    quarters_with_reports,
    single_quarter_dependence,
    robustness_score,
    artefact_flags,
    artefact_flags <> '' as has_artefact_flag,
    -- FAERS is a US database, so US dominance is unremarkable. Concentration in a
    -- single *non-US* country is the pattern worth flagging.
    reporter_concentration >= 0.5 and coalesce(top_country, 'US') <> 'US'
        as concentrated_outside_us
from {{ source('prodrome', 'stat_diagnostics') }}
where run_id = {{ current_run() }}
