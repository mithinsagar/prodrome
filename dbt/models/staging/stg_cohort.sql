-- The tracked cohort for this run, with the identification provenance attached.
--
-- `relies_on_name_matching` is surfaced here rather than computed per mart because
-- it qualifies every figure about a drug: below 95% UNII coverage, the drug's
-- reports were found largely by reporter-supplied names rather than FDA's substance
-- harmonisation. See prodrome.selector for the measurements behind the threshold.
select
    run_id,
    drug_unii,
    drug_name,
    spl_set_id,
    substance_names,
    brand_names,
    unii_coverage,
    coalesce(unii_coverage, 1.0) < 0.95 as relies_on_name_matching,
    spl_set_id is not null            as has_label_timeline,
    notes
from {{ source('prodrome', 'raw_cohort') }}
where run_id = {{ current_run() }}
