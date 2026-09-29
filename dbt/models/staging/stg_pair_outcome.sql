-- Signal onset, label onset, lead time and censoring, per pair per criterion.
select
    run_id,
    drug_unii,
    reaction,
    criterion,
    status,
    signal_quarter,
    label_quarter,
    lead_time_quarters,
    censored,
    in_survival_set,
    is_event,
    firing_fraction,
    -- Lead time in months reads better on a dashboard axis than quarters, and is
    -- how regulatory timelines are usually discussed.
    lead_time_quarters * 3 as lead_time_months
from {{ source('prodrome', 'stat_pair_outcome') }}
where run_id = {{ current_run() }}
