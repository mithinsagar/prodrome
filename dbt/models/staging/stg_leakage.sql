-- Point-in-time versus retrospective statistics, per labelled pair.
select
    run_id,
    drug_unii,
    reaction,
    statistic,
    at_signal_onset,
    at_label_change,
    retrospective,
    inflation_ratio,
    reports_before_label,
    reports_after_label,
    notoriety_surge_ratio,
    notoriety_window_quarters,
    is_notorious,
    -- The headline framing: a conventional retrospective analysis would have
    -- reported the `retrospective` value and implicitly claimed the method found it.
    inflation_ratio >= 2.0 as materially_inflated
from {{ source('prodrome', 'stat_leakage') }}
where run_id = {{ current_run() }}
