-- Every estimator and criterion flag, per pair per quarter.
--
-- Criterion flags are unpivoted into a long `criterion`/`fired` shape in
-- int_criterion_firing; they are kept wide here because the marts that show a
-- single row per pair-quarter want them as columns.
select
    run_id,
    drug_unii,
    reaction,
    as_of_quarter,
    {{ quarter_sort_key('as_of_quarter') }} as quarter_index,
    {{ quarter_label_to_date('as_of_quarter') }} as quarter_end_date,
    a, b, c, d, n,
    expected,
    prr, prr_ci_lower, prr_ci_upper,
    ror, ror_ci_lower, ror_ci_upper,
    chi2_yates,
    rrr,
    log2_oe_shrunk, log2_oe_ci_lower, log2_oe_ci_upper,
    ebgm, eb05, eb95,
    gps_posterior_noise_weight,
    raw_p,
    calibrated_p,
    -- The comparison that motivates empirical calibration: a pair that is
    -- nominally significant but not calibrated-significant is one whose apparent
    -- significance is explained by where the null actually sits.
    raw_p < 0.05 and coalesce(calibrated_p, 1.0) >= 0.05 as significance_lost_to_calibration,
    degenerate,
    fired_mhra_prr,
    fired_ema_ror025,
    fired_who_oe025,
    fired_dubious_prr_only,
    n_criteria_firing
from {{ source('prodrome', 'stat_disproportionality') }}
where run_id = {{ current_run() }}
