-- Open label gaps as of the latest quarter: reactions reported
-- disproportionately for a drug whose label does not describe them.
--
-- This is the drill-down surface for the dashboard. It deliberately carries the
-- artefact diagnostics alongside the statistics, because a gap with a litigation
-- pattern and a gap with steady multi-country clinical reporting are not the same
-- finding, and presenting them on one undifferentiated list is how a reviewer's
-- attention gets wasted.
with latest_quarter as (
    select max(as_of_quarter) as q from {{ ref('int_pair_quarter') }}
),
current_state as (
    select p.*
    from {{ ref('int_pair_quarter') }} p
    cross join latest_quarter l
    where p.as_of_quarter = l.q
),
onsets as (
    select drug_unii, reaction, signal_quarter, firing_fraction
    from {{ ref('stg_pair_outcome') }}
    where criterion = '{{ var("primary_criterion") }}'
)
select
    c.drug_unii,
    c.drug_name,
    c.reaction,
    c.as_of_quarter,
    c.a                       as reports,
    c.expected,
    c.prr,
    c.ror,
    c.ror_ci_lower,
    c.log2_oe_shrunk,
    c.ebgm,
    c.eb05,
    c.raw_p,
    c.calibrated_p,
    c.significance_lost_to_calibration,
    c.n_criteria_firing,
    c.fired_mhra_prr,
    c.fired_ema_ror025,
    c.fired_who_oe025,
    o.signal_quarter,
    o.firing_fraction,
    c.label_version_in_force,
    c.robustness_score,
    c.artefact_flags,
    c.has_artefact_flag,
    c.relies_on_name_matching,
    q.hazard_probability,
    q.rank as priority_rank
from current_state c
left join onsets o using (drug_unii, reaction)
left join {{ source('prodrome', 'model_priority_queue') }} q
       on q.run_id = c.run_id and q.drug_unii = c.drug_unii and q.reaction = c.reaction
where c.is_label_gap_at_quarter
  and c.a >= {{ var('min_case_count') }}
  and c.n_criteria_firing > 0
order by coalesce(q.hazard_probability, 0) desc, c.eb05 desc nulls last
