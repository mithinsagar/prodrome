-- The project's headline result: how much lead time each signal-detection
-- criterion actually buys, and what it costs in false alarms.
--
-- One row per criterion. This is the table that answers the question the project
-- was built to ask, and the one an interviewer should be pointed at first.
with outcomes as (
    select * from {{ ref('stg_pair_outcome') }}
),
curve as (
    select * from {{ source('prodrome', 'model_survival_curve') }}
    where run_id = {{ current_run() }}
),
per_criterion as (
    select
        criterion,
        count(*)                                                as pairs_evaluated,
        count_if(status = 'labelled_after_signal')              as signalled_then_labelled,
        count_if(status = 'labelled_before_signal')             as labelled_before_signal,
        count_if(status = 'censored')                           as signalled_not_yet_labelled,
        count_if(status = 'prevalent_at_baseline')              as excluded_left_truncated,
        count_if(status = 'no_signal')                          as never_signalled,
        count_if(status = 'unobservable')                       as unobservable,
        median(case when status = 'labelled_after_signal'
                    then lead_time_quarters end)                as median_lead_quarters,
        median(case when status = 'labelled_after_signal'
                    then lead_time_months end)                  as median_lead_months,
        avg(firing_fraction)                                    as mean_firing_fraction
    from outcomes
    group by criterion
)
select
    p.criterion,
    p.pairs_evaluated,
    p.signalled_then_labelled,
    p.signalled_not_yet_labelled,
    p.labelled_before_signal,
    p.never_signalled,
    p.excluded_left_truncated,
    p.unobservable,
    p.median_lead_quarters,
    p.median_lead_months,
    p.mean_firing_fraction,
    c.median_quarters_to_label as km_median_quarters,
    c.labelled_by_1y,
    c.labelled_by_2y,
    c.labelled_by_3y,
    -- Precision of the criterion as a *predictor of labelling*: of the pairs where
    -- it fired and the outcome is known, how often did a label change follow.
    case
        when p.signalled_then_labelled + p.signalled_not_yet_labelled > 0
        then p.signalled_then_labelled::double
             / (p.signalled_then_labelled + p.signalled_not_yet_labelled)
    end as share_of_signals_labelled,
    -- The cost side: how many pairs it fired on that never led anywhere. A
    -- criterion with long lead time and a huge alarm count is not obviously better
    -- than one with short lead time and few.
    p.signalled_not_yet_labelled as open_alarms
from per_criterion p
left join curve c on c.criterion = p.criterion
order by p.median_lead_quarters desc nulls last
