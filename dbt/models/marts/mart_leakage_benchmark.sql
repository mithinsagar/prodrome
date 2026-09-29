-- How much a conventional retrospective analysis overstates its own signals.
--
-- The comparison the project exists to make. Each row is a pair that did get
-- labelled, showing what the statistic looked like when the label changed versus
-- what it looks like on the full database today.
select
    l.drug_unii,
    c.drug_name,
    l.reaction,
    l.statistic,
    l.at_signal_onset,
    l.at_label_change,
    l.retrospective,
    l.inflation_ratio,
    l.materially_inflated,
    l.reports_before_label,
    l.reports_after_label,
    l.notoriety_surge_ratio,
    l.is_notorious,
    o.signal_quarter,
    o.label_quarter,
    o.lead_time_quarters,
    o.status,
    -- The cleanest statement of the problem: the label change was followed by a
    -- reporting surge *and* the retrospective statistic is materially larger, so
    -- part of the evidence a retrospective analysis cites was created by the
    -- outcome it is being used to predict.
    l.is_notorious and l.materially_inflated as leakage_confirmed
from {{ ref('stg_leakage') }} l
join {{ ref('stg_cohort') }} c using (run_id, drug_unii)
left join {{ ref('stg_pair_outcome') }} o
       on o.run_id = l.run_id and o.drug_unii = l.drug_unii and o.reaction = l.reaction
      and o.criterion = '{{ var("primary_criterion") }}'
order by l.inflation_ratio desc nulls last
