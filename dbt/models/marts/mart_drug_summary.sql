-- One row per drug: the dashboard's entry point and the Tableau dimension table.
with gaps as (
    select drug_unii, count(*) as open_label_gaps
    from {{ ref('mart_label_gap') }}
    group by drug_unii
),
outcomes as (
    select
        drug_unii,
        count_if(status = 'labelled_after_signal') as signals_that_were_labelled,
        count_if(status = 'censored')              as open_signals,
        count_if(status = 'prevalent_at_baseline') as already_labelled_at_baseline,
        median(case when status = 'labelled_after_signal'
                    then lead_time_quarters end)   as median_lead_quarters
    from {{ ref('stg_pair_outcome') }}
    where criterion = '{{ var("primary_criterion") }}'
    group by drug_unii
),
labels as (
    select
        drug_unii,
        count(*)                as label_versions,
        count_if(is_usable)     as usable_label_versions,
        min(authoritative_date) as first_label_date,
        max(authoritative_date) as latest_label_date
    from {{ ref('stg_label_version') }}
    group by drug_unii
),
volume as (
    select
        drug_unii,
        count(distinct reaction) as reactions_tracked,
        max(a + b)               as latest_drug_reports
    from {{ ref('stg_contingency') }}
    group by drug_unii
)
select
    c.drug_unii,
    c.drug_name,
    c.spl_set_id,
    c.unii_coverage,
    c.relies_on_name_matching,
    c.has_label_timeline,
    v.reactions_tracked,
    v.latest_drug_reports,
    l.label_versions,
    l.usable_label_versions,
    l.first_label_date,
    l.latest_label_date,
    o.signals_that_were_labelled,
    o.open_signals,
    o.already_labelled_at_baseline,
    o.median_lead_quarters,
    coalesce(g.open_label_gaps, 0) as open_label_gaps,
    c.notes
from {{ ref('stg_cohort') }} c
left join volume   v using (drug_unii)
left join labels   l using (drug_unii)
left join outcomes o using (drug_unii)
left join gaps     g using (drug_unii)
order by coalesce(g.open_label_gaps, 0) desc, c.drug_name
