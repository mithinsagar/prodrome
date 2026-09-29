-- The analytical spine: one row per (drug, reaction, quarter) with statistics,
-- cohort context and what the label said at that moment.
--
-- The label join is the part that does not exist in conventional tooling. It is a
-- range join against the reconstructed version intervals, which answers "what did
-- the label say as of this quarter" rather than "what does the label say now".
-- Without it, a label-gap flag on a 2019 row would be computed against a 2026
-- label -- crediting the analysis with knowledge it could not have had.
with spine as (
    select
        s.*,
        c.drug_name,
        c.unii_coverage,
        c.relies_on_name_matching,
        c.has_label_timeline
    from {{ ref('stg_disproportionality') }} s
    join {{ ref('stg_cohort') }} c using (run_id, drug_unii)
),

-- The label version in force at each quarter end, and its verdict.
label_state as (
    select
        sp.drug_unii,
        sp.reaction,
        sp.as_of_quarter,
        m.spl_version,
        m.is_labelled,
        m.is_assessable,
        m.verdict,
        m.decision_layer,
        row_number() over (
            partition by sp.drug_unii, sp.reaction, sp.as_of_quarter
            order by v.authoritative_date desc, m.spl_version desc
        ) as recency
    from spine sp
    join {{ ref('stg_label_mention') }} m
      on m.drug_unii = sp.drug_unii and m.reaction = sp.reaction
    join {{ ref('stg_label_version') }} v
      on v.drug_unii = m.drug_unii and v.spl_version = m.spl_version
    where v.authoritative_date <= sp.quarter_end_date
),

in_force as (
    select * from label_state where recency = 1
)

select
    sp.*,
    f.spl_version         as label_version_in_force,
    f.is_labelled         as labelled_at_quarter,
    f.is_assessable       as label_assessable_at_quarter,
    f.verdict             as label_verdict_at_quarter,
    f.decision_layer      as label_decision_layer,
    -- A label gap is a *confident* absence at that point in time: the label was
    -- assessable and did not describe the reaction. An unassessable label is
    -- neither a gap nor a mention.
    case
        when f.is_assessable is null   then null
        when not f.is_assessable       then null
        else not f.is_labelled
    end as is_label_gap_at_quarter,
    d.robustness_score,
    d.artefact_flags,
    d.has_artefact_flag
from spine sp
left join in_force f
  on f.drug_unii = sp.drug_unii and f.reaction = sp.reaction
 and f.as_of_quarter = sp.as_of_quarter
left join {{ ref('stg_diagnostics') }} d
  on d.drug_unii = sp.drug_unii and d.reaction = sp.reaction
