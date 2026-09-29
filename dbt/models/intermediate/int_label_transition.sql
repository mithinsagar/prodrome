-- Detected label changes: the version at which a reaction first appears.
--
-- This is the event the whole project predicts, so it is isolated in its own model
-- where it can be inspected and tested directly. `previous_is_labelled = false`
-- alongside `is_labelled = true` is a genuine addition; the reverse ordering is a
-- *removal*, which is rare and usually indicates a parse problem rather than a
-- regulatory action -- so it is surfaced rather than silently tolerated.
with sequenced as (
    select
        drug_unii,
        reaction,
        spl_version,
        authoritative_date,
        is_labelled,
        is_assessable,
        decision_layer,
        evidence_section_name,
        evidence_snippet,
        lag(is_labelled) over (
            partition by drug_unii, reaction order by authoritative_date, spl_version
        ) as previous_is_labelled
    from {{ ref('stg_label_mention') }}
    where is_assessable
)
select
    drug_unii,
    reaction,
    spl_version,
    authoritative_date as transition_date,
    {{ "cast(extract(year from authoritative_date) as varchar) || 'Q' || cast(extract(quarter from authoritative_date) as varchar)" }} as transition_quarter,
    decision_layer,
    evidence_section_name,
    evidence_snippet,
    case
        when previous_is_labelled is null and is_labelled then 'present_at_baseline'
        when not previous_is_labelled and is_labelled     then 'added'
        when previous_is_labelled and not is_labelled     then 'removed'
        else 'unchanged'
    end as transition_type
from sequenced
where previous_is_labelled is distinct from is_labelled
   or previous_is_labelled is null
