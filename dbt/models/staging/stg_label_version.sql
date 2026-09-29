-- The reconstructed label timeline, with validity intervals.
--
-- `valid_to` is the day before the next version took effect, so the intervals
-- tile the timeline without overlap. This is what makes "what did the label say on
-- date X" answerable by a range join, which is the whole point of reconstructing
-- the history rather than reading only the current label.
with ordered as (
    select
        run_id,
        drug_unii,
        spl_set_id,
        spl_version,
        published_date,
        effective_date,
        authoritative_date,
        core_char_count,
        section_count,
        is_usable,
        lead(authoritative_date) over (
            partition by drug_unii order by authoritative_date, spl_version
        ) as next_date,
        row_number() over (
            partition by drug_unii order by authoritative_date, spl_version
        ) as version_rank
    from {{ source('prodrome', 'raw_label_version') }}
    where run_id = {{ current_run() }}
)
select
    *,
    authoritative_date as valid_from,
    coalesce(next_date - interval 1 day, date '9999-12-31') as valid_to,
    version_rank = 1 as is_baseline_version
from ordered
