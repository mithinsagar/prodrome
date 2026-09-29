-- Point-in-time 2x2 cells, with the derived margins made explicit.
--
-- The margins are recomputed here rather than stored, so that a test can assert
-- they agree with the cells: if `a + b` ever stops equalling the drug total, the
-- ingest layer has a bug and every ratio built on it is wrong.
select
    run_id,
    drug_unii,
    reaction,
    as_of_quarter,
    {{ quarter_sort_key('as_of_quarter') }} as quarter_index,
    {{ quarter_label_to_date('as_of_quarter') }} as quarter_end_date,
    a,
    b,
    c,
    d,
    a + b            as drug_total,
    a + c            as reaction_total,
    a + b + c + d    as grand_total,
    a_source,
    -- A cell whose `a` came from a truncated aggregation needed its own request.
    -- Tracked through to the marts because a high share is the signal that the
    -- pipeline is running without an API key.
    a_source = 'targeted' as needed_targeted_request
from {{ source('prodrome', 'raw_contingency') }}
where run_id = {{ current_run() }}
