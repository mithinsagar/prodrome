-- Criterion flags unpivoted to one row per (pair, quarter, criterion).
--
-- The wide-to-long turn is what lets every downstream comparison be written once
-- and faceted by criterion, instead of four near-identical copies of each mart --
-- which is how the criteria would drift apart.
with wide as (select * from {{ ref('stg_disproportionality') }})

{% set criteria = ['mhra_prr', 'ema_ror025', 'who_oe025', 'dubious_prr_only'] %}
{% for criterion in criteria %}
select
    run_id,
    drug_unii,
    reaction,
    as_of_quarter,
    quarter_index,
    '{{ criterion }}' as criterion,
    fired_{{ criterion }} as fired,
    a,
    prr,
    ror,
    ror_ci_lower,
    chi2_yates,
    log2_oe_shrunk,
    ebgm,
    eb05
from wide
{% if not loop.last %}union all{% endif %}
{% endfor %}
