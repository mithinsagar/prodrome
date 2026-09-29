-- One row describing the published run, so any figure can be traced to the
-- configuration, the data vintage and the API traffic that produced it.
with r as (
    select * from {{ source('prodrome', 'runs') }} where run_id = {{ current_run() }}
),
h as (
    select * from {{ source('prodrome', 'model_hazard_report') }}
    where run_id = {{ current_run() }}
),
coverage as (
    select
        count(*)                                     as drugs,
        count_if(relies_on_name_matching)            as drugs_reliant_on_name_matching,
        min(unii_coverage)                           as worst_unii_coverage
    from {{ ref('stg_cohort') }}
),
cells as (
    select
        count(*)                                     as contingency_cells,
        count_if(needed_targeted_request)            as cells_needing_targeted_request,
        count(distinct as_of_quarter)                as quarters
    from {{ ref('stg_contingency') }}
)
select
    r.run_id,
    r.started_at,
    r.finished_at,
    r.prodrome_version,
    r.config_digest,
    r.first_quarter,
    r.last_quarter,
    r.openfda_last_updated,
    r.has_openfda_key,
    r.embed_backend,
    r.requests,
    r.cache_hits,
    r.retries,
    cov.drugs,
    cov.drugs_reliant_on_name_matching,
    cov.worst_unii_coverage,
    ce.contingency_cells,
    ce.cells_needing_targeted_request,
    ce.quarters,
    h.roc_auc                as model_roc_auc,
    h.average_precision      as model_average_precision,
    h.precision_at_50        as model_precision_at_50,
    h.lift_at_50             as model_lift_at_50,
    h.base_rate              as model_base_rate,
    h.holdout_from_quarter   as model_holdout_from_quarter,
    h.fitted                 as model_fitted,
    h.note                   as model_note
from r
cross join coverage cov
cross join cells ce
left join h on h.run_id = r.run_id
