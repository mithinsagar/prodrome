-- Per-version verdicts on whether a label describes a reaction.
--
-- Unusable versions are carried through with `is_assessable = false` rather than
-- filtered out. That distinction is load-bearing: treating an unparseable label as
-- "not labelled" would fabricate a label gap, and treating it as "labelled" would
-- erase a real one.
select
    m.run_id,
    m.drug_unii,
    m.spl_version,
    m.authoritative_date,
    m.reaction,
    m.verdict,
    m.is_labelled,
    v.is_usable as is_assessable,
    v.is_baseline_version,
    m.best_semantic_score,
    m.best_semantic_z,
    m.evidence_method,
    m.evidence_score,
    m.evidence_section_name,
    m.evidence_tier,
    m.evidence_snippet,
    m.evidence_via,
    -- Which layer settled it, for the precision report in docs/METHODS.md.
    case
        when m.evidence_method like 'lexical:%' then 'lexical'
        when m.evidence_method = 'semantic'     then 'semantic'
        else 'none'
    end as decision_layer
from {{ source('prodrome', 'raw_label_mention') }} m
join {{ ref('stg_label_version') }} v
  on v.drug_unii = m.drug_unii
 and v.spl_version = m.spl_version
where m.run_id = {{ current_run() }}
