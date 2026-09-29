-- Quarter-by-quarter series per pair, for the trend view and the Tableau extract.
--
-- Restricted to pairs that ever fired a criterion: the full cross product of every
-- pair and quarter is an order of magnitude larger and the extra rows are all
-- flat lines at no signal.
with fired_ever as (
    select distinct drug_unii, reaction
    from {{ ref('stg_disproportionality') }}
    where n_criteria_firing > 0
)
select
    p.drug_unii,
    p.drug_name,
    p.reaction,
    p.as_of_quarter,
    p.quarter_index,
    p.quarter_end_date,
    p.a  as reports,
    p.expected,
    p.prr,
    p.prr_ci_lower,
    p.prr_ci_upper,
    p.ror,
    p.ror_ci_lower,
    p.ror_ci_upper,
    p.log2_oe_shrunk,
    p.log2_oe_ci_lower,
    p.ebgm,
    p.eb05,
    p.chi2_yates,
    p.raw_p,
    p.calibrated_p,
    p.n_criteria_firing,
    p.fired_mhra_prr,
    p.fired_ema_ror025,
    p.fired_who_oe025,
    p.fired_dubious_prr_only,
    p.labelled_at_quarter,
    p.is_label_gap_at_quarter,
    p.label_version_in_force,
    t.transition_type,
    t.evidence_section_name as label_change_section
from {{ ref('int_pair_quarter') }} p
join fired_ever f using (drug_unii, reaction)
left join {{ ref('int_label_transition') }} t
       on t.drug_unii = p.drug_unii and t.reaction = p.reaction
      and t.transition_quarter = p.as_of_quarter
      and t.transition_type = 'added'
order by p.drug_unii, p.reaction, p.quarter_index
