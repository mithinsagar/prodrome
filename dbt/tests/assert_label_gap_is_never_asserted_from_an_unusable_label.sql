-- A label gap must rest on a label we could actually read.
--
-- The most dangerous failure mode in the label-matching layer is treating an
-- unparseable or truncated label document as "this reaction is not described",
-- which manufactures exactly the finding the project reports. The verdict
-- vocabulary keeps `unknown` distinct from `not_labelled` for this reason, and this
-- test asserts the distinction survives into the marts.
select drug_unii, reaction, as_of_quarter, is_label_gap_at_quarter,
       label_assessable_at_quarter, label_verdict_at_quarter
from {{ ref('int_pair_quarter') }}
where is_label_gap_at_quarter
  and (not label_assessable_at_quarter
       or label_verdict_at_quarter = 'unknown'
       or label_assessable_at_quarter is null)
