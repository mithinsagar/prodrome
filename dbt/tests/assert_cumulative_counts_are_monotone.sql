-- Cumulative point-in-time counts must never decrease.
--
-- This is the strongest correctness check in the project, and it is specific to
-- the point-in-time design. Every cell counts reports with `receivedate` on or
-- before the quarter end, so for a fixed pair the count is cumulative and can only
-- rise. A decrease means one of:
--
--   * a window was built with the wrong start date, so quarters are not nested;
--   * `receiptdate` crept in somewhere in place of `receivedate` -- receiptdate
--     moves forward when a report is amended, which un-nests the windows;
--   * two quarters' counts came from different openFDA index generations.
--
-- All three silently corrupt every lead-time measurement, and none would be
-- visible in any aggregate. A small tolerance is allowed because openFDA does
-- occasionally remove duplicate reports between index builds, which legitimately
-- lowers a historical count by a handful.
with sequenced as (
    select
        drug_unii,
        reaction,
        as_of_quarter,
        quarter_index,
        a,
        lag(a) over (
            partition by drug_unii, reaction order by quarter_index
        ) as previous_a
    from {{ ref('stg_contingency') }}
)
select drug_unii, reaction, as_of_quarter, previous_a, a, previous_a - a as decrease
from sequenced
where previous_a is not null
  and a < previous_a - greatest(5, previous_a * 0.02)
