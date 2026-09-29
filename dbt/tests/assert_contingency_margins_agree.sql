-- The 2x2 cells must describe a real table.
--
-- `a` cannot exceed either margin, and every cell must be non-negative. These are
-- not hypothetical: openFDA serves the four counts behind a table from four
-- separate requests, and if it reindexes between them the margins disagree. The
-- Python layer rejects such tables (see Contingency.from_marginals), so this test
-- asserts that guard is actually holding rather than trusting it.
select drug_unii, reaction, as_of_quarter, a, b, c, d
from {{ ref('stg_contingency') }}
where a < 0 or b < 0 or c < 0 or d < 0
   or a > drug_total
   or a > reaction_total
   or grand_total < drug_total
   or grand_total < reaction_total
