-- The GPS posterior must be ordered: EB05 <= EBGM <= EB95.
--
-- These come from a numerically-solved mixture quantile (there is no closed form),
-- so an ordering violation is the signature of the root finder having converged to
-- the wrong bracket. A tiny tolerance absorbs floating-point noise at the boundary.
select drug_unii, reaction, as_of_quarter, eb05, ebgm, eb95
from {{ ref('stg_disproportionality') }}
where ebgm is not null
  and (eb05 > ebgm + 1e-6 or ebgm > eb95 + 1e-6 or eb05 <= 0)
