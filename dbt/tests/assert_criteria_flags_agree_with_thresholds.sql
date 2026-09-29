-- Criterion flags must agree with the thresholds they claim to implement.
--
-- The flags are written by the Python criteria registry; this recomputes two of
-- them in SQL and asserts they match. It is a genuine cross-check rather than a
-- tautology: the two implementations share no code, so a change to a threshold in
-- one place and not the other is caught here instead of silently shifting every
-- reported lead time.
--
-- NULL handling matters: a NaN estimator is written as NULL, and a NULL comparison
-- must fail closed (flag false), never fire.
select
    drug_unii, reaction, as_of_quarter,
    a, prr, chi2_yates, ror_ci_lower,
    fired_mhra_prr, fired_ema_ror025
from {{ ref('stg_disproportionality') }}
where fired_mhra_prr
      is distinct from (a >= 3 and coalesce(prr, 0) >= 2.0 and coalesce(chi2_yates, 0) >= 4.0)
   or fired_ema_ror025
      is distinct from (a >= 3 and coalesce(ror_ci_lower, 0) > 1.0)
