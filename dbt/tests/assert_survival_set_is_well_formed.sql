-- Only incident pairs with a signal may enter the survival analysis.
--
-- A pair already labelled at baseline is left-truncated: its label date precedes
-- the archive, so including it would assign it an arbitrary origin and bias lead
-- time toward zero. A pair with no signal has no origin at all. Either leaking into
-- the survival set invalidates the Kaplan-Meier estimate, and the estimate would
-- still look perfectly plausible.
select drug_unii, reaction, criterion, status, in_survival_set, signal_quarter
from {{ ref('stg_pair_outcome') }}
where in_survival_set
  and (status not in ('labelled_after_signal', 'censored') or signal_quarter is null)
