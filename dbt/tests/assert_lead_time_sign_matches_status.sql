-- Lead time and status must tell the same story.
--
-- `labelled_after_signal` means the label followed the signal, so lead time must be
-- non-negative; `labelled_before_signal` means the reverse. A sign that disagrees
-- with the status means the classification in prodrome.latency.onset and the
-- arithmetic have diverged -- which would invert the project's central claim while
-- leaving every summary statistic looking reasonable.
select drug_unii, reaction, criterion, status, lead_time_quarters
from {{ ref('stg_pair_outcome') }}
where (status = 'labelled_after_signal'  and lead_time_quarters < 0)
   or (status = 'labelled_before_signal' and lead_time_quarters >= 0)
   or (status = 'labelled_after_signal'  and lead_time_quarters is null)
   or (status in ('no_signal', 'prevalent_at_baseline', 'unobservable')
       and lead_time_quarters is not null)
