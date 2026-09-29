# Security

## Reporting

Open a GitHub issue. This is a research tool with no users to notify and no
deployment to patch, so there is no private disclosure channel; please do not include
your own API keys in a report.

## Secrets

No credential is required to run the pipeline. Two optional ones exist:

| variable | purpose | consequence if absent |
|---|---|---|
| `PRODROME_OPENFDA_API_KEY` | raises the quota from 1,000 to 120,000 requests/day and count aggregations from 100 to 1,000 terms | a full backfill will not complete |
| `PRODROME_LLM_API_KEY` | optional narrative brief | the deterministic template is published instead |

Both are read from the environment or `.env`, which is gitignored. Neither is written
to the warehouse, the exports or the logs: `prodrome doctor` prints the openFDA key
redacted, and `HttpCache.key` deliberately excludes `api_key` from the cache key so a
cached response never contains it. There is a test asserting the last point
(`test_api_key_is_sent_but_not_cached`).

In CI they come from repository secrets and are never echoed. The weekly workflow's
run summary contains counts and the brief, not credentials.

## Handling untrusted input

**XML.** Label documents are parsed with the standard library's `ElementTree` rather
than `defusedxml`. The threat model is narrow and stated at the parse sites: documents
come from NLM's DailyMed over HTTPS, and `ElementTree` resolves neither external
entities nor DTDs, so XXE and external-reference attacks do not apply. The residual
exposure is entity-expansion denial of service against a document already downloaded,
on a developer's own machine. If this project ever parsed SPL from an untrusted
source, `defusedxml` would be the right change.

**ZIP archives.** Label archives are read from memory with `zipfile`, and exactly one
`.xml` member is expected — a different member count raises rather than guessing.
Nothing is extracted to disk, so path traversal through a crafted archive entry is not
reachable.

**HTTP responses.** DailyMed answers a request for a nonexistent label version with
HTTP 200 and an HTML error page, so archive fetches verify the `PK` magic bytes rather
than trusting the status code. `Retry-After` is attacker-influenceable input and is
validated: a negative, NaN, unparseable or past-dated value is treated as malformed
and falls through to normal backoff, never becoming a sleep duration.

**Outbound requests.** The LLM client allowlists both scheme and host, so a mistyped
or injected provider URL cannot become a request to `file:` or an arbitrary host.

**SQL.** Values are always bound parameters. Identifiers are interpolated in two
places only — the table name in `Warehouse.append_rows` and the schema-qualified name
from `Warehouse.find_table` — and both come from this repository's own schema or from
`information_schema`, never from user input.

**Generated text.** The weekly brief's numbers are extracted and matched against a
fixed evidence pack before publication, and a brief containing an unaccounted-for
number is rejected in favour of the deterministic template. The same layer blocks
causal and clinical language. This is a correctness control rather than a security
one, but it is the control that stops a generated document making a claim the data
does not support.

## What this tool is not

prodrome reads public data and writes local files. It has no authentication, no
network listener, no multi-user state and no write access to anything upstream. The
dashboard is static with no backend.

It is **not for clinical use.** openFDA's own disclaimer applies: do not rely on it to
make decisions regarding medical care.
