# Academic reference importer

Build an explicit list of security and cryptography conference papers, then
measure and retrieve abstract coverage. Requires Python 3.9+. Proceedings
collection uses BeautifulSoup; the original OpenAlex/Semantic Scholar importer
uses the standard library.

For semantic search, [generate embeddings on a GPU server and query them on your
Mac](docs/embeddings.md). The embedding tools default to Qwen3-Embedding-4B on
both machines and support Python 3.9.6. The corpus and generated vectors stay
outside Git. Switching from Nemotron requires rebuilding the index; the new
default output is `data/qwen3-4b-embeddings.sqlite3`.

Sync code through [GitHub](https://github.com/arifiorino/reference_finder): commit
your changes and run `git push origin main` on your Mac, then run
`git pull --ff-only origin main` in the server checkout. The
[embedding guide](docs/embeddings.md#push-code-from-your-mac-and-pull-on-the-server)
covers initial cloning, switching an existing bundle checkout to GitHub, and
transferring the SQLite corpus and completed index separately.

Choose dependencies for the machine's role:

| Role | Requirements file | Python |
| --- | --- | --- |
| Mac client: embed a query and search a saved index | `requirements-client.txt` | 3.9–3.12, including 3.9.6 |
| GPU server: embed the paper corpus | `requirements-server.txt` | 3.9–3.12, including 3.9.6 |
| Collect paper lists and abstracts | `requirements-webscraping.txt` | 3.9+ |

Both embedding environments load the same model and therefore share the model
dependencies in `requirements-embeddings.txt`. Their setup commands are in the
[embedding guide](docs/embeddings.md). `requirements.txt` remains a legacy alias
for the webscraping dependencies; it does not install embedding dependencies.

The default is the **current year plus the preceding nine years**: **2017–2026**
when run in 2026. The proceedings inventory uses event years; the original API
discovery commands use publication years as recorded by those APIs.
The current year's coverage can change as conferences publish and indexes update.

## Build the proceedings inventory (recommended)

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-webscraping.txt
.venv/bin/python build_corpus.py --start-year 2017 --end-year 2026
```

This adds records to the existing `data/references.sqlite3` and saves a one-time
SQLite backup at `data/references.before-roster.sqlite3` before the first run.
It enumerates each main proceedings volume in DBLP, plus official USENIX and
IACR CryptoDB paper lists, and the official CCS 2026 accepted-paper list while
that year's proceedings are not yet indexed. BeautifulSoup extracts titles and available abstracts
directly from the official HTML, including collapsed abstracts. No API keys are
needed for this stage. DBLP HTML currently blocks automated access, so DBLP
enumeration uses its explicitly supported public SPARQL API; HTML parser helpers
also support user-saved DBLP tables of contents. No browser challenge is bypassed.

The output is:

- `data/corpus/index.html`: searchable inventory with conference, year, and
  abstract-availability filters. Open it locally in a browser.
- `data/corpus/papers.csv`: every roster-listed paper, its title/authors/DOI,
  abstract text or absence, source, evidence URLs, and SQLite paper ID.
- `data/corpus/coverage.csv`: counts per conference/year.
- `data/corpus/coverage.json`: source-specific counts, status, notes, and URLs.
- SQLite `roster_entries` and `roster_runs`: the roster evidence and scan status.

The inventory counts **event years**, not merely API publication years. For
example, ASIACRYPT 2024 proceedings published in 2025 still belong to the 2024
conference. DBLP queries include all main volumes and verify each volume's
paper count. Workshops and companion volumes are excluded. Published entries
in the main proceedings can include posters or invited contributions; official
schedule-only talks, panels, and old-paper reprises are excluded.

“Source lists retrieved” means the indexed/published source was fully read,
so it cannot establish coverage of unpublished or unindexed papers.
Current-year lists are labeled provisional; unavailable sources are shown as
unavailable, not as verified zero-paper conferences. Source counts are exposed
so disagreements are visible. API-only records remain in the database but are
excluded from roster totals until matched to proceedings evidence.

Network pages are cached under `data/page_cache`, paced, and retried. Normal
reruns skip finished source scans. Use `--refresh` to revisit changing sources;
omit it again when resuming a stopped run. `--max-requests N` bounds requests.

```sh
# Regenerate the inventory without network requests.
.venv/bin/python build_corpus.py --report-only

# Revisit this year's official lists and DBLP entries.
.venv/bin/python build_corpus.py --start-year 2026 --end-year 2026 --refresh

# After proceedings collection, fill more missing abstracts using the two APIs.
python3 download_abstracts.py --phase enrich --start-year 2017 --end-year 2026
.venv/bin/python build_corpus.py --report-only
```

Missing official abstracts remain missing unless the API database has one.
For example, older IACR event lists can have every title but no abstract text.
DBLP itself is a bibliography and supplies identifiers, not abstracts. Matching
uses DOI/DBLP IDs or an unambiguous exact normalized title in the same conference
and event year. Reviewed title variants are recorded with reasons in
`reference_finder/roster_aliases.json`; arbitrary fuzzy matches are not merged.
Conflicting known DOIs are not merged on title alone.
Reviewed exclusions and unresolved source discrepancies are recorded in
`reference_finder/roster_review.json`. Unresolved entries remain listed with
review notes in the HTML and paper CSV; confirmed schedule-only items are excluded.

## Original API importer

From this directory:

```sh
# Recommended for a full import; the script also supports anonymous access.
export OPENALEX_API_KEY='your-openalex-key'
export SEMANTIC_SCHOLAR_API_KEY='your-semantic-scholar-key'

python3 download_abstracts.py --start-year 2017 --end-year 2026
```

The database is created at `data/references.sqlite3`. Keys come from environment
variables, are sent as request headers, and are not stored in SQLite. `.env.example`
documents their names; `.env` files are **not automatically loaded**.

Get an [OpenAlex key](https://openalex.org/settings/api) and
[request a Semantic Scholar key](https://www.semanticscholar.org/product/api).
Both APIs permit anonymous calls, but a full import is more reliable with keys.
OpenAlex provides a larger daily allowance with a free key; Semantic Scholar's
shared anonymous pool can return HTTP 429 even at low request rates. No paid
plan is required by this script. See the providers' current
[OpenAlex authentication](https://help.openalex.org/api/authentication/) and
[Semantic Scholar API](https://www.semanticscholar.org/product/api) documentation
for allowances and terms.

For a small trial:

```sh
python3 download_abstracts.py --venues sp --start-year 2025 --end-year 2025 --max-requests 10
```

Run the same command again to resume. Remove `--max-requests` to finish the
selected scope. The limit counts HTTP attempts, including retries. Ctrl-C also
preserves committed progress. Use a different `--db` path for an isolated trial.

## Included conferences

| CLI key | Conference |
| --- | --- |
| `ccs` | ACM Conference on Computer and Communications Security |
| `usenix-security` | USENIX Security Symposium |
| `sp` | IEEE Symposium on Security and Privacy |
| `crypto` | International Cryptology Conference (CRYPTO) |
| `eurosp` | IEEE European Symposium on Security and Privacy |
| `asiaccs` | ACM Asia Conference on Computer and Communications Security |
| `eurocrypt` | EUROCRYPT |
| `asiacrypt` | ASIACRYPT |

“USENIX” means USENIX Security here. Associated workshops, companion events,
and proceedings-volume/front-matter records are excluded where the metadata
identifies them. There are no invented Europe/Asia counterparts for conferences
without them. Adjust aliases in `reference_finder/venues.py` if providers change
their venue labels.

## Retrieval and completeness

1. Semantic Scholar's **bulk venue search** discovers papers, paginating by token
   for every selected conference/year. A paper must match a displayed or canonical
   venue label; alternate acronyms alone are not sufficient evidence.
2. OpenAlex source searches find conference sources using validated names.
   Cursor-paginated work searches include the conference source in **any location**.
   Broad series such as LNCS are never treated as a specific conference.
3. For papers still lacking an abstract, exact DOI lookups in OpenAlex and
   Semantic Scholar batch lookups fill gaps. Each paper's original provider has
   already been checked during discovery, so it is not immediately queried again.
4. SQLite stores metadata even when neither provider has an abstract, including
   raw provider JSON and the selected abstract's provenance. Coverage is printed
   for every conference/year, including zeros.

**These APIs cannot guarantee every paper or every abstract.** OpenAlex explicitly
[groups LNCS conference papers under the book series](https://help.openalex.org/data/sources/)
and does not identify each chapter's proceedings book. Semantic Scholar also has
indexing gaps and can omit abstracts at publishers' request. DOI enrichment can
fill an abstract only for a paper discovered by at least one API. Live checks
confirmed these limitations for cryptography conferences.

A completed import means all selected API queries finished. It does not establish
a complete proceedings corpus. Auditing that would require an authoritative
paper/DOI list from conference proceedings or another bibliography source.
The report's denominator is **discovered papers**, not all published papers.
Missing abstracts without DOIs are reported separately; no fuzzy title lookup is
used to fill them. Source metadata errors can still cause misclassification.

## Resume, refresh, and inspect

### Recovering from the original importer

The original filter rejected Semantic Scholar's `Book` tag even when a record
also had `Conference` or `JournalArticle`. This incorrectly dropped many real
ACM papers. The corrected filter accepts those mixed tags while still excluding
explicit proceedings-volume titles. Semantic Scholar discovery checkpoints now
use version 2, so the next normal run automatically rechecks the old pages once.
Existing papers remain and deduplicate; you do not need to delete the database
or pass `--refresh` for this repair.

```sh
python3 download_abstracts.py --start-year 2017 --end-year 2026
```

The report now separates unfinished API discovery from missing abstracts in
already discovered papers. A zero in the `No abstract` column does **not** mean
all conference papers were found. A provider failure can leave later venue/year
queries unstarted; rerunning resumes them. An API key helps with rate limiting
but cannot repair gaps in the provider's index.

The proceedings inventory enumerates papers from
[DBLP proceedings](https://dblp.org/db/conf/crypto/index.html) or official
conference lists before using OpenAlex/Semantic Scholar to enrich them by DOI or
other verified identifiers. For abstracts unavailable from either API,
[USENIX technical sessions](https://www.usenix.org/conference/usenixsecurity25/technical-sessions)
and [IACR CryptoDB](https://www.iacr.org/cryptodb/) provide direct sources.
`build_corpus.py` implements those sources. It reads official USENIX/IACR
abstracts even when no DOI exists; the separate API enrichment phase still
requires a DOI.

### Normal operation

Page records and pagination checkpoints commit in the **same transaction**, so a
failed or interrupted page is safe to repeat. Completed discovery queries and
successful/empty DOI checks are skipped on the next run. Transient network, 429,
and server errors use bounded exponential backoff and `Retry-After`.

```sh
# Revisit an evolving year and retry missing abstracts. Existing papers are kept.
python3 download_abstracts.py --start-year 2026 --end-year 2026 --refresh

# Fetch only missing abstracts for records already in the database.
python3 download_abstracts.py --phase enrich

# Retry previous empty DOI lookups without rediscovering papers.
python3 download_abstracts.py --phase enrich --refresh

# Use a single provider if the other is unavailable.
python3 download_abstracts.py --providers openalex

# Show stored coverage, with no network requests.
python3 download_abstracts.py --report-only

python3 download_abstracts.py --help
```

Use `--refresh` once when restarting a completed scan, then omit it on subsequent
resumes. Refresh rescans selected conference/year jobs and retries missing
abstracts; it does not delete previously imported papers or erase existing
abstracts when a provider now returns null. If you change venue matching rules
and need to remove previously misclassified papers, import into a new database.

Exit status: `0` selected stages finished (or report printed), `1` error/incomplete
stage, `2` requested HTTP limit reached, `130` interrupted. If a provider fails,
the other can still progress and the command reports a nonzero status. If the
server requests a wait longer than one minute, resume later. To change pacing,
use `--openalex-interval` or `--s2-interval` (seconds); the S2 default is 1.1 seconds.

## Database

| Table | Contents |
| --- | --- |
| `papers` | Deduplicated title, DOI, publication year/date, authors JSON, abstract, abstract source, URL |
| `paper_venues` | Conference/year membership and the provider that discovered it |
| `source_records` | Provider ID, original JSON, fetch time, and linked paper ID |
| `openalex_sources` | Matched OpenAlex source IDs and names |
| `checkpoints` | Discovery cursor, completion flag, fetched/accepted counts |
| `enrichment_attempts` | DOI lookup outcome and time, including missing abstracts |
| `roster_entries` | Proceedings evidence linking each source-listed paper to its conference/event year |
| `roster_runs` | Source completion, per-source counts, URLs, and notes |

Provider IDs and normalized DOIs deduplicate records. When a DOI is absent,
matching normalized title, conference, year, and first-author name can merge a
record only when the candidate is unambiguous; records with different known DOIs
are not merged by title. Later DOI evidence can join previously separate records.
Official USENIX/IACR abstracts take precedence, followed by Semantic Scholar and
OpenAlex; original provider records remain in `source_records`.

Example topic lookup using SQLite (simple text matching, ready for a future search UI):

```sql
SELECT p.id, p.title, p.publication_year, p.doi, p.abstract,
       GROUP_CONCAT(DISTINCT v.venue) AS venues
FROM papers p JOIN paper_venues v ON v.paper_id = p.id
WHERE p.title LIKE '%zero knowledge%' OR p.abstract LIKE '%zero knowledge%'
GROUP BY p.id
ORDER BY p.publication_year DESC;
```

Run `sqlite3 data/references.sqlite3` to query it interactively. Provider identifiers:

```sql
SELECT p.title, r.provider, r.external_id
FROM papers p JOIN source_records r ON r.paper_id = p.id;
```

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests use recorded-shaped synthetic records and mocked HTTP responses. They
cover pagination/resumption, transaction rollback, abstract reconstruction,
deduplication, venue filtering, DOI enrichment, retries, and credential-safe errors.
API references: [OpenAlex paging](https://help.openalex.org/api/paging/) and
[Semantic Scholar Graph API](https://api.semanticscholar.org/api-docs/graph).
