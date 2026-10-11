# Reference Finder

**Find papers related to the research you are writing.** Reference Finder builds
a searchable collection of papers from leading security and cryptography
conferences over the past ten years. It gathers paper lists from conference
websites and proceedings, retrieves available abstracts through OpenAlex,
Semantic Scholar, and official conference pages, and uses
[`Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) to turn each
paper's title and abstract into a numerical representation of its meaning.
Provide your own title and abstract to find the most similar papers, helping you
discover related work even when authors use different terminology.

The configured collection covers **eight conferences**: CCS, USENIX Security,
IEEE S&P, CRYPTO, EuroS&P, AsiaCCS, EUROCRYPT, and ASIACRYPT. The default collection
window is the **current year plus the preceding nine years**: **2017–2026** when
run in 2026. Published source lists and abstract availability determine coverage;
current-year proceedings may still be incomplete. Missing abstracts are marked
explicitly, and those papers can still be searched by title.

## How it works

1. **Collect paper lists.** BeautifulSoup parses official conference pages;
   DBLP proceedings enumeration supplies additional paper and volume records.
2. **Retrieve abstracts.** OpenAlex and Semantic Scholar enrich the inventory,
   alongside abstracts available directly from official conference pages.
3. **Generate embeddings.** Qwen3-Embedding-4B embeds every title and available
   abstract, without a keyword prefilter. Vectors are saved in a separate database,
   linked by paper ID to the title and abstract in the paper database.
4. **Search with your paper.** The same model embeds your title and abstract,
   then ranks stored papers by cosine similarity. Only your query needs new
   model inference; existing paper embeddings are reused.

Long abstracts are split into overlapping chunks, and a paper's score is its
highest chunk similarity. Rankings help identify papers to read; a similarity
score is not a probability of relevance or a conclusion about novelty.

New indexes store vectors as **16-bit floats (FP16)** to reduce database size.
Search converts those values to FP32, renormalizes them, and computes cosine
similarity in FP32. This storage choice is separate from the model's inference
precision: `--dtype` still controls model execution, and CPU mode still needs
roughly 16 GB for FP32 model weights alone. Rounding can change close rankings;
FP16 storage is not a guarantee that every query returns an identical ordering.

## Search the databases

The project uses two files:

- **`data/papers.sqlite3`**: titles, abstracts, citation metadata, and collection progress.
- **`data/embeddings.sqlite3`**: FP16 vectors, paper IDs, text hashes, and model settings.

The embeddings database contains **no titles, abstracts, or chunk text**. Search
reads both files and checks the paper IDs and text hashes before using vectors.
File paths are supplied at runtime, so the databases can move between machines.

Create a UTF-8 `target.txt` containing your title, a blank line, and your abstract.
Client and server support **Python 3.9–3.12**, including 3.9.6. The first search
downloads the Qwen model; later runs reuse its local cache. Only your query needs
new inference. Search uses the checkpoint and formatting saved with the vectors.

### Mac with Apple Silicon

From the project directory, use a supported Python version to create the client
environment and install its dependencies:

```sh
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install -r requirements-client.txt

# Skip each unpack command if its output already exists.
.venv-qwen/bin/python package_database.py unpack --archive snapshots/papers.sqlite3.xz --output data/papers.sqlite3
.venv-qwen/bin/python package_database.py unpack --archive snapshots/embeddings.sqlite3.xz --output data/embeddings.sqlite3

.venv-qwen/bin/python embed_papers.py search \
  --db data/papers.sqlite3 --index data/embeddings.sqlite3 \
  --query-file target.txt \
  --device mps --dtype float16 --top-k 50
```

`mps` uses the Apple GPU. The model needs roughly 8 GB for FP16 weights alone,
plus inference memory and macOS; close memory-heavy applications on an 18 GB Mac.
For subsequent searches, run only the search command after updating `target.txt`.

### Windows PC without a GPU

Use 64-bit Python 3.9–3.12. These PowerShell commands select Python 3.11; replace
`-3.11` with your installed supported version. Calling the environment's Python
directly avoids PowerShell activation-policy changes.

```powershell
py -3.11 -m venv .venv-qwen
.\.venv-qwen\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-qwen\Scripts\python.exe -m pip install -r requirements-client.txt

# Skip each unpack command if its output already exists.
.\.venv-qwen\Scripts\python.exe package_database.py unpack --archive snapshots/papers.sqlite3.xz --output data/papers.sqlite3
.\.venv-qwen\Scripts\python.exe package_database.py unpack --archive snapshots/embeddings.sqlite3.xz --output data/embeddings.sqlite3

.\.venv-qwen\Scripts\python.exe -X utf8 embed_papers.py search --db data/papers.sqlite3 --index data/embeddings.sqlite3 --query-file target.txt --device cpu --dtype float32 --top-k 50
```

CPU mode needs roughly **16 GB for FP32 model weights alone**; **32 GB system RAM
is recommended** to leave room for Windows and inference. Model loading and
embedding the query are much slower than comparing the saved vectors. Generating
the full corpus on the GPU server is preferable. Windows execution has not been
tested on a Windows machine.

For a Windows PC with an NVIDIA GPU, install a compatible CUDA build of
[PyTorch 2.7.1](https://pytorch.org/get-started/previous-versions/) instead of the
CPU-only wheel, then install `requirements-client.txt`. Search with:

```powershell
.\.venv-qwen\Scripts\python.exe -X utf8 embed_papers.py search --db data/papers.sqlite3 --index data/embeddings.sqlite3 --query-file target.txt --device cuda --dtype auto --top-k 50
```

### Results and coverage

Results show titles, publication years, similarity scores, and paper links.
Title-only matches are labeled; add `--abstracts-only` to exclude them. Use
`--top-k` to choose the number of results. Inspect index coverage without loading
the model:

```sh
.venv-qwen/bin/python embed_papers.py status --db data/papers.sqlite3 --index data/embeddings.sqlite3
```

On Windows, use `.\.venv-qwen\Scripts\python.exe` in place of
`.venv-qwen/bin/python`.

Add `--json` to a search command to include full available abstracts and metadata
in machine-readable results. For example, on the Mac:

```sh
.venv-qwen/bin/python embed_papers.py search \
  --db data/papers.sqlite3 --index data/embeddings.sqlite3 \
  --query-file target.txt \
  --device mps --dtype float16 --top-k 50 --json > results.json
```

On Windows, set PowerShell's native-output encoding before saving JSON so older
PowerShell versions preserve non-ASCII titles:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
.\.venv-qwen\Scripts\python.exe -X utf8 embed_papers.py search --db data/papers.sqlite3 --index data/embeddings.sqlite3 --query-file target.txt --device cpu --dtype float32 --top-k 50 --json | Out-File -Encoding utf8 results.json
```

## Upload the paper database to ChatGPT

**Upload `papers.sqlite3` alongside your draft paper** if your ChatGPT session
accepts SQLite files and can run Python. It contains readable titles and
abstracts with no embeddings, so it is smaller and useful on its own. Decompress
`snapshots/papers.sqlite3.xz` first. File support depends on the session.

This upload lets ChatGPT inspect the paper text; it does not run Qwen similarity
search automatically. For the project's cosine rankings, run local search with
`--json`, then upload **`results.json` and your draft**. Ask it to distinguish
claims supported by abstracts from claims that require reading full papers.

## Two databases, with no duplicated abstracts

The supplied files contain **12,425 paper records**, **11,323 available abstracts**,
and **12,425 FP16 embeddings**. The paper database is useful independently for
ChatGPT uploads, browsing, or moving paper text to a GPU server. The embeddings
database stores only numerical vectors and the identifiers/settings needed to
interpret them. Search joins them by paper ID; text hashes prevent stale or
mismatched papers from being paired with vectors.

Collection writes to `papers.sqlite3`. Building reads that file and adds vectors
to `embeddings.sqlite3`. A changed title or abstract becomes pending and its
embedding is regenerated on the next build. Metadata-only edits do not require
new inference. Long papers can have several vectors, but their text is not
repeated in the vectors database.

The repository ships exactly two compressed snapshots:

| Archive | Restored SQLite | Compressed | Contents |
| --- | ---: | ---: | --- |
| `snapshots/papers.sqlite3.xz` | 39.7 MiB | 6.6 MiB | Paper text, citations, conference membership, and collection progress |
| `snapshots/embeddings.sqlite3.xz` | 63.3 MiB | 51.5 MiB | Vectors, paper IDs, hashes, and model configuration; no paper text |

There is no additional corpus database. Sizes, SHA-256 checksums, coverage, and
migration validation are recorded in [the manifest](snapshots/manifest.json).
Model weights are downloaded separately. SQLite files are uncompressed; XZ is
lossless packaging for transfer and Git. Older snapshots remain in Git history.

Migration preserved all paper records and all 12,425 vector blobs exactly.
Four queries encoded with the full Qwen model retained identical top-50 rankings,
scores, and paper metadata when searched against the split files.

To package updates, use a new output filename:

```sh
python package_database.py pack --db data/papers.sqlite3 --output data/papers-transfer.sqlite3.xz
python package_database.py pack --db data/embeddings.sqlite3 --output data/embeddings-transfer.sqlite3.xz --page-size 1024
```

Packing takes a consistent SQLite snapshot, including committed WAL data, and
checks integrity and archive checksums. Pack/unpack refuse to overwrite existing
outputs. Transfer only the paper database when the GPU server needs new paper
text; transfer only the embeddings database back after generation. Search needs
both matching files.

### Upgrade an older embedding database

The supplied snapshots already use the split layout. To remove duplicated text
from an older Qwen index without running the model:

```sh
python strip_embedding_text.py --db data/papers.sqlite3 --index data/qwen3-4b-embeddings.sqlite3 --output data/embeddings-new.sqlite3
python embed_papers.py status --db data/papers.sqlite3 --index data/embeddings-new.sqlite3
```

If your collection is still named `references.sqlite3`, use that path for `--db`
or rename it to `papers.sqlite3` after stopping database users. The migration
validates ID/text matches and copies vector bytes unchanged. It preserves its
inputs; after verification, replace the old vector database with the result and
remove the superseded copies. A separate `corpus.sqlite3` is no longer needed.

## Generate embeddings on a GPU server

The [embedding guide](docs/embeddings.md) covers Git push/pull, transferring the
small paper database to an H100, building the separate vectors database, and
searching on a Mac or Windows PC. No corpus-export step is needed.

Choose dependencies for the machine's role:

| Role | Requirements file | Python |
| --- | --- | --- |
| Mac or Windows client: embed a query and search an index | `requirements-client.txt` | 3.9–3.12, including 3.9.6 |
| GPU server: embed the paper corpus | `requirements-server.txt` | 3.9–3.12, including 3.9.6 |
| Collect paper lists and abstracts | `requirements-webscraping.txt` | 3.9+ |

Both embedding environments load the same model and share dependencies in
`requirements-embeddings.txt`. `requirements.txt` remains a legacy alias for the
webscraping dependencies; it does not install embedding dependencies.

## Build the proceedings inventory (recommended)

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-webscraping.txt
.venv/bin/python build_corpus.py --start-year 2017 --end-year 2026
```

This adds records to `data/papers.sqlite3`, the paper-text database. Embedding
generation reads it and writes vectors separately to `data/embeddings.sqlite3`.
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

The database is created at `data/papers.sqlite3`. Keys come from environment
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
   provider identifiers and the selected abstract's source. Raw response bodies
   are discarded. Coverage is printed for every conference/year, including zeros.

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

## Paper database

| Table | Contents |
| --- | --- |
| `papers` | Canonical title, abstract, authors, publication information, DOI, and URL |
| `paper_venues` | Conference/year membership and the provider that discovered it |
| `source_records` | Provider ID, DBLP matching key, fetch time, and linked paper ID; no response body |
| `openalex_sources` | Matched OpenAlex source IDs and names |
| `checkpoints` | Discovery cursor, completion flag, fetched/accepted counts |
| `enrichment_attempts` | DOI lookup outcome and time, including missing abstracts |
| `roster_entries` | Source-listed title, authors, DOI, URL, conference/year, and concise review notes; no response body |
| `roster_runs` | Source completion, per-source counts, URLs, and notes |

Provider IDs and normalized DOIs deduplicate records. When a DOI is absent,
matching normalized title, conference, year, and first-author name can merge a
record only when the candidate is unambiguous; records with different known DOIs
are not merged by title. Later DOI evidence can join previously separate records.
Official USENIX/IACR abstracts take precedence, followed by Semantic Scholar and
OpenAlex. Provider identifiers remain in `source_records`; raw responses are
discarded after extracting the fields needed by the collection.

Example topic lookup using SQLite (simple text matching, ready for a future search UI):

```sql
SELECT p.id, p.title, p.publication_year, p.doi, p.abstract,
       GROUP_CONCAT(DISTINCT v.venue) AS venues
FROM papers p JOIN paper_venues v ON v.paper_id = p.id
WHERE p.title LIKE '%zero knowledge%' OR p.abstract LIKE '%zero knowledge%'
GROUP BY p.id
ORDER BY p.publication_year DESC;
```

Run `sqlite3 data/papers.sqlite3` to query it interactively. Provider identifiers:

```sql
SELECT p.title, r.provider, r.external_id
FROM papers p JOIN source_records r ON r.paper_id = p.id;
```

The separate embeddings database has three tables: `paper_state` stores paper IDs,
text hashes, and completion state; `chunks` stores vectors and token counts;
`embedding_meta` stores the model configuration. None stores titles or abstracts.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests use recorded-shaped synthetic records and mocked HTTP responses. They
cover pagination/resumption, transaction rollback, abstract reconstruction,
deduplication, venue filtering, DOI enrichment, retries, and credential-safe errors.
API references: [OpenAlex paging](https://help.openalex.org/api/paging/) and
[Semantic Scholar Graph API](https://api.semanticscholar.org/api-docs/graph).
