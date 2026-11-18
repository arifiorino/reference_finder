# Update the paper collection

The collector supports fifteen venues over the current year and preceding nine
years: CCS, USENIX Security, IEEE S&P, CRYPTO, EuroS&P, AsiaCCS, EUROCRYPT,
ASIACRYPT, PETS/PoPETs, NDSS, TCC, PKC, SIGMOD, VLDB/PVLDB, and ICDE. Paper text goes into
`data/papers.sqlite3`; collection does not modify `data/embeddings.sqlite3`.

Install the collection dependencies, then collect the three database venues:

```sh
python3 -m pip install -r requirements-webscraping.txt
python3 build_corpus.py --db data/papers.sqlite3 --venues sigmod vldb icde --start-year 2017 --end-year 2026
```

Omit `--venues` to collect all supported venues. Completed source/year jobs and
downloaded pages are reused on rerun. Use `--max-requests 100` to limit a run,
then rerun without `--refresh` to resume. Use `--refresh` when you want to check
updated source lists, particularly for the current year.

The additional venues use these sources:

| Venue key | Paper lists and abstracts |
| --- | --- |
| `pets` | All four issues of each PoPETs volume, with official paper-page abstracts; DBLP journal records cross-check the roster. PETS presentations are not counted again. |
| `ndss` | Official accepted-paper lists and programs, with individual paper pages for missing abstracts; DBLP main proceedings cross-check the roster. |
| `tcc` | DBLP main proceedings and IACR CryptoDB rosters and available abstracts; the official accepted-paper list supplies the 2026 papers and abstracts before DBLP indexing. |
| `pkc` | DBLP main proceedings and IACR CryptoDB rosters and available abstracts. |
| `sigmod` | DBLP main proceedings through 2022; explicit SIGMOD PACMMOD issues from 2023 onward, excluding PODS. Official 2024/2025 OpenTOCs supply abstracts. The 2025 OpenTOCs omit one SIGMOD issue, so DBLP supplies that part of the roster. Companion proceedings are excluded. |
| `vldb` | DBLP PVLDB articles grouped by publication year, verified against official PVLDB volume pages; available publisher abstracts are matched by paper identity. VLDB Journal articles are excluded. |
| `icde` | DBLP main ICDE proceedings, including papers, demos, and tutorials published there; workshops and companion proceedings are excluded. |

PVLDB publication years are used because journal volumes span calendar years and
papers can roll over to a later VLDB presentation year. Reports label this basis
explicitly; a PVLDB publication year is not an annual conference attendance list.
SIGMOD uses conference years and keeps each paper's publication year separately.

Editors' introductions and proceedings volumes are excluded. Source failures,
partial lists, and missing abstracts are reported separately; a completed
source scan does not guarantee that an evolving conference has published every
paper. If an official source is inaccessible, the other sources can still
collect paper metadata. For example, `--sources dblp pets ndss tcc` skips CryptoDB.

Fill remaining abstracts by DOI using the existing API clients:

```sh
python3 download_abstracts.py --db data/papers.sqlite3 --venues sigmod vldb icde --start-year 2017 --end-year 2026 --phase enrich
python3 build_corpus.py --db data/papers.sqlite3 --report-only
```

The clients read `OPENALEX_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` from the
environment when supplied. API availability and quotas can limit enrichment;
existing abstracts are retained. Counts and per-paper details are exported to
`data/corpus/coverage.csv`, `data/corpus/papers.csv`, and `data/corpus/index.html`.

If IACR is inaccessible or APIs lack abstracts, fetch the public publisher
abstracts for known TCC/PKC chapter DOIs:

```sh
python3 download_abstracts.py --db data/papers.sqlite3 --venues tcc pkc --start-year 2017 --end-year 2026 --phase enrich --providers springer
```

This optional fallback verifies the publisher DOI and title against the stored
paper. It retrieves HTML abstracts, without downloading full papers. Failed
requests remain retryable. Collection dependencies are required for this option.

After collecting, follow the [GPU embedding instructions](embeddings.md) to
transfer the updated paper database and embed the new records. Existing vectors
are reused when their paper text has not changed. Until that build finishes,
the new papers are available as text but are not included in embedding search.
