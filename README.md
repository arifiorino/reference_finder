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

## Find related papers with ChatGPT

Download and extract [papers.sqlite3.xz](snapshots/papers.sqlite3.xz), then upload
`papers.sqlite3` alongside your draft to a ChatGPT session that supports SQLite
files and Python. Ask it to review the titles and abstracts for related work and
explain how the closest papers compare with your idea.

## Go deeper with embedding search

For a more in-depth search, use Qwen3-Embedding-4B on your computer to compare
your title and abstract with the saved embeddings for the collection. You get a
ranked list of similar papers with scores and links; the paper embeddings are
already included, so you only need to run the search.

You need Git and Python 3.9–3.12. The first search downloads the model and may take
a while; subsequent searches reuse it. After setup, save your paper's title, a
blank line, and its abstract in a UTF-8 plain-text file named `target.txt` in the
`reference_finder` folder.

## Mac instructions

These instructions use an Apple Silicon Mac. Open Terminal and run the setup
commands below; if you already cloned the repository, open its folder and start
at `python3 -m venv`. Skip each unpack command if its output file already exists.

```sh
git clone https://github.com/arifiorino/reference_finder.git
cd reference_finder
python3 -m venv .venv-qwen
.venv-qwen/bin/python -m pip install -r requirements-client.txt
.venv-qwen/bin/python package_database.py unpack --archive snapshots/papers.sqlite3.xz --output data/papers.sqlite3
.venv-qwen/bin/python package_database.py unpack --archive snapshots/embeddings.sqlite3.xz --output data/embeddings.sqlite3
```

Create `target.txt` as described above, then search:

```sh
.venv-qwen/bin/python embed_papers.py search \
  --db data/papers.sqlite3 --index data/embeddings.sqlite3 \
  --query-file target.txt --device mps --top-k 50
```

For another paper, update `target.txt` and rerun only the search command. Change
`--top-k 50` to choose how many results to show.

## Windows PC instructions

These instructions work without a dedicated GPU; 32 GB of RAM is recommended.
Open PowerShell and run the setup commands below. They select 64-bit Python 3.11;
replace `-3.11` if you use another supported version. If you already cloned the
repository, open its folder and start at `py -3.11 -m venv`. Skip each unpack
command if its output file already exists.

```powershell
git clone https://github.com/arifiorino/reference_finder.git
cd reference_finder
py -3.11 -m venv .venv-qwen
.\.venv-qwen\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-qwen\Scripts\python.exe -m pip install -r requirements-client.txt
.\.venv-qwen\Scripts\python.exe package_database.py unpack --archive snapshots/papers.sqlite3.xz --output data/papers.sqlite3
.\.venv-qwen\Scripts\python.exe package_database.py unpack --archive snapshots/embeddings.sqlite3.xz --output data/embeddings.sqlite3
```

Create `target.txt` as described above, then search:

```powershell
.\.venv-qwen\Scripts\python.exe -X utf8 embed_papers.py search --db data/papers.sqlite3 --index data/embeddings.sqlite3 --query-file target.txt --device cpu --top-k 50
```

For another paper, update `target.txt` and rerun only the search command. Change
`--top-k 50` to choose how many results to show. Searches on a CPU will take longer
than on a GPU.
