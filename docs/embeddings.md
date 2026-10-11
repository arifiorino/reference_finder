# Generate embeddings on a GPU server

`embed_papers.py` embeds every paper's title and available abstract, without a
keyword filter. It defaults to
[`nvidia/Nemotron-3-Embed-1B-BF16`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16).
Build the index on your H100, then copy the index to your Mac for local queries.
No paid API key is needed; the first model load downloads the public checkpoint.

The code is in Git. Databases, model downloads, credentials, and generated reports
are ignored. No GitHub or other hosted remote is assumed by these instructions.

## Transfer the code and paper text

On your Mac, from the project directory, export a compact SQLite corpus. The
export reads SQLite consistently even when the source uses WAL; do not copy a
live source database with a plain `cp` or `scp`.

If `data/transfer/corpus.sqlite3` already exists, skip the export to use that
snapshot, or choose a new filename for a fresh snapshot. Exports never overwrite
an existing file.

```sh
python3 embed_papers.py export \
  --db data/references.sqlite3 \
  --output data/transfer/corpus.sqlite3

# Include the current committed code and history in a portable Git bundle.
git bundle create data/transfer/reference_finder.bundle --all

# Replace USER@GPU_HOST with your server's SSH address.
ssh USER@GPU_HOST 'mkdir -p ~/reference-transfer'
scp data/transfer/reference_finder.bundle data/transfer/corpus.sqlite3 \
  USER@GPU_HOST:~/reference-transfer/
```

On the server:

```sh
git clone ~/reference-transfer/reference_finder.bundle ~/reference_finder
cd ~/reference_finder
mkdir -p data/transfer
cp ~/reference-transfer/corpus.sqlite3 data/transfer/corpus.sqlite3
```

A bundle is a portable local repository, not a hosted service. Its `origin` points
to the transferred bundle. To receive later changes, transfer a newly generated
bundle and fetch it, or configure a normal Git remote. Commit code changes before
creating a new bundle; uncommitted changes are not included.

## Install the GPU environment

Use Python 3.10 or later; Python 3.11 is a suitable default. If your GPU image
already includes working PyTorch with CUDA, keep that installation and install
the embedding requirements in its environment. Otherwise create an environment
and install the CUDA build of PyTorch recommended by the official
[PyTorch installer](https://pytorch.org/get-started/locally/) for your server.

```sh
# For a fresh environment; skip this if using an existing configured ML environment.
python3.11 -m venv .venv
source .venv/bin/activate

# Install the appropriate CUDA PyTorch wheel first using the official command.
python -m pip install -r requirements-embeddings.txt
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

`requirements-embeddings.txt` supplies Sentence Transformers, Transformers, and
the remaining dependencies. It does not replace PyTorch directly. Confirm that
the last command reports CUDA available before starting the H100 job.

## Benchmark, build, and resume

Run a 500-paper benchmark before the full job:

```sh
python embed_papers.py benchmark \
  --db data/transfer/corpus.sqlite3 \
  --device cuda --batch-size 32 --limit 500
```

The benchmark does not write an embedding index. Model download, loading, warmup,
text lengths, and batching affect runtime, so use its measured throughput for
your estimate. Processing 15,000 papers within an hour requires at least 4.2
papers/second, plus setup time. An H100 completion time has not been verified for
this project.

Build the index:

```sh
python embed_papers.py build \
  --db data/transfer/corpus.sqlite3 \
  --index data/embeddings.sqlite3 \
  --device cuda --batch-size 32

python embed_papers.py status --index data/embeddings.sqlite3
```

For a pilot that saves reusable progress, add `--limit 500` to `build`, then rerun
without the limit. An interrupted build resumes with the same command. Papers
whose input text changes are re-embedded. The index records the model checkpoint
revision and embedding configuration, and rejects incompatible resumes; use a
new index for a different model or chunking configuration.

CUDA uses BF16 automatically where supported, including H100. If `--device` is
omitted, the tool chooses CUDA when available and otherwise CPU. Lower
`--batch-size` if GPU memory is exhausted.

## Copy the index back and query on your Mac

After the build has finished, run this from the project directory on your Mac:

```sh
scp USER@GPU_HOST:~/reference_finder/data/embeddings.sqlite3 data/embeddings.sqlite3

# In your local Python 3.10+ environment:
python -m pip install torch
python -m pip install -r requirements-embeddings.txt

# Put your target title/abstract or research question in target.txt.
python embed_papers.py search \
  --index data/embeddings.sqlite3 \
  --query-file target.txt --device cpu --top-k 25
```

Add `--json` for machine-readable results. The self-contained index holds paper
metadata and normalized chunk vectors, so searching does not require the source
corpus database. Local CPU inference uses float32. Only the new query needs model
inference; similarity scoring against the stored vectors happens locally.

The index pins the checkpoint used to build it, and search uses that checkpoint
and matching query formatting. Do not mix vectors from different models or
checkpoint revisions, even if they have the same dimension. The first query on
your Mac downloads the model; later runs can reuse its local cache.
Queries must fit the token limit saved in the index, including the query prompt.
An overlong query raises an error instead of silently truncating it; shorten it
or run several focused queries.

## What the scores mean

- The default chunk limit is 2,048 tokens with 64 tokens of overlap. Long
  abstracts are split into chunks rather than silently truncated. These settings
  are recorded in the index.
- Corpus inputs use the model's document formatting; searches use its query
  formatting. The score for a paper is its highest chunk cosine similarity.
- Missing abstracts remain explicitly marked as title-only records. An embedding
  cannot recover their missing content.
- Similarity ranks papers for review. It is not a probability of relevance or a
  judgment that a paper makes the same contribution. Try several focused queries
  and inspect the actual abstracts, particularly for a novelty review.

Use `python embed_papers.py --help` and the corresponding subcommand's `--help`
for all options. Export and status do not require the optional ML dependencies.
