# Generate embeddings on a GPU server

`embed_papers.py` embeds every paper's title and available abstract, without a
keyword filter. It defaults to
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
on both the server and client, using dependencies that support Python 3.9.6.
Build the index on your H100, then copy it to your Mac for local queries.
No paid API key is needed; the first model load downloads the public checkpoint.

**Switching from Nemotron requires a new build.** Keep the old index if desired,
but generate the Qwen index at `data/qwen3-4b-embeddings.sqlite3`. Query vectors
from Qwen cannot be compared to stored Nemotron vectors. The new filename is the
CLI default, so the previous `data/embeddings.sqlite3` is left in place.

The code is in Git. Databases, model downloads, credentials, and generated reports
are ignored. No GitHub or other hosted remote is assumed by these instructions.

## Transfer the code and paper text

On your Mac, from the project directory, export a compact SQLite corpus. The
export reads SQLite consistently even when the source uses WAL; do not copy a
live source database with a plain `cp` or `scp`.

If `data/transfer/corpus.sqlite3` already exists, skip the export to use that
snapshot, or choose a new filename for a fresh snapshot. Exports never overwrite
an existing file. Switching embedding models does not require re-exporting an
unchanged corpus.

```sh
python3 embed_papers.py export \
  --db data/references.sqlite3 \
  --output data/transfer/corpus.sqlite3

# Include the current committed code and history in a portable Git bundle.
git bundle create data/transfer/reference_finder.bundle main HEAD

# Replace USER@GPU_HOST with your server's SSH address.
ssh USER@GPU_HOST 'mkdir -p ~/reference-transfer'
scp data/transfer/reference_finder.bundle data/transfer/corpus.sqlite3 \
  USER@GPU_HOST:~/reference-transfer/
```

For a **first installation** on the server:

```sh
git clone ~/reference-transfer/reference_finder.bundle ~/reference_finder
cd ~/reference_finder
mkdir -p data/transfer
cp ~/reference-transfer/corpus.sqlite3 data/transfer/corpus.sqlite3
```

For an **existing server checkout**, transfer the new bundle as above and update
the code with:

```sh
cd ~/reference_finder
git fetch ~/reference-transfer/reference_finder.bundle main
git merge --ff-only FETCH_HEAD
```

If you also exported an updated corpus, copy that snapshot into the server's
`data/transfer/` directory after any active embedding job has stopped. The merge
preserves local work and will stop if the histories cannot fast-forward; it does
not discard server changes.

A bundle is a portable local repository, not a hosted service. Its `origin`
points to the transferred bundle. For later updates, transfer a fresh bundle and
repeat the fetch and merge, or configure a normal Git remote. Commit code changes
before creating a bundle; uncommitted changes are not included.

## Install the GPU environment

Use Python 3.9–3.12; Python 3.9.6 is supported. If you already have a working H100
environment, activate it and install the server requirements without `--upgrade`:

```sh
cd ~/reference_finder
python --version
python -m pip install -r requirements-server.txt
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

The server allows PyTorch 2.6 or later within the 2.x series, so an already
installed compatible CUDA build can be retained. Confirm that the last command
reports CUDA available before starting the H100 job.

For a fresh environment, create and activate a virtual environment first:

```sh
python3 -m venv .venv-server
source .venv-server/bin/activate
```

Then install a CUDA build of PyTorch compatible with your Python and server from
the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/),
followed by `python -m pip install -r requirements-server.txt`. Python 3.9 may
require a previous PyTorch release; the latest release need not support it.

Client and server share pinned Transformers 4.57.6, Sentence Transformers 5.1.2,
and NumPy 1.26.4 dependencies. Transformers provides the Qwen architecture and
tokenizer; Sentence Transformers loads the checkpoint's pooling and prompt
configuration. Documents use the checkpoint's document formatting; queries use
its retrieval instruction. Both machines load the same immutable model revision.
The webscraper's `requirements-webscraping.txt` remains separate.

The client environment was checked on Python 3.9.6 with real CPU and MPS/FP16
inference using a tiny random Qwen model. These smoke tests verify loading,
padding, pooling, and query formatting without downloading the 4B weights;
they do not measure the full model's memory use, speed, or retrieval quality.

## Benchmark, build, and resume

Run a 500-paper benchmark before the full job:

```sh
python embed_papers.py benchmark \
  --db data/transfer/corpus.sqlite3 \
  --device cuda --dtype bfloat16 --batch-size 16 --limit 500
```

The benchmark does not write an embedding index. Model download, loading, warmup,
text lengths, and batching affect runtime, so use its measured throughput for
your estimate. The earlier Nemotron throughput does not establish Qwen's speed.
Processing 15,000 papers within an hour requires at least 4.2 papers/second, plus
setup time. Qwen 4B inference on your H100 and Mac has not been tested as part of
this code update.

Build the new Qwen index:

```sh
python embed_papers.py build \
  --db data/transfer/corpus.sqlite3 \
  --index data/qwen3-4b-embeddings.sqlite3 \
  --device cuda --dtype bfloat16 --batch-size 16

python embed_papers.py status --index data/qwen3-4b-embeddings.sqlite3
```

Check that `pending` is `0` before transferring a completed index. For a pilot
that saves reusable progress, add `--limit 500` to `build`, then rerun without the
limit. An interrupted build resumes with the same command. Papers whose input
text changes are re-embedded. The index records the model checkpoint revision
and embedding configuration, and rejects incompatible resumes; use a new index
for a different model or chunking configuration.

A batch size of 16 is a starting point. Increase it if your benchmark shows a
benefit and enough GPU memory is available; lower it if memory is exhausted.
The H100 supports BF16. If `--device` is omitted, the tool chooses CUDA, then
Apple MPS, then CPU. Automatic precision is BF16 on the H100, FP16 on MPS, and
FP32 on CPU.

## Copy the index back and query on your Mac

After the build has finished, run this from the project directory on your Mac:

```sh
cd /Users/arifiorino/Developer/reference_finder
scp USER@GPU_HOST:~/reference_finder/data/qwen3-4b-embeddings.sqlite3 \
  data/qwen3-4b-embeddings.sqlite3
```

Create a separate client environment using your existing Python 3.9.6 or another
supported Python version. The client pins PyTorch 2.7.1, which supports Python
3.9 and Apple Silicon. No CUDA installation is needed on your Mac.

```sh
python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
python -m pip install -r requirements-client.txt

python embed_papers.py status --index data/qwen3-4b-embeddings.sqlite3

# Put your target title/abstract or research question in target.txt.
python embed_papers.py search \
  --index data/qwen3-4b-embeddings.sqlite3 \
  --query-file target.txt \
  --device mps --dtype float16 --top-k 50
```

For later sessions, run `source .venv-qwen/bin/activate` before searching. `mps`
uses your Apple Silicon GPU. FP16 weights alone require roughly 8 GB, with
additional memory for inference and macOS. Close memory-heavy applications on
your 18 GB Mac before loading the model. CPU with the default FP32 would require
roughly 16 GB for weights alone and may cause substantial swapping on this Mac.

Add `--json > results.json` to save machine-readable results. The self-contained
index holds paper metadata and normalized chunk vectors, so searching does not
require the source corpus database. Only the new query needs model inference;
similarity scoring against the stored vectors happens locally.

The index pins the checkpoint used to build it, and search uses that checkpoint
and matching query formatting. Do not mix vectors from different models or
checkpoint revisions, even if they have the same dimension. The first query on
your Mac downloads the model; later runs can reuse its local cache. Server BF16
and client FP16 inference use the same model and embedding space, with small
floating-point differences expected.

Queries must fit the token limit saved in the index, including the query prompt.
An overlong query raises an error instead of silently truncating it; shorten it
or run several focused queries.

## What the scores mean

- The default chunk limit is 2,048 tokens with 64 tokens of overlap. Long
  abstracts are split into chunks rather than silently truncated. These settings
  are recorded in the index.
- The score for a paper is its highest chunk cosine similarity. Higher scores
  indicate more similar embeddings.
- Missing abstracts remain explicitly marked as title-only records. An embedding
  cannot recover their missing content.
- Similarity ranks papers for review. It is not a probability of relevance or a
  judgment that a paper makes the same contribution. Try several focused queries
  and inspect the actual abstracts, particularly for a novelty review.

Use `python embed_papers.py --help` and the corresponding subcommand's `--help`
for all options. Export and status do not require the optional ML dependencies.
