# Generate embeddings on a GPU and search locally

The project uses **two databases**:

| File | Contents | Used independently? |
| --- | --- | --- |
| `data/papers.sqlite3` | Titles, abstracts, citation metadata, collection progress | Yes: browsing, ChatGPT uploads, or GPU input |
| `data/embeddings.sqlite3` | FP16 vectors, paper IDs, text hashes, model settings | Search also needs the matching paper database |

The vector database never stores titles, abstracts, or chunk text. Paper IDs link
vectors to records; hashes check that the title and abstract still match the
input used to generate each embedding. Paths are supplied by the caller, so both
files can move between machines. There is no third corpus database.

Server and client use `Qwen/Qwen3-Embedding-4B` with the same saved checkpoint and
query formatting. Python 3.9–3.12, including 3.9.6, is supported. See the
[README](../README.md#go-deeper-with-embedding-search) for Mac and Windows setup.

## Sync code through Git

Push committed changes from your Mac:

```sh
git push origin main
```

On an existing GPU server checkout:

```sh
cd ~/reference_finder
git pull --ff-only origin main
```

For a first installation:

```sh
git clone git@github.com:arifiorino/reference_finder.git ~/reference_finder
cd ~/reference_finder
```

Git transfers code and committed archives in `snapshots/`. Working databases
under `data/` are ignored, so changes made to them do not travel with a push or
pull. Pull code updates between embedding jobs.

## Install the GPU environment

If you have a working CUDA environment, activate it and install the server
requirements. For a new environment:

```sh
python3 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install -r requirements-server.txt
python -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available())'
```

Use a CUDA build of PyTorch compatible with the server and Python version. Server
requirements allow PyTorch 2.6 or later in the 2.x series and retain an already
compatible installation. The client pins PyTorch 2.7.1. Both share Transformers
4.57.6, Sentence Transformers 5.1.2, and NumPy 1.26.4. Collection dependencies are
separate in `requirements-webscraping.txt`.

## Send paper data to the server

For the supplied paper set, restore the paper archive on the server:

```sh
python package_database.py unpack --archive snapshots/papers.sqlite3.xz --output data/papers.sqlite3
```

After pulling an updated archive, restore it again to a fresh temporary filename
if the working database already exists, then replace the old copy while database
users are stopped. Keep a backup if you collected additional papers locally.
Keep the existing matching Qwen embedding database: the build command below
reuses its valid vectors and generates only the missing or changed ones.

If you have newer papers on your Mac, package just that database and send it:

```sh
python3 package_database.py pack --db data/papers.sqlite3 --output data/papers-transfer.sqlite3.xz
scp data/papers-transfer.sqlite3.xz USER@GPU_HOST:~/reference_finder/data/
```

Restore the transferred archive on the server. Use a new temporary output path
if `data/papers.sqlite3` already exists, verify the incoming database, then replace
the old copy while database users are stopped. Packing includes committed WAL
contents, so it is safer than copying an open SQLite main file directly.

There is no corpus export. `build_corpus.py` and `download_abstracts.py` both write
to `data/papers.sqlite3` by default. The paper file contains no embeddings and
can also be uploaded by itself to a compatible ChatGPT session.

## Benchmark, build, and resume

A benchmark measures inference without saving vectors:

```sh
python embed_papers.py benchmark --db data/papers.sqlite3 --device cuda --dtype bfloat16 --batch-size 16 --limit 500
```

Build the separate vectors file:

```sh
python embed_papers.py build --db data/papers.sqlite3 --index data/embeddings.sqlite3 --device cuda --dtype bfloat16 --batch-size 16
python embed_papers.py status --db data/papers.sqlite3 --index data/embeddings.sqlite3
```

The paper database is read-only during embedding generation. Interrupted builds
resume committed batches. Add `--limit 500` for a pilot, then rerun without it to
finish. Changed titles or abstracts require new embeddings; unchanged text and
metadata-only edits reuse saved vectors. Search excludes stale or missing paper
matches rather than associating a vector with the wrong text.

The index pins the model revision, prompts, dimensions, chunk settings, and
vector storage format. Incompatible resumes are rejected. Check `pending` before
transferring a completed build. Restore `snapshots/embeddings.sqlite3.xz` if you
do not already have a Qwen index, then run the build to include any papers added
since that snapshot. Existing papers can be searched before the build finishes.

A batch size of 16 is a starting point. Increase it if the benchmark benefits and
memory permits. Automatic model precision uses BF16 on the H100, FP16 on Apple
MPS, and FP32 on CPU. H100 throughput has not been measured as part of this
update; benchmark it on your server.

## Recover an older Nemotron index

If the build reports that model settings differ, inspect the saved index:

```sh
python embed_papers.py status --db data/papers.sqlite3 --index data/embeddings.sqlite3
```

An index whose `model_id` is `nvidia/Nemotron-3-Embed-1B-BF16` contains older
Nemotron vectors. It cannot be resumed with Qwen3-Embedding-4B. Changing
`--dtype` does not convert vectors between models. For this legacy index format,
the reported paper count belongs to the index itself, so it does not establish
whether the separate paper database has been updated.

Pull the code and restore both current archives to fresh filenames, keeping
the old files intact:

```sh
git pull --ff-only origin main
python package_database.py unpack --archive snapshots/papers.sqlite3.xz --output data/papers-current.sqlite3
python package_database.py unpack --archive snapshots/embeddings.sqlite3.xz --output data/embeddings-qwen.sqlite3
python embed_papers.py status --db data/papers-current.sqlite3 --index data/embeddings-qwen.sqlite3
python embed_papers.py build --db data/papers-current.sqlite3 --index data/embeddings-qwen.sqlite3 --device cuda --dtype bfloat16 --batch-size 16
```

Use these same two paths for subsequent builds and publication. If either output
already exists, inspect it or choose another fresh filename; unpack never
overwrites an existing database. This reuses the bundled Qwen vectors and builds
only pending or changed papers. The November 8, 2026 paper update has 15,768
papers; its original Qwen snapshot has 12,425 vectors and 3,343 pending papers.

## Push completed embeddings through Git

After a build finishes, publish the compressed index and its matching metadata.
For the fresh filenames in the recovery instructions above:

```sh
python publish_embeddings.py --db data/papers-current.sqlite3 --index data/embeddings-qwen.sqlite3
git add snapshots/embeddings.sqlite3.xz snapshots/manifest.json
git commit -m "Update paper embeddings"
git push origin main
```

For the normal `data/papers.sqlite3` and `data/embeddings.sqlite3` paths, simply
run `python publish_embeddings.py` before the same Git commands. The publisher
requires complete embeddings for the paper snapshot already in the repository,
with matching model settings. It verifies a restored copy before updating the
archive and manifest; it does not commit or push anything itself. Working
databases under `data/` remain ignored by Git.
Each committed archive version remains in Git history, so repeated updates
increase the size of a full clone.

On the Mac, pull and restore the updated archive:

```sh
git pull --ff-only origin main
python3 package_database.py unpack --archive snapshots/embeddings.sqlite3.xz --output data/embeddings-incoming.sqlite3
python3 embed_papers.py status --db data/papers.sqlite3 --index data/embeddings-incoming.sqlite3
```

Check that `pending`, `stale`, and `missing_papers` are all zero. You can search
using `--index data/embeddings-incoming.sqlite3`, or replace the previous
`data/embeddings.sqlite3` with that file while searches are stopped. The paper
database on the Mac must be the same paper snapshot used for the server build.
A Git pull alone does not update unpacked databases.

## Bring only the embeddings back

After the server build finishes:

```sh
python package_database.py pack --db data/embeddings.sqlite3 --output data/embeddings-transfer.sqlite3.xz --page-size 1024
```

On the Mac, download and restore that archive:

```sh
scp USER@GPU_HOST:~/reference_finder/data/embeddings-transfer.sqlite3.xz data/embeddings-transfer.sqlite3.xz
python3 package_database.py unpack --archive data/embeddings-transfer.sqlite3.xz --output data/embeddings-incoming.sqlite3
python3 embed_papers.py status --db data/papers.sqlite3 --index data/embeddings-incoming.sqlite3
```

After verification, stop searches and replace the previous `data/embeddings.sqlite3`
with the incoming file. Choose a fresh temporary filename if one already exists.
The matching paper database is already on the Mac, so it does not need to travel
back. If the papers changed independently, send the updated paper database to
the server and build its missing or changed vectors before relying on coverage.

For the Git workflow, use [Push completed embeddings through Git](#push-completed-embeddings-through-git).

## Search on the client

Put your title and abstract in `target.txt`. On an Apple Silicon Mac:

```sh
.venv-qwen/bin/python embed_papers.py search --db data/papers.sqlite3 --index data/embeddings.sqlite3 --query-file target.txt --device mps --dtype float16 --top-k 50
```

In Windows PowerShell without a GPU:

```powershell
.\.venv-qwen\Scripts\python.exe -X utf8 embed_papers.py search --db data/papers.sqlite3 --index data/embeddings.sqlite3 --query-file target.txt --device cpu --dtype float32 --top-k 50
```

FP16 model weights require roughly 8 GB before inference overhead. CPU FP32
weights need roughly 16 GB; allow additional RAM for the OS and inference. FP16
vector storage does not reduce model weight memory. The first query downloads
the model; subsequent runs reuse the cache. BF16 server inference and FP16 client
inference share an embedding space with small numerical differences expected.

Add `--json` for titles, abstracts, citations, and scores. `--abstracts-only`
excludes title-only records. Queries must fit the saved context, including the
prompt; shorten an overlong query or use several focused queries.

## Storage and scores

Each 2,560-dimensional FP16 Qwen vector uses 5,120 bytes. Search promotes saved
vectors to FP32 and renormalizes them for cosine scoring. `--storage-dtype`
controls saved vector precision; `--dtype` controls model inference. XZ is
lossless packaging for the saved database.

Long papers use overlapping chunks (default 2,048 tokens and 64-token overlap).
Only vectors, chunk indices, and token counts are saved in the embedding file.
The score is the highest chunk cosine similarity; results retrieve the canonical
title and abstract from the paper database. The result is a candidate to read,
not a probability of relevance or proof of an identical research contribution.

The supplied legacy vectors can be migrated with
[`strip_embedding_text.py`](../strip_embedding_text.py) without model inference.
This changes the database layout; it does not convert Nemotron vectors into
Qwen vectors.
Old self-contained index formats remain readable, but new builds use the linked
layout. Full Qwen inference has been checked on Mac MPS; Windows and H100
execution have not been tested here.
