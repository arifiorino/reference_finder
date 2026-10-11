"""Portable corpus snapshots and resumable, model-pinned embedding indexes.

This module deliberately does not import a model framework. Its persistence and
coverage logic can be exercised without downloading weights or owning a GPU.
"""
from __future__ import annotations

from array import array
from contextlib import closing
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import time


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_INDEX = "data/qwen3-4b-embeddings.sqlite3"
FORMAT_VERSION = 1
LOG = logging.getLogger(__name__)
PAPER_FIELDS = ("id", "title", "abstract", "publication_year", "doi", "url", "authors_json")
PAPER_COLUMNS = """id INTEGER PRIMARY KEY, title TEXT NOT NULL, abstract TEXT,
    publication_year INTEGER, doi TEXT, url TEXT, authors_json TEXT NOT NULL,
    venues_json TEXT NOT NULL"""


def read_only(path):
    path = Path(path).expanduser().resolve()
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def read_corpus(path):
    """Read all papers from one SQLite snapshot, including API-only entries."""
    with closing(read_only(path)) as conn:
        conn.execute("BEGIN")
        columns = {r[1] for r in conn.execute("PRAGMA table_info(papers)")}
        if not {"id", "title", "abstract"} <= columns:
            raise ValueError("Corpus must contain a papers table with id, title and abstract")
        selected = [name if name in columns else "NULL AS " + name for name in PAPER_FIELDS]
        selected.append("venues_json" if "venues_json" in columns else "'[]' AS venues_json")
        rows = [dict(r) for r in conn.execute("SELECT " + ", ".join(selected) + " FROM papers ORDER BY id")]
        venues = {}
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_venues'").fetchone():
            for r in conn.execute("SELECT DISTINCT paper_id, venue, year FROM paper_venues ORDER BY paper_id,venue,year"):
                venues.setdefault(r[0], []).append({"venue": r[1], "year": r[2]})
        for row in rows:
            row["title"] = (row["title"] or "").strip()
            row["abstract"] = (row["abstract"] or "").strip() or None
            row["authors_json"] = row["authors_json"] or "[]"
            if row["id"] in venues:
                row["venues_json"] = json.dumps(venues[row["id"]], ensure_ascii=False)
            row["venues_json"] = row["venues_json"] or "[]"
            if not paper_text(row):
                raise ValueError("Paper %s has neither a title nor an abstract" % row["id"])
        return rows


def paper_text(paper):
    return (paper["title"] + "\n\n" + (paper["abstract"] or "")).strip()


def text_hash(paper):
    return hashlib.sha256(paper_text(paper).encode("utf-8")).hexdigest()


def export_corpus(source, output):
    """Export only paper metadata/text; never copy a live SQLite main file."""
    rows = read_corpus(source)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Export already exists; choose a new output path: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".corpus-", suffix=".sqlite3", dir=output.parent)
    os.close(handle)
    try:
        with closing(sqlite3.connect(temporary)) as conn:
            conn.execute("CREATE TABLE papers (" + PAPER_COLUMNS + ")")
            names = PAPER_FIELDS + ("venues_json",)
            conn.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?)",
                             [tuple(row[k] for k in names) for row in rows])
            conn.commit()
        # Linking in the same directory publishes atomically without overwriting.
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return {"output": str(output), "papers": len(rows),
            "abstracts": sum(bool(row["abstract"]) for row in rows)}


def load_config(path):
    with closing(read_only(path)) as conn:
        try:
            row = conn.execute("SELECT value FROM embedding_meta WHERE key='config'").fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError("Not an embedding index: %s" % path) from exc
    if row is None:
        raise ValueError("Embedding index has no configuration")
    config = json.loads(row[0])
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported embedding index format")
    return config


def encoder_config(encoder, max_tokens, overlap):
    return dict(format_version=FORMAT_VERSION, model_id=encoder.model_id,
                revision=encoder.revision, dimension=encoder.dimension,
                max_tokens=max_tokens, overlap=overlap,
                document_prompt=encoder.document_prompt, query_prompt=encoder.query_prompt,
                text_format="title-newline-newline-abstract-v1", normalization="l2",
                vector_format="float32-le", aggregation="max-chunk")


def open_index(path, config):
    path = Path(path).expanduser()
    if path.exists():
        previous = load_config(path)
        if previous != config:
            changed = sorted(k for k in set(previous) | set(config) if previous.get(k) != config.get(k))
            raise ValueError("Index settings differ (%s); use a new --index path" % ", ".join(changed))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    # DELETE journaling leaves one portable file when each transaction finishes.
    conn.execute("PRAGMA journal_mode=DELETE")
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS embedding_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        conn.execute("INSERT OR IGNORE INTO embedding_meta VALUES ('config',?)", (json.dumps(config, sort_keys=True),))
        conn.execute("CREATE TABLE IF NOT EXISTS papers (" + PAPER_COLUMNS + ", source_hash TEXT NOT NULL, embedded INTEGER NOT NULL DEFAULT 0)")
        conn.execute("""CREATE TABLE IF NOT EXISTS chunks (
            paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL,
            vector BLOB NOT NULL, PRIMARY KEY(paper_id,chunk_index))""")
    return conn


def sync_papers(conn, rows):
    """Invalidate changed text, refresh metadata, and remove absent source IDs."""
    old = {r["id"]: (r["source_hash"], r["embedded"]) for r in conn.execute("SELECT id,source_hash,embedded FROM papers")}
    ids = {p["id"] for p in rows}
    with conn:
        conn.executemany("DELETE FROM papers WHERE id=?", [(i,) for i in old.keys() - ids])
        for p in rows:
            digest = text_hash(p)
            same = p["id"] in old and old[p["id"]][0] == digest
            if not same:
                conn.execute("DELETE FROM chunks WHERE paper_id=?", (p["id"],))
            values = tuple(p[k] for k in PAPER_FIELDS + ("venues_json",))
            conn.execute("""INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET title=excluded.title,abstract=excluded.abstract,
                publication_year=excluded.publication_year,doi=excluded.doi,url=excluded.url,
                authors_json=excluded.authors_json,venues_json=excluded.venues_json,
                source_hash=excluded.source_hash,embedded=excluded.embedded""",
                         values + (digest, old[p["id"]][1] if same else 0))


def pack_vector(vector, dimension):
    values = [float(v) for v in vector]
    if len(values) != dimension or not all(math.isfinite(v) for v in values):
        raise ValueError("Model returned an invalid embedding dimension or non-finite values")
    length = math.hypot(*values)
    if length == 0:
        raise ValueError("Model returned a zero embedding")
    if not math.isfinite(length):
        raise ValueError("Model returned an embedding with an invalid norm")
    result = array("f", (v / length for v in values))
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def unpack_vector(blob):
    values = array("f")
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def index_status(path):
    config = load_config(path)
    with closing(read_only(path)) as conn:
        conn.execute("BEGIN")
        row = conn.execute("""SELECT COUNT(*) AS papers, COALESCE(SUM(abstract IS NOT NULL),0) AS abstracts,
            COALESCE(SUM(embedded),0) AS embedded,
            COALESCE(SUM(embedded=1 AND abstract IS NULL),0) AS embedded_title_only FROM papers""").fetchone()
        result = dict(row)
        result["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    result["pending"] = result["papers"] - result["embedded"]
    result["title_only"] = result["papers"] - result["abstracts"]
    result["config"] = config
    return result


def build_index(source, destination, encoder, config, batch_size=32, limit=None):
    if Path(source).expanduser().resolve() == Path(destination).expanduser().resolve():
        raise ValueError("Corpus and embedding index must be different files")
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Batch size and limit must be positive")
    rows = read_corpus(source)
    with closing(open_index(destination, config)) as conn:
        sync_papers(conn, rows)
        pending = [dict(r) for r in conn.execute("SELECT * FROM papers WHERE embedded=0 ORDER BY id")]
        selected = pending if limit is None else pending[:limit]
        LOG.info("%d papers in corpus, %d pending; embedding %d this run", len(rows), len(pending), len(selected))
        started = time.monotonic()
        done = 0
        for offset in range(0, len(selected), batch_size):
            group = selected[offset:offset + batch_size]
            prepared = [(p, encoder.document_chunks(paper_text(p), overlap=config["overlap"])) for p in group]
            flat = [chunk for _, chunks in prepared for chunk in chunks]
            if any(not chunks for _, chunks in prepared):
                raise ValueError("Tokenizer produced no chunks for a nonempty paper")
            vectors = encoder.encode_documents([c["text"] for c in flat], batch_size=batch_size)
            if len(vectors) != len(flat):
                raise ValueError("Model returned the wrong number of embeddings")
            # Validate before committing; all chunks of a paper commit together.
            blobs = [pack_vector(v, config["dimension"]) for v in vectors]
            position = 0
            with conn:
                for paper, chunks in prepared:
                    conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper["id"],))
                    for i, chunk in enumerate(chunks):
                        conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                                     (paper["id"], i, chunk["text"], chunk["token_count"], blobs[position]))
                        position += 1
                    conn.execute("UPDATE papers SET embedded=1 WHERE id=?", (paper["id"],))
            done += len(group)
            elapsed = max(time.monotonic() - started, 0.001)
            LOG.info("Committed %d/%d papers (%.1f papers/s; %.1f min remaining this run)",
                     done, len(selected), done / elapsed, (len(selected) - done) * elapsed / done / 60)
    return index_status(destination)


def benchmark(source, encoder, overlap=64, limit=500, batch_size=32):
    rows = read_corpus(source)
    if not rows or limit < 1 or batch_size < 1:
        raise ValueError("Benchmark needs a nonempty corpus and positive limit/batch size")
    sample = random.Random(0).sample(rows, min(limit, len(rows)))
    warmup = encoder.document_chunks(paper_text(sample[0]), overlap=overlap)
    encoder.encode_documents([warmup[0]["text"]], batch_size=1)
    encoder.synchronize()
    started = time.monotonic()
    chunks = [c for p in sample for c in encoder.document_chunks(paper_text(p), overlap=overlap)]
    encoder.encode_documents([c["text"] for c in chunks], batch_size=batch_size)
    encoder.synchronize()
    seconds = max(time.monotonic() - started, 1e-9)
    return dict(papers=len(sample), chunks=len(chunks), title_only=sum(not p["abstract"] for p in sample),
                tokens=sum(c["token_count"] for c in chunks), seconds=seconds,
                papers_per_second=len(sample) / seconds,
                projected_corpus_minutes=seconds * len(rows) / len(sample) / 60,
                corpus_papers=len(rows), device=encoder.device, dtype=encoder.dtype,
                model_id=encoder.model_id, revision=encoder.revision,
                note="Projection from a deterministic random sample after warmup; excludes model loading and index writes.")


def search_index(path, encoder, query, top_k=25, abstracts_only=False):
    if not query.strip() or top_k < 1:
        raise ValueError("Query must be nonempty and top-k must be positive")
    config = load_config(path)
    actual = encoder_config(encoder, config["max_tokens"], config["overlap"])
    if actual != config:
        raise ValueError("Query encoder does not match the index configuration")
    vector = encoder.encode_queries([query], batch_size=1)[0]
    packed = pack_vector(vector, config["dimension"])
    try:
        import numpy as np
    except ImportError:
        np = None
    q = np.frombuffer(packed, dtype="<f4") if np is not None else unpack_vector(packed)
    with closing(read_only(path)) as conn:
        conn.execute("BEGIN")
        candidates = conn.execute("""SELECT c.paper_id,c.chunk_index,c.text,c.vector FROM chunks c
            JOIN papers p ON p.id=c.paper_id WHERE p.embedded=1""" +
            (" AND p.abstract IS NOT NULL" if abstracts_only else "")).fetchall()
        if np is not None and candidates:
            matrix = np.vstack([np.frombuffer(r["vector"], dtype="<f4") for r in candidates])
            if matrix.shape[1] != config["dimension"]:
                raise ValueError("Corrupt index: vector dimension mismatch")
            scores = matrix @ q
        else:
            scores = []
            for row in candidates:
                values = unpack_vector(row["vector"])
                if len(values) != config["dimension"]:
                    raise ValueError("Corrupt index: vector dimension mismatch")
                scores.append(sum(a * b for a, b in zip(values, q)))
        best = {}
        for row, score in zip(candidates, scores):
            score = float(score)
            if row["paper_id"] not in best or score > best[row["paper_id"]]["score"]:
                best[row["paper_id"]] = dict(paper_id=row["paper_id"], score=score,
                    chunk_index=row["chunk_index"], matching_text=row["text"])
        ranked = sorted(best.values(), key=lambda r: (-r["score"], r["paper_id"]))[:top_k]
        for result in ranked:
            p = dict(conn.execute("SELECT * FROM papers WHERE id=?", (result["paper_id"],)).fetchone())
            result.update({k: p[k] for k in ("title", "abstract", "publication_year", "doi", "url")})
            result["has_abstract"] = bool(p["abstract"])
            result["authors"] = json.loads(p["authors_json"])
            result["venues"] = json.loads(p["venues_json"])
    return ranked
