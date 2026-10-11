"""Remove duplicate paper text from existing vectors without model dependencies."""
from __future__ import annotations

from contextlib import closing
import json
import re
import sqlite3

from .database_package import (
    _check_database, _paths, _publish, _readonly, _sha256, _temporary_directory,
)
from .embeddings import load_config, paper_text, text_hash, unpack_vector


def _columns(connection, table):
    return [row[1] for row in connection.execute('PRAGMA table_info("%s")' % table)]


def _require_papers(connection):
    if not {"id", "title", "abstract"} <= set(_columns(connection, "papers")):
        raise ValueError("--db must contain a papers table with id, title and abstract")
    if connection.execute("SELECT 1 FROM papers GROUP BY id HAVING COUNT(*)>1 LIMIT 1").fetchone():
        raise ValueError("Paper database contains duplicate paper IDs")
    if _columns(connection, "embedding_meta"):
        raise ValueError("--db must be the paper database, not an embedding index")


def _require_legacy_index(connection, config):
    required = {
        "papers": {"id", "title", "abstract", "source_hash", "embedded"},
        "chunks": {"paper_id", "chunk_index", "text", "token_count", "vector"},
        "embedding_meta": {"key", "value"},
    }
    if any(not names <= set(_columns(connection, table)) for table, names in required.items()):
        raise ValueError("--index must be a legacy embedding index with saved text chunks")
    if (config.get("format_version"), config.get("vector_format")) not in (
        (1, "float32-le"), (2, "float16-le"),
    ):
        raise ValueError("Migration requires a legacy float32 or float16 embedding index")
    expected = {"text_format": "title-newline-newline-abstract-v1", "normalization": "l2",
                "aggregation": "max-chunk"}
    known_config = set(expected) | {"format_version", "vector_format", "dimension", "model_id", "revision",
                                   "max_tokens", "overlap", "document_prompt", "query_prompt"}
    if set(config) != known_config:
        raise ValueError("Unsupported embedding configuration fields")
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError("Unsupported embedding configuration: %s" % key)
    if not isinstance(config.get("model_id"), str) or not config["model_id"].strip():
        raise ValueError("Embedding configuration has no model identifier")
    if not isinstance(config.get("revision"), str) or not re.fullmatch(r"[a-fA-F0-9]{40}", config["revision"]):
        raise ValueError("Embedding configuration must pin a model commit revision")
    for key in ("document_prompt", "query_prompt"):
        if not isinstance(config.get(key), str):
            raise ValueError("Invalid embedding configuration: %s" % key)
    for key, minimum in (("max_tokens", 1), ("overlap", 0)):
        if type(config.get(key)) is not int or config[key] < minimum:
            raise ValueError("Invalid embedding configuration: %s" % key)
    if connection.execute("""SELECT 1 FROM chunks c LEFT JOIN papers p ON p.id=c.paper_id
        WHERE p.id IS NULL LIMIT 1""").fetchone():
        raise ValueError("Legacy index has chunks referring to missing papers")


def _normalized_paper(row):
    return {"title": (row["title"] or "").strip(),
            "abstract": (row["abstract"] or "").strip() or None}


def _validate_chunk_texts(text, chunks, paper_id):
    """Prove the saved chunks cover the source, including a possibly lost tail.

    Chunks are exact overlapping substrings in forward order. Retaining every
    possible start handles repeated passages without guessing an occurrence.
    """
    positions = [(-1, 0)]
    for chunk in chunks:
        value = chunk["text"]
        if not isinstance(value, str) or not value:
            raise ValueError("Invalid chunk text for paper %s" % paper_id)
        next_positions = []
        start = text.find(value)
        maximum = max(end for _, end in positions)
        while start != -1 and start <= maximum:
            if any(previous_start < start <= end for previous_start, end in positions):
                next_positions.append((start, start + len(value)))
            start = text.find(value, start + 1)
        if not next_positions:
            raise ValueError("Chunk text does not cover paper %s in order" % paper_id)
        positions = next_positions
    if not any(end == len(text) for _, end in positions):
        raise ValueError("Incomplete chunks for paper %s: source text tail is missing" % paper_id)


def _backup(source, destination):
    with closing(_readonly(source)) as original, closing(sqlite3.connect(destination)) as copy:
        original.backup(copy, pages=256)
        copy.execute("PRAGMA journal_mode=DELETE")
    _check_database(destination)


def strip_embedding_text(db, index, output):
    """Publish a linked vector database while leaving both inputs untouched.

    Read-only snapshots include committed WAL data. Every indexed paper must
    match the separate paper database by ID and encoded-text hash. Vector
    bytes are retained exactly; text is never written to the output database.
    """
    source, output = _paths(db, output)
    index, _ = _paths(index, output)
    if source == index:
        raise ValueError("Paper database and legacy index must be different files")
    source_bytes, index_bytes = source.stat().st_size, index.stat().st_size
    with _temporary_directory(output.parent) as directory:
        paper_snapshot = directory / "papers.sqlite3"
        legacy_snapshot = directory / "legacy-index.sqlite3"
        snapshot = directory / "embeddings.sqlite3"
        _backup(source, paper_snapshot)
        _backup(index, legacy_snapshot)
        with closing(_readonly(paper_snapshot)) as papers_connection:
            _require_papers(papers_connection)
        config = load_config(legacy_snapshot)
        with closing(_readonly(paper_snapshot)) as papers_connection, \
                closing(_readonly(legacy_snapshot)) as legacy, \
                closing(sqlite3.connect(snapshot)) as vectors:
            papers_connection.row_factory = sqlite3.Row
            legacy.row_factory = sqlite3.Row
            _require_legacy_index(legacy, config)
            if legacy.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("Legacy index failed its foreign-key check")
            converted_config = dict(config, format_version=3, storage_layout="linked-papers-v1")
            copied_chunks = copied_papers = vector_bytes = 0
            vectors.execute("PRAGMA foreign_keys=ON")
            vectors.execute("PRAGMA journal_mode=DELETE")
            vectors.executescript("""
                CREATE TABLE embedding_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE paper_state (
                    paper_id INTEGER PRIMARY KEY, source_hash TEXT NOT NULL,
                    embedded INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE chunks (
                    paper_id INTEGER NOT NULL REFERENCES paper_state(paper_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL, token_count INTEGER NOT NULL,
                    vector BLOB NOT NULL, PRIMARY KEY(paper_id,chunk_index)
                );
            """)
            with vectors:
                # Include papers added since the legacy index was generated as
                # pending work, without copying their titles or abstracts.
                for paper in papers_connection.execute("SELECT id,title,abstract FROM papers ORDER BY id"):
                    normalized = _normalized_paper(paper)
                    if type(paper["id"]) is not int or not paper_text(normalized):
                        raise ValueError("Invalid paper ID or empty paper text in the paper database")
                    vectors.execute("INSERT INTO paper_state VALUES (?,?,0)",
                                    (paper["id"], text_hash(normalized)))
                for row in legacy.execute("SELECT * FROM papers ORDER BY id"):
                    paper_id = row["id"]
                    current = vectors.execute("SELECT source_hash FROM paper_state WHERE paper_id=?", (paper_id,)).fetchone()
                    if current is None:
                        raise ValueError("Index paper %s is absent from the paper database" % paper_id)
                    normalized = _normalized_paper(row)
                    digest = text_hash(normalized)
                    if not paper_text(normalized) or row["source_hash"] != digest or current[0] != digest:
                        raise ValueError("Text hash mismatch for paper %s; re-embed changed text" % paper_id)
                    if row["embedded"] not in (0, 1):
                        raise ValueError("Invalid embedded state for paper %s" % paper_id)
                    chunks = legacy.execute("SELECT * FROM chunks WHERE paper_id=? ORDER BY chunk_index", (paper_id,)).fetchall()
                    if bool(row["embedded"]) != bool(chunks):
                        raise ValueError("Incomplete embedding state for paper %s" % paper_id)
                    if chunks:
                        if [chunk["chunk_index"] for chunk in chunks] != list(range(len(chunks))):
                            raise ValueError("Noncontiguous chunks for paper %s" % paper_id)
                        _validate_chunk_texts(paper_text(normalized), chunks, paper_id)
                    for chunk in chunks:
                        count = chunk["token_count"]
                        if type(count) is not int or not 0 < count <= config["max_tokens"]:
                            raise ValueError("Invalid token count for paper %s" % paper_id)
                        try:
                            blob = chunk["vector"]
                            values = unpack_vector(blob, config["vector_format"])
                            if len(values) != config["dimension"]:
                                raise ValueError("wrong embedding dimension")
                        except (ValueError, TypeError, OverflowError) as error:
                            raise ValueError("Invalid vector for paper %s chunk %s: %s" %
                                             (paper_id, chunk["chunk_index"], error)) from error
                        vectors.execute("INSERT INTO chunks VALUES (?,?,?,?)",
                                        (paper_id, chunk["chunk_index"], count, blob))
                        copied_chunks += 1
                        vector_bytes += len(blob)
                    vectors.execute("UPDATE paper_state SET embedded=? WHERE paper_id=?", (row["embedded"], paper_id))
                    copied_papers += bool(chunks)
                # Only model configuration belongs in the vector database;
                # arbitrary legacy metadata could contain another text copy.
                vectors.execute("INSERT INTO embedding_meta VALUES ('config',?)",
                                (json.dumps(converted_config, sort_keys=True),))
                if vectors.execute("PRAGMA foreign_key_check").fetchone():
                    raise ValueError("Linked index failed its foreign-key check")
            papers, abstracts = papers_connection.execute("SELECT COUNT(*),COUNT(abstract) FROM papers").fetchone()
            vectors.execute("PRAGMA page_size=1024")
            vectors.execute("VACUUM")
        _check_database(snapshot)
        result = {
            "source": str(source), "index": str(index), "output": str(output),
            "source_bytes": source_bytes, "index_bytes": index_bytes,
            "output_bytes": snapshot.stat().st_size, "output_sha256": _sha256(snapshot),
            "papers": papers, "abstracts": abstracts, "embedded": copied_papers,
            "pending": papers - copied_papers, "chunks": copied_chunks, "vector_bytes": vector_bytes,
            "config": converted_config, "integrity_check": "ok",
        }
        _publish(snapshot, output)
        return result
