#!/usr/bin/env python3
"""Verify completed linked embeddings and update the two files to commit to Git."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import itertools
import json
import lzma
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

from reference_finder.database_package import pack_database, unpack_database
from reference_finder.embeddings import (
    DEFAULT_INDEX, DEFAULT_MODEL, DEFAULT_PAPER_DB, FORMAT_VERSION,
    index_status, load_config, read_corpus, read_only, text_hash, unpack_vector,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def id_text_digest(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def corpus_digest(path):
    return id_text_digest([[row["id"], text_hash(row)] for row in read_corpus(path)])


def check_integrity(conn):
    require([tuple(row) for row in conn.execute("PRAGMA integrity_check")] == [("ok",)], "SQLite integrity check failed")
    require(conn.execute("PRAGMA foreign_key_check").fetchone() is None, "SQLite foreign-key check failed")


def inspect_index(conn, config):
    """Reject text-bearing schemas and malformed vectors, retaining audit hashes."""
    expected = {"embedding_meta": ["key", "value"],
                "paper_state": ["paper_id", "source_hash", "embedded"],
                "chunks": ["paper_id", "chunk_index", "token_count", "vector"]}
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    require(tables == set(expected), "Only the linked vector tables may be published; no paper text or extra tables")
    for table, columns in expected.items():
        require([r[1] for r in conn.execute("PRAGMA table_info(" + table + ")")] == columns,
                "Unexpected columns in %s; the embedding archive must contain no paper text" % table)
    check_integrity(conn)
    states = list(conn.execute("SELECT paper_id,source_hash,embedded FROM paper_state ORDER BY paper_id"))
    require(states and all(r[2] == 1 for r in states), "Every paper must have completed embeddings")
    counts = {table: conn.execute("SELECT COUNT(*) FROM " + table).fetchone()[0] for table in expected}
    vector_hash = hashlib.sha256()
    next_chunk = {}
    for row in conn.execute("SELECT paper_id,chunk_index,token_count,vector FROM chunks ORDER BY paper_id,chunk_index"):
        paper_id, chunk, tokens, blob = tuple(row)
        require(chunk == next_chunk.get(paper_id, 0), "Chunk indices must start at zero and be contiguous")
        require(type(tokens) is int and tokens > 0, "Invalid chunk token count")
        require(len(unpack_vector(blob, config["vector_format"])) == config["dimension"], "Wrong vector dimension")
        next_chunk[paper_id] = chunk + 1
        vector_hash.update(blob)
    require(set(next_chunk) == {r[0] for r in states}, "Every paper must have at least one vector")
    return counts, id_text_digest([[r[0], r[1]] for r in states]), vector_hash.hexdigest()


def verify_same_tables(original, restored):
    for table, order in (("embedding_meta", "key"), ("paper_state", "paper_id"), ("chunks", "paper_id,chunk_index")):
        query = "SELECT * FROM " + table + " ORDER BY " + order
        for left, right in itertools.zip_longest(original.execute(query), restored.execute(query)):
            require(left is not None and right is not None and tuple(left) == tuple(right),
                    "Restored embedding values differ in " + table)


def publish_embeddings(db=DEFAULT_PAPER_DB, index=DEFAULT_INDEX, snapshot_dir="snapshots"):
    db, index, directory = (Path(p).expanduser().resolve() for p in (db, index, snapshot_dir))
    manifest_path = directory / "manifest.json"
    paper_archive = directory / "papers.sqlite3.xz"
    archive_path = directory / "embeddings.sqlite3.xz"
    require(all(p.is_file() for p in (db, index, manifest_path, paper_archive, archive_path)), "Paper/index databases and existing snapshot files are required")
    require(not os.path.samefile(db, index), "Paper and embedding databases must be different files")
    outputs = (manifest_path, paper_archive, archive_path)
    require(not any(os.path.samefile(source, output) for source in (db, index) for output in outputs),
            "Input databases must not overlap snapshot output files")
    require(not any(os.path.samefile(left, right) for left, right in itertools.combinations(outputs, 2)),
            "Snapshot files must be distinct files")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = {entry["archive"]: entry for entry in manifest["files"]}
    require(len(entries) == len(manifest["files"]) and set(entries) == {"papers.sqlite3.xz", "embeddings.sqlite3.xz"},
            "Manifest must describe exactly one paper archive and one embedding archive")
    paper_entry, old_entry = entries["papers.sqlite3.xz"], entries["embeddings.sqlite3.xz"]
    require(file_hash(paper_archive) == paper_entry["archive_sha256"], "Paper archive differs from its manifest; restore the current repository snapshots")
    old_archive_hash = file_hash(archive_path)
    require(old_archive_hash == old_entry["archive_sha256"], "Embedding archive differs from its manifest")
    paper_digest = corpus_digest(db)
    require(paper_digest == paper_entry["paper_id_text_hash_sha256"],
            "Working paper database differs from the published paper archive; restore the matching paper archive before publishing")
    with closing(read_only(db)) as papers:
        check_integrity(papers)
    config = load_config(index)
    require(config["format_version"] == FORMAT_VERSION,
            "Publish requires a linked format-3 index; restore the bundled Qwen index and rebuild its pending papers")
    require(config["model_id"] == DEFAULT_MODEL and config == old_entry["model_config"],
            "Model/settings differ from the repository's Qwen index; restore the bundled index before resuming")
    status = index_status(index, paper_db=db)
    require(status["embedded"] > 0 and not any(status[k] for k in ("pending", "stale", "missing_papers")),
            "Embedding build is incomplete or mismatched; finish build and check status before publishing")

    with tempfile.TemporaryDirectory(prefix=".sqlite-package-publish-", dir=directory) as temporary:
        work = Path(temporary)
        packed_archive, restored_path = work / "embeddings.sqlite3.xz", work / "verified.sqlite3"
        with closing(read_only(index)) as original:
            original.execute("BEGIN")
            counts, state_digest, vector_digest = inspect_index(original, config)
            require(state_digest == paper_digest, "Embedded paper IDs/text hashes do not match the published papers")
            packed = pack_database(index, packed_archive, page_size=1024)
            unpack_database(packed_archive, restored_path)
            with closing(read_only(restored_path)) as restored:
                check_integrity(restored)
                verify_same_tables(original, restored)
                restored_vectors = hashlib.sha256()
                for row in restored.execute("SELECT vector FROM chunks ORDER BY paper_id,chunk_index"):
                    restored_vectors.update(row[0])
                require(restored_vectors.hexdigest() == vector_digest, "Restored vector digest differs")
        restored_status = index_status(restored_path, paper_db=db)
        require(restored_status == status and corpus_digest(db) == paper_digest,
                "Paper/index data changed while packaging; retry after all builds and collection stop")
        require(manifest_path.read_bytes() == manifest_bytes and file_hash(archive_path) == old_archive_hash
                and file_hash(paper_archive) == paper_entry["archive_sha256"],
                "Repository snapshots changed while packaging; retry after syncing")

        entry = dict(old_entry)
        entry.update(sqlite_bytes=packed["snapshot_bytes"], archive_bytes=packed["archive_bytes"],
                     sqlite_sha256=packed["snapshot_sha256"], archive_sha256=packed["archive_sha256"],
                     page_size=1024, integrity_check="ok", tables=counts, model_config=config,
                     paper_id_text_hash_sha256=state_digest, vector_sha256=vector_digest,
                     coverage={k: v for k, v in status.items() if k != "config"}, contains_paper_text=False)
        for field in ("coverage_scope", "paper_id_text_hash_scope"):
            entry.pop(field, None)
        entry["link"] = dict(entry.get("link", {}))
        entry["link"].pop("digest_relationship", None)
        manifest["files"] = [entry if item["archive"] == "embeddings.sqlite3.xz" else item for item in manifest["files"]]
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["validation"] = {
            "restored_embedding_table_values_verified": True,
            "paper_and_embedding_integrity_checks": "ok", "paper_and_embedding_foreign_key_checks": "ok",
            "vector_index_contains_no_paper_text": True, "paper_and_embedding_id_text_digests_match": True,
            "id_text_hash_digest_format": "SHA-256 of compact JSON [[paper_id, encoded_text_sha256], ...], ordered by paper_id",
            "vector_sha256_format": "SHA-256 of concatenated raw vector blobs, ordered by paper_id and chunk_index",
            "embedding_coverage": {"valid": status["embedded"], "pending": 0, "stale": 0, "missing_papers": 0},
        }
        manifest["collection_caveats"] = [note for note in manifest.get("collection_caveats", [])
                                            if not note.startswith("The embedding archive is unchanged and covers the original")]
        new_manifest = work / "manifest.json"
        with new_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        previous_archive = work / "previous.sqlite3.xz"
        try:
            os.link(archive_path, previous_archive)
        except OSError:
            shutil.copy2(archive_path, previous_archive)
        os.replace(packed_archive, archive_path)
        try:
            os.replace(new_manifest, manifest_path)
        except OSError:
            os.replace(previous_archive, archive_path)
            raise
    return {"archive": str(archive_path), "manifest": str(manifest_path),
            "papers": status["papers"], "embedded": status["embedded"], "pending": 0,
            "archive_bytes": packed["archive_bytes"], "archive_sha256": packed["archive_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_PAPER_DB)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--snapshot-dir", default="snapshots")
    args = parser.parse_args(argv)
    try:
        result = publish_embeddings(args.db, args.index, args.snapshot_dir)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, lzma.LZMAError, EOFError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
