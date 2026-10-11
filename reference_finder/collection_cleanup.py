"""Remove legacy raw responses from a collection copy, without API/model calls."""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from .database_package import (
    _check_database, _paths, _publish, _readonly, _sha256, _temporary_directory,
)
from .storage import Database


def _columns(connection, table):
    return {row[1] for row in connection.execute('PRAGMA table_info("%s")' % table)}


def _require_collection(connection):
    """Reject corpus/embedding files before the collection initializer can run."""
    required = {
        "papers": {"id", "title", "normalized_title", "abstract", "abstract_source", "created_at"},
        "source_records": {"provider", "external_id", "paper_id", "fetched_at"},
        "paper_venues": {"paper_id", "venue", "year", "provider"},
        "checkpoints": {"job", "cursor", "done", "fetched", "accepted", "updated_at"},
    }
    for table, columns in required.items():
        if not columns <= _columns(connection, table):
            raise ValueError("Input must be a collection database, not a corpus or embedding index")
    if not {"raw_json", "dblp_key"} & _columns(connection, "source_records"):
        raise ValueError("Collection source records have an unsupported schema")
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, 1, 2):
        raise ValueError("Unsupported collection schema version %s" % version)


def _raw_bytes(connection):
    return sum(
        connection.execute(
            'SELECT COALESCE(SUM(length(CAST(raw_json AS BLOB))),0) FROM "%s"' % table
        ).fetchone()[0]
        for table in ("source_records", "roster_entries")
        if "raw_json" in _columns(connection, table)
    )


def _paper_digest(connection):
    digest = hashlib.sha256()
    for row in connection.execute("SELECT * FROM papers ORDER BY id"):
        payload = json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.digest()


def _coverage(connection):
    return [
        {"venue": row[0], "year": row[1], "papers": row[2], "abstracts": row[3]}
        for row in connection.execute("""SELECT v.venue,v.year,COUNT(DISTINCT p.id),
            COUNT(DISTINCT CASE WHEN p.abstract IS NOT NULL THEN p.id END)
            FROM papers p JOIN paper_venues v ON p.id=v.paper_id
            GROUP BY v.venue,v.year ORDER BY v.venue,v.year""")
    ]


def prune_collection(db, output):
    """Publish a compact collection with canonical abstracts and no raw payloads.

    Read-only SQLite backup includes committed WAL contents without changing
    the source. Migration, checks, and VACUUM occur in a temporary copy; output
    is published only after paper contents and venue coverage are verified.
    """
    source, output = _paths(db, output)
    source_bytes = source.stat().st_size
    wal = Path(str(source) + "-wal")
    try:
        source_wal_bytes = wal.stat().st_size
    except FileNotFoundError:
        source_wal_bytes = 0
    with _temporary_directory(output.parent) as directory:
        snapshot = directory / "collection.sqlite3"
        with closing(_readonly(source)) as original, closing(sqlite3.connect(snapshot)) as copy:
            _require_collection(original)
            original.backup(copy, pages=256)
            copy.execute("PRAGMA journal_mode=DELETE")
        _check_database(snapshot)
        with closing(_readonly(snapshot)) as copy:
            _require_collection(copy)
            raw_bytes = _raw_bytes(copy)
            original_digest = _paper_digest(copy)
            original_coverage = _coverage(copy)
        original_snapshot_bytes = snapshot.stat().st_size
        database = Database(snapshot)
        database.conn.close()
        with closing(sqlite3.connect(snapshot)) as copy:
            if any("raw_json" in _columns(copy, table) for table in ("source_records", "roster_entries")):
                raise ValueError("Collection migration did not remove raw response columns")
            if _paper_digest(copy) != original_digest or _coverage(copy) != original_coverage:
                raise ValueError("Collection migration changed paper contents or venue coverage")
            copy.execute("PRAGMA journal_mode=DELETE")
            copy.execute("VACUUM")
            if copy.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise ValueError("Pruned collection failed its foreign-key check")
            papers, abstracts = copy.execute("SELECT COUNT(*),COUNT(abstract) FROM papers").fetchone()
            counts = {table: copy.execute('SELECT COUNT(*) FROM "%s"' % table).fetchone()[0]
                      for table in ("source_records", "roster_entries")}
            version = copy.execute("PRAGMA user_version").fetchone()[0]
        _check_database(snapshot)
        result = dict(
            source=str(source), output=str(output), source_bytes=source_bytes,
            source_wal_bytes=source_wal_bytes, original_snapshot_bytes=original_snapshot_bytes,
            output_bytes=snapshot.stat().st_size, removed_raw_json_bytes=raw_bytes,
            papers=papers, abstracts=abstracts, without_abstract=papers-abstracts,
            schema_version=version, integrity_check="ok", coverage=original_coverage,
            output_sha256=_sha256(snapshot), **counts,
        )
        _publish(snapshot, output)
        return result
