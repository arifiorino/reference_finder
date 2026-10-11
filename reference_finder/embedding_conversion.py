"""Convert saved embedding precision without model libraries or inference."""
from __future__ import annotations

from contextlib import closing
import json
import math
import sqlite3

from .database_package import _check_database, _paths, _publish, _temporary_directory
from .embeddings import (
    FORMAT_VERSION, index_status, load_config, pack_vector, read_only, unpack_vector,
)


def convert_index(index, output):
    """Publish a compact FP16 copy, preserving the source and all other values.

    SQLite backup takes a consistent snapshot, including committed WAL pages.
    Conversion and metadata updates share one transaction in a temporary copy;
    failures never publish a partially converted index or alter the source.
    """
    source, output = _paths(index, output)
    source_bytes = source.stat().st_size
    with _temporary_directory(output.parent) as directory:
        snapshot = directory / "converted.sqlite3"
        with closing(read_only(source)) as original, closing(sqlite3.connect(snapshot)) as copy:
            original.backup(copy, pages=256)
            copy.execute("PRAGMA journal_mode=DELETE")
        _check_database(snapshot)
        config = load_config(snapshot)
        if config["vector_format"] != "float32-le":
            raise ValueError("Index already uses float16 storage; no conversion is necessary")
        dimension = config["dimension"]
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("Embedding index has an invalid dimension")
        original_snapshot_bytes = snapshot.stat().st_size
        converted = 0
        old_vector_bytes = 0
        new_vector_bytes = 0
        with closing(sqlite3.connect(snapshot)) as copy:
            with copy:
                cursor = copy.execute("SELECT paper_id,chunk_index,vector FROM chunks ORDER BY paper_id,chunk_index")
                while True:
                    rows = cursor.fetchmany(256)
                    if not rows:
                        break
                    updates = []
                    for paper_id, chunk_index, blob in rows:
                        try:
                            values = unpack_vector(blob, "float32-le")
                            if len(values) != dimension or not all(math.isfinite(v) for v in values):
                                raise ValueError("invalid dimension or non-finite values")
                            norm = math.hypot(*values)
                            if not math.isfinite(norm) or not norm:
                                raise ValueError("invalid embedding norm")
                            packed = pack_vector(values, dimension, "float16-le")
                        except (ValueError, TypeError, OverflowError) as exc:
                            raise ValueError("Invalid vector for paper %s chunk %s: %s" %
                                             (paper_id, chunk_index, exc)) from exc
                        updates.append((packed, paper_id, chunk_index))
                        converted += 1
                        old_vector_bytes += len(blob)
                        new_vector_bytes += len(packed)
                    copy.executemany("UPDATE chunks SET vector=? WHERE paper_id=? AND chunk_index=?", updates)
                converted_config = dict(config, format_version=FORMAT_VERSION, vector_format="float16-le")
                copy.execute("UPDATE embedding_meta SET value=? WHERE key='config'",
                             (json.dumps(converted_config, sort_keys=True),))
            # Free the pages previously occupied by the larger float32 blobs.
            copy.execute("PRAGMA page_size=4096")
            copy.execute("VACUUM")
            if copy.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise ValueError("Converted index failed its foreign-key check")
        _check_database(snapshot)
        status = index_status(snapshot)
        result = dict(
            source=str(source), output=str(output), source_bytes=source_bytes,
            original_snapshot_bytes=original_snapshot_bytes,
            output_bytes=snapshot.stat().st_size, converted_chunks=converted,
            original_vector_bytes=old_vector_bytes, vector_bytes=new_vector_bytes,
            integrity_check="ok", **status,
        )
        _publish(snapshot, output)
        return result
