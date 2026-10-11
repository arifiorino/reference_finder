"""Lossless, verified SQLite snapshots and XZ archives using only the stdlib."""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import lzma
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile


PAGE_SIZES = tuple(2 ** power for power in range(9, 17))
COPY_BYTES = 1024 * 1024


def _paths(source, output):
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if source == output.resolve():
        raise ValueError("Input and output must be different files")
    if not source.is_file():
        raise ValueError("Input file does not exist: %s" % source)
    # lexists also refuses to replace a dangling symlink.
    if os.path.lexists(output):
        raise FileExistsError("Output already exists; choose a new path: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return source, output


@contextmanager
def _temporary_directory(parent):
    directory = Path(tempfile.mkdtemp(prefix=".sqlite-package-", dir=parent))
    try:
        yield directory
    finally:
        # All database/file handles are closed before this point, including on
        # Windows. Cleanup failures must not hide the original operation error.
        shutil.rmtree(directory, ignore_errors=True)


def _readonly(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def _check_database(path):
    with path.open("rb") as source:
        if source.read(16) != b"SQLite format 3\x00":
            raise ValueError("File does not contain a SQLite 3 database header")
    with closing(_readonly(path)) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchall()
        if result != [("ok",)]:
            raise ValueError("SQLite integrity check failed: %r" % result[:5])


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(COPY_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(source, output):
    """Never overwrite a path, including one created after initial validation."""
    created = False
    try:
        with source.open("rb") as reader, output.open("xb") as writer:
            created = True
            shutil.copyfileobj(reader, writer, COPY_BYTES)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        if created:
            try:
                output.unlink()
            except OSError:
                pass
        raise


def pack_database(db, output, page_size=4096):
    """Back up a live database, compact the copy, and publish a verified archive.

    SQLite backup includes committed WAL contents. VACUUM changes physical
    layout (and potentially implicit rowids); table values, blobs, declared
    keys, schema, and application settings are retained without quantization.
    """
    if page_size not in PAGE_SIZES:
        raise ValueError("page_size must be a power of two between 512 and 65536")
    source, output = _paths(db, output)
    source_bytes = source.stat().st_size
    wal = Path(str(source) + "-wal")
    try:
        source_wal_bytes = wal.stat().st_size
    except FileNotFoundError:
        source_wal_bytes = 0
    with _temporary_directory(output.parent) as directory:
        snapshot = directory / "snapshot.sqlite3"
        archive = directory / "snapshot.sqlite3.xz"
        with closing(_readonly(source)) as original, closing(sqlite3.connect(snapshot)) as copy:
            original.backup(copy, pages=256)
            # A restored archive must not depend on any WAL/SHM companion files.
            copy.execute("PRAGMA journal_mode=DELETE")
            copy.execute("PRAGMA page_size=%d" % page_size)
            copy.execute("VACUUM")
        _check_database(snapshot)
        snapshot_sha256 = _sha256(snapshot)
        with snapshot.open("rb") as reader, lzma.open(
            archive, "wb", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=9
        ) as writer:
            shutil.copyfileobj(reader, writer, COPY_BYTES)
        verified = hashlib.sha256()
        with lzma.open(archive, "rb") as reader:
            for block in iter(lambda: reader.read(COPY_BYTES), b""):
                verified.update(block)
        if verified.hexdigest() != snapshot_sha256:
            raise ValueError("Archive verification failed: decompressed SHA-256 differs")
        result = {
            "source": str(source),
            "archive": str(output),
            "source_bytes": source_bytes,
            "source_wal_bytes": source_wal_bytes,
            "snapshot_bytes": snapshot.stat().st_size,
            "archive_bytes": archive.stat().st_size,
            "page_size": page_size,
            "snapshot_sha256": snapshot_sha256,
            "archive_sha256": _sha256(archive),
            "integrity_check": "ok",
        }
        _publish(archive, output)
        return result


def unpack_database(archive, output):
    """Verify an XZ stream and restored SQLite file before publishing a copy."""
    archive, output = _paths(archive, output)
    with _temporary_directory(output.parent) as directory:
        snapshot = directory / "snapshot.sqlite3"
        digest = hashlib.sha256()
        with lzma.open(archive, "rb") as reader, snapshot.open("xb") as writer:
            for block in iter(lambda: reader.read(COPY_BYTES), b""):
                digest.update(block)
                writer.write(block)
        _check_database(snapshot)
        result = {
            "archive": str(archive),
            "output": str(output),
            "archive_bytes": archive.stat().st_size,
            "snapshot_bytes": snapshot.stat().st_size,
            "snapshot_sha256": digest.hexdigest(),
            "archive_sha256": _sha256(archive),
            "integrity_check": "ok",
        }
        _publish(snapshot, output)
        return result
