"""Lossless, verified SQLite snapshots and XZ archives using only the stdlib."""
from __future__ import annotations

from contextlib import closing, contextmanager
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile


PAGE_SIZES = tuple(2 ** power for power in range(9, 17))
COPY_BYTES = 1024 * 1024
ARCHIVE_PART_BYTES = 48 * 1024 * 1024


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


def _archive_layout(archive):
    """Resolve physical archive files without trusting descriptor paths."""
    archive = Path(archive).expanduser().resolve()
    descriptor = Path(str(archive) + ".parts.json")
    if os.path.lexists(archive):
        if os.path.lexists(descriptor):
            raise ValueError("Ambiguous archive: both single file and multipart descriptor exist: %s" % archive)
        if not archive.is_file():
            raise ValueError("Archive is not a file: %s" % archive)
        return archive, None, [archive]
    if not descriptor.is_file():
        raise ValueError("Archive or multipart descriptor does not exist: %s" % archive)
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "split-xz-v1":
        raise ValueError("Unsupported multipart archive descriptor")
    if (type(payload.get("archive_bytes")) is not int or payload["archive_bytes"] <= 0
            or not isinstance(payload.get("archive_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", payload["archive_sha256"])):
        raise ValueError("Invalid multipart archive size or SHA-256")
    parts = payload.get("parts")
    if not isinstance(parts, list) or not parts:
        raise ValueError("Multipart archive must have a nonempty parts list")
    paths = []
    for number, part in enumerate(parts, 1):
        expected_name = archive.name + ".part%03d" % number
        if (not isinstance(part, dict) or part.get("name") != expected_name
                or type(part.get("bytes")) is not int or part["bytes"] <= 0
                or not isinstance(part.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"])):
            raise ValueError("Invalid multipart archive part %d" % number)
        path = archive.parent / expected_name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or non-regular archive part: %s" % path)
        paths.append(path)
    if sum(part["bytes"] for part in parts) != payload["archive_bytes"]:
        raise ValueError("Multipart sizes do not add up to the archive size")
    return archive, payload, paths + [descriptor]


def archive_files(archive):
    """Return physical files for a single archive or validated part inventory.

    Multipart paths are numbered pieces followed by their JSON descriptor.
    Content hashes are checked by ``archive_info`` and ``unpack_database``.
    """
    return _archive_layout(archive)[2]


def _read_archive(archive, writer=None):
    """Hash or materialize the exact compressed stream, checking every part."""
    _, descriptor, paths = _archive_layout(archive)
    data_paths = paths[:-1] if descriptor else paths
    digest, total = hashlib.sha256(), 0
    for number, path in enumerate(data_paths):
        part_digest, part_bytes = hashlib.sha256(), 0
        with path.open("rb") as reader:
            for block in iter(lambda: reader.read(COPY_BYTES), b""):
                digest.update(block)
                part_digest.update(block)
                part_bytes += len(block)
                if writer is not None:
                    writer.write(block)
        total += part_bytes
        if descriptor:
            expected = descriptor["parts"][number]
            if part_bytes != expected["bytes"] or part_digest.hexdigest() != expected["sha256"]:
                raise ValueError("Archive part size or SHA-256 differs: %s" % path)
    result = {"archive_bytes": total, "archive_sha256": digest.hexdigest()}
    if descriptor:
        if total != descriptor["archive_bytes"] or result["archive_sha256"] != descriptor["archive_sha256"]:
            raise ValueError("Multipart archive total size or SHA-256 differs")
        result["parts"] = descriptor["parts"]
    return result


def archive_info(archive):
    """Verify and return compressed bytes/hash for either archive layout."""
    return _read_archive(archive)


def split_archive(source, output_base, part_bytes=ARCHIVE_PART_BYTES):
    """Split compressed bytes losslessly, refusing to overwrite any output.

    The logical output base remains absent. Its ``.parts.json`` descriptor and
    numbered ``.part001`` files together represent exactly the original bytes.
    """
    if type(part_bytes) is not int or part_bytes < 1:
        raise ValueError("part_bytes must be a positive integer")
    source, output = _paths(source, output_base)
    if any(output.parent.glob(output.name + ".part*")):
        raise FileExistsError("Multipart output already exists; choose a new path: %s" % output)
    if source.stat().st_size == 0:
        raise ValueError("Cannot split an empty archive")
    created = []
    digest, total, parts = hashlib.sha256(), 0, []
    try:
        with source.open("rb") as reader:
            while True:
                first = reader.read(min(COPY_BYTES, part_bytes))
                if not first:
                    break
                path = output.parent / (output.name + ".part%03d" % (len(parts) + 1))
                part_digest, count = hashlib.sha256(), 0
                with path.open("xb") as writer:
                    created.append(path)
                    block = first
                    while block:
                        writer.write(block)
                        part_digest.update(block)
                        digest.update(block)
                        count += len(block)
                        if count == part_bytes:
                            break
                        block = reader.read(min(COPY_BYTES, part_bytes - count))
                    writer.flush()
                    os.fsync(writer.fileno())
                total += count
                parts.append({"name": path.name, "bytes": count, "sha256": part_digest.hexdigest()})
        descriptor = {"format": "split-xz-v1", "archive_bytes": total,
                      "archive_sha256": digest.hexdigest(), "parts": parts}
        # A changing input must not create an apparently valid partial package.
        if source.stat().st_size != total or _sha256(source) != descriptor["archive_sha256"]:
            raise ValueError("Source archive changed during splitting")
        if os.path.lexists(output):
            raise FileExistsError("Logical archive output was created during splitting: %s" % output)
        descriptor_path = Path(str(output) + ".parts.json")
        with descriptor_path.open("x", encoding="utf-8") as writer:
            created.append(descriptor_path)
            json.dump(descriptor, writer, indent=2)
            writer.write("\n")
            writer.flush()
            os.fsync(writer.fileno())
        if archive_info(output) != {key: value for key, value in descriptor.items() if key != "format"}:
            raise ValueError("Split archive verification failed")
        return descriptor
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except OSError:
                pass
        raise


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
    """Restore a single or multipart XZ stream after verifying all its bytes."""
    archive, _, _ = _archive_layout(archive)
    output = Path(output).expanduser().absolute()
    if archive == output.resolve():
        raise ValueError("Input and output must be different files")
    if os.path.lexists(output):
        raise FileExistsError("Output already exists; choose a new path: %s" % output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_directory(output.parent) as directory:
        snapshot = directory / "snapshot.sqlite3"
        compressed = directory / "snapshot.sqlite3.xz"
        # Materialize the verified stream so later reads cannot race part edits.
        with compressed.open("xb") as writer:
            info = _read_archive(archive, writer)
        digest = hashlib.sha256()
        with lzma.open(compressed, "rb") as reader, snapshot.open("xb") as writer:
            for block in iter(lambda: reader.read(COPY_BYTES), b""):
                digest.update(block)
                writer.write(block)
        _check_database(snapshot)
        result = {
            "archive": str(archive),
            "output": str(output),
            "archive_bytes": info["archive_bytes"],
            "snapshot_bytes": snapshot.stat().st_size,
            "snapshot_sha256": digest.hexdigest(),
            "archive_sha256": info["archive_sha256"],
            "integrity_check": "ok",
        }
        _publish(snapshot, output)
        return result
