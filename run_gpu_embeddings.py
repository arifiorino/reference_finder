#!/usr/bin/env python3
"""Prepare the latest repository papers and resume Qwen embeddings on a CUDA GPU."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile

from reference_finder.database_package import archive_files, archive_info, unpack_database
from reference_finder.embeddings import (
    DEFAULT_MODEL, FORMAT_VERSION, index_status, load_config, read_only, same_database,
)

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_verified(archive, destination, entry, expected_config=None):
    """Publish a verified restore without replacing an existing database."""
    if archive_info(archive)["archive_sha256"] != entry["archive_sha256"]:
        raise ValueError("Archive differs from snapshots/manifest.json: %s. Finish git pull before running." % archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gpu-restore-", dir=destination.parent) as temporary:
        restored = Path(temporary) / "restored.sqlite3"
        result = unpack_database(archive, restored)
        if result["snapshot_sha256"] != entry["sqlite_sha256"]:
            raise ValueError("Restored database differs from snapshots/manifest.json: %s" % archive)
        if expected_config is not None and load_config(restored) != expected_config:
            raise ValueError("Bundled index configuration differs from the snapshot manifest")
        # Same-directory hard links publish atomically and refuse an overwrite.
        os.link(restored, destination)


def prepare(snapshot_dir, work_dir, index):
    snapshot_dir, work_dir, index = (Path(p).expanduser().resolve() for p in (snapshot_dir, work_dir, index))
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {entry["archive"]: entry for entry in manifest["files"]}
    if len(entries) != len(manifest["files"]) or set(entries) != {"papers.sqlite3.xz", "embeddings.sqlite3.xz"}:
        raise ValueError("Manifest must describe one paper archive and one embedding archive")
    for entry in entries.values():
        for key in ("sqlite_sha256", "archive_sha256"):
            if not isinstance(entry.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", entry[key]):
                raise ValueError("Invalid %s in snapshot manifest" % key)
    papers_entry, embeddings_entry = entries["papers.sqlite3.xz"], entries["embeddings.sqlite3.xz"]
    expected = embeddings_entry["model_config"]
    if expected.get("model_id") != DEFAULT_MODEL or expected.get("format_version") != FORMAT_VERSION:
        raise ValueError("Pull the current repository snapshots containing the linked Qwen index")
    papers = work_dir / ("papers-%s.sqlite3" % papers_entry["sqlite_sha256"][:16])
    protected = [manifest_path, snapshot_dir / "papers.sqlite3.xz", snapshot_dir / "embeddings.sqlite3.xz", papers]
    for archive in (snapshot_dir / "papers.sqlite3.xz", snapshot_dir / "embeddings.sqlite3.xz"):
        if archive.exists() or Path(str(archive) + ".parts.json").exists():
            protected.extend(archive_files(archive))
    if any(same_database(index, path) for path in protected):
        raise ValueError("Embedding index must be separate from the paper database and repository snapshots")
    # Refuse incompatible existing work before restoring anything; never replace it.
    if index.exists() and load_config(index) != expected:
        raise ValueError("Existing index uses different model/settings: %s. Keep it and rerun with "
                         "--index data/embeddings-qwen.sqlite3 (or another unused path)." % index)
    if papers.exists():
        if any(Path(str(papers) + suffix).exists() and Path(str(papers) + suffix).stat().st_size
               for suffix in ("-wal", "-journal")):
            raise ValueError("Cached paper snapshot has a SQLite journal: %s. Close its writers and move "
                             "that copy aside before rerunning." % papers)
        if file_hash(papers) != papers_entry["sqlite_sha256"]:
            raise ValueError("Cached paper snapshot changed: %s. Move it aside and rerun to restore a verified copy." % papers)
    else:
        restore_verified(snapshot_dir / "papers.sqlite3.xz", papers, papers_entry)
    if not index.exists():
        restore_verified(snapshot_dir / "embeddings.sqlite3.xz", index, embeddings_entry, expected_config=expected)
    if load_config(index) != expected:
        raise ValueError("Restored index configuration differs from the snapshot manifest")
    with closing(read_only(index)) as conn:
        if [tuple(r) for r in conn.execute("PRAGMA integrity_check")] != [("ok",)] or conn.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Embedding database failed SQLite integrity checks: %s" % index)
    status = index_status(index, paper_db=papers)
    if status["missing_papers"]:
        raise ValueError("Index contains %d papers absent from the current repository snapshot. "
                         "Keep that index and choose an unused --index path to preserve those vectors." % status["missing_papers"])
    return papers, status


def require_cuda(dtype):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the GPU environment first: python -m pip install -r requirements-server.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Run on an allocated GPU node "
                           "with a CUDA-enabled PyTorch installation; use --prepare-only to check data without a GPU.")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16. Rerun with --dtype float16.")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def show_status(papers, index, status):
    print("Paper snapshot: %s" % papers, flush=True)
    print("Embedding index: %s" % index, flush=True)
    print("Papers: %s | Valid embeddings: %s | Pending: %s | Stale: %s" % tuple(
        format(status[key], ",") for key in ("papers", "embedded", "pending", "stale")), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data/embeddings.sqlite3",
                        help="existing matching Qwen index, or a new path to restore the bundled index")
    parser.add_argument("--batch-size", type=positive, default=16)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "auto"), default="bfloat16")
    parser.add_argument("--limit", type=positive, help="maximum pending papers for a trial; omit to finish")
    parser.add_argument("--prepare-only", action="store_true", help="restore/check data and print coverage without loading a model")
    args = parser.parse_args(argv)
    index = args.index.expanduser().resolve()
    try:
        papers, status = prepare(ROOT / "snapshots", ROOT / "data/server", index)
        show_status(papers, index, status)
        if args.prepare_only:
            return 0
        if status["pending"]:
            require_cuda(args.dtype)
            command = [sys.executable, str(ROOT / "embed_papers.py"), "build", "--db", str(papers),
                       "--index", str(index), "--device", "cuda", "--dtype", args.dtype,
                       "--batch-size", str(args.batch_size)]
            if args.limit:
                command += ["--limit", str(args.limit)]
            print("Building missing or changed embeddings; rerun this script to resume.", flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode:
                print("Build stopped. Committed batches are saved; rerun the same command to resume.", file=sys.stderr)
                return result.returncode if result.returncode > 0 else 1
            status = index_status(index, paper_db=papers)
            show_status(papers, index, status)
        if any(status[key] for key in ("pending", "stale", "missing_papers")):
            print("Partial build: rerun without --limit to finish.", flush=True)
            return 0 if args.limit else 1
        print("Complete: every paper in this snapshot has a valid embedding.", flush=True)
        command = [sys.executable, str(ROOT / "publish_embeddings.py"), "--db", str(papers),
                   "--index", str(index), "--snapshot-dir", str(ROOT / "snapshots")]
        printable = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        print("To package and verify the completed embeddings separately:\n" + printable, flush=True)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume saved batches.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, lzma.LZMAError, EOFError, RuntimeError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
