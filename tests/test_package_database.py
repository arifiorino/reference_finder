"""Lossless SQLite archive checks, including live WALs and safe failures."""
from contextlib import closing, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import lzma
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import package_database
from reference_finder.database_package import pack_database, unpack_database


class DatabasePackageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "papers ü #%.sqlite3"
        self.archive = self.root / "papers.sqlite3.xz"
        self.restored = self.root / "restored.sqlite3"
        with closing(sqlite3.connect(self.source)) as connection:
            connection.executescript("""
                PRAGMA user_version=17;
                PRAGMA application_id=12345;
                CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE papers (id INTEGER PRIMARY KEY, title TEXT, vector BLOB);
                CREATE INDEX paper_titles ON papers(title);
                CREATE VIEW titles AS SELECT title FROM papers;
                INSERT INTO config VALUES ('model', 'Qwen/Qwen3-Embedding-4B');
            """)
            connection.execute("INSERT INTO papers VALUES (?,?,?)", (3, "证明 — résumé 🧪", bytes(range(256))))
            connection.execute("INSERT INTO papers VALUES (?,?,?)", (42, "No abstract", None))
            connection.commit()

    def contents(self, path):
        with closing(sqlite3.connect(path)) as connection:
            return {
                "schema": connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY name").fetchall(),
                "papers": connection.execute("SELECT * FROM papers ORDER BY id").fetchall(),
                "config": connection.execute("SELECT * FROM config ORDER BY key").fetchall(),
                "user_version": connection.execute("PRAGMA user_version").fetchone(),
                "application_id": connection.execute("PRAGMA application_id").fetchone(),
            }

    def test_roundtrip_preserves_schema_config_unicode_blobs_and_source(self):
        original_bytes = self.source.read_bytes()
        expected = self.contents(self.source)
        packed = pack_database(self.source, self.archive, page_size=1024)
        restored = unpack_database(self.archive, self.restored)
        self.assertEqual(self.contents(self.restored), expected)
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(packed["snapshot_sha256"], restored["snapshot_sha256"])
        self.assertEqual(packed["archive_sha256"], restored["archive_sha256"])
        self.assertEqual(restored["snapshot_sha256"], hashlib.sha256(self.restored.read_bytes()).hexdigest())
        self.assertLess(packed["archive_bytes"], packed["snapshot_bytes"])
        with closing(sqlite3.connect(self.restored)) as connection:
            self.assertEqual(connection.execute("PRAGMA page_size").fetchone(), (1024,))
        self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])

    def test_live_wal_snapshot_includes_committed_rows_without_checkpointing_source(self):
        with closing(sqlite3.connect(self.source)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("INSERT INTO papers VALUES (?,?,?)", (100, "Committed only in WAL", b"new vector"))
            writer.commit()
            main_before = self.source.read_bytes()
            wal = Path(str(self.source) + "-wal")
            wal_before = wal.read_bytes()
            packed = pack_database(self.source, self.archive)
            unpack_database(self.archive, self.restored)
            self.assertGreater(packed["source_wal_bytes"], 0)
            self.assertEqual(self.contents(self.restored), self.contents(self.source))
            self.assertEqual(self.source.read_bytes(), main_before)
            self.assertEqual(wal.read_bytes(), wal_before)
            self.assertFalse(Path(str(self.restored) + "-wal").exists())

    def test_existing_outputs_are_preserved_for_pack_and_unpack(self):
        self.archive.write_bytes(b"preserve archive")
        with self.assertRaises(FileExistsError):
            pack_database(self.source, self.archive)
        self.assertEqual(self.archive.read_bytes(), b"preserve archive")
        self.archive.unlink()
        pack_database(self.source, self.archive)
        self.restored.write_bytes(b"preserve restored database")
        with self.assertRaises(FileExistsError):
            unpack_database(self.archive, self.restored)
        self.assertEqual(self.restored.read_bytes(), b"preserve restored database")

    def test_same_path_is_rejected_without_mutation(self):
        original = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "different"):
            pack_database(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)

    def test_corrupt_or_non_database_archives_leave_no_output_or_temporary_files(self):
        for content in (b"not xz", lzma.compress(b""), lzma.compress(b"not a SQLite database")):
            with self.subTest(content=content[:6]):
                self.archive.write_bytes(content)
                with self.assertRaises((lzma.LZMAError, sqlite3.DatabaseError, ValueError, EOFError)):
                    unpack_database(self.archive, self.restored)
                self.assertFalse(self.restored.exists())
                self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])
        pack_database(self.source, self.root / "valid.xz")
        self.archive.write_bytes((self.root / "valid.xz").read_bytes()[:-12])
        with self.assertRaises((lzma.LZMAError, EOFError)):
            unpack_database(self.archive, self.restored)
        self.assertFalse(self.restored.exists())

    def test_publish_failure_removes_partial_output(self):
        pack_database(self.source, self.archive)

        def fail_copy(reader, writer, length):
            writer.write(b"partial")
            raise OSError("disk full")

        with patch("reference_finder.database_package.shutil.copyfileobj", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "disk full"):
                unpack_database(self.archive, self.restored)
        self.assertFalse(self.restored.exists())
        self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])

    def test_output_created_during_preparation_is_not_overwritten(self):
        pack_database(self.source, self.archive)
        from reference_finder import database_package
        original_check = database_package._check_database

        def concurrent_file(path):
            original_check(path)
            self.restored.write_bytes(b"created concurrently")

        with patch.object(database_package, "_check_database", side_effect=concurrent_file):
            with self.assertRaises(FileExistsError):
                unpack_database(self.archive, self.restored)
        self.assertEqual(self.restored.read_bytes(), b"created concurrently")

    def test_cli_reports_json_and_actionable_error(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = package_database.main(["pack", "--db", str(self.source), "--output", str(self.archive)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue())["integrity_check"], "ok")
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = package_database.main(["unpack", "--archive", str(self.archive), "--output", str(self.source)])
        self.assertEqual(result, 1)
        self.assertIn("Output already exists", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
