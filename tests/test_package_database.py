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
from reference_finder.database_package import (
    archive_files, archive_info, pack_database, split_archive, unpack_database,
)


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

    def make_split(self):
        packed = pack_database(self.source, self.archive, page_size=1024)
        base = self.root / "split" / "papers.sqlite3.xz"
        descriptor = split_archive(self.archive, base, part_bytes=127)
        return base, descriptor, packed

    def test_split_roundtrip_preserves_exact_archive_bytes_and_all_sqlite_values(self):
        base, descriptor, packed = self.make_split()
        self.assertFalse(base.exists())
        files = archive_files(base)
        self.assertGreater(len(descriptor["parts"]), 1)
        self.assertEqual(files[-1], Path(str(base) + ".parts.json").resolve())
        self.assertEqual(b"".join(path.read_bytes() for path in files[:-1]), self.archive.read_bytes())
        self.assertTrue(all(0 < part["bytes"] <= 127 for part in descriptor["parts"]))
        self.assertEqual(archive_info(base), {key: value for key, value in descriptor.items() if key != "format"})
        result = unpack_database(base, self.restored)
        self.assertEqual(result["archive_sha256"], packed["archive_sha256"])
        self.assertEqual(result["snapshot_sha256"], packed["snapshot_sha256"])
        self.assertEqual(result["archive_bytes"], self.archive.stat().st_size)
        self.assertEqual(self.contents(self.restored), self.contents(self.source))
        self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])

    def test_single_archive_info_and_file_inventory_remain_compatible(self):
        packed = pack_database(self.source, self.archive)
        self.assertEqual(archive_files(self.archive), [self.archive.resolve()])
        self.assertEqual(archive_info(self.archive), {key: packed[key] for key in ("archive_bytes", "archive_sha256")})

    def test_split_existing_outputs_are_preserved(self):
        base, _, _ = self.make_split()
        before = {path: path.read_bytes() for path in archive_files(base)}
        with self.assertRaises(FileExistsError):
            split_archive(self.archive, base, part_bytes=127)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.restored.write_bytes(b"keep existing output")
        with self.assertRaises(FileExistsError):
            unpack_database(base, self.restored)
        self.assertEqual(self.restored.read_bytes(), b"keep existing output")

    def test_split_refuses_partial_preexisting_inventory_without_creating_more_files(self):
        pack_database(self.source, self.archive)
        base = self.root / "split.sqlite3.xz"
        partial = Path(str(base) + ".part002")
        partial.write_bytes(b"existing partial upload")
        with self.assertRaises(FileExistsError):
            split_archive(self.archive, base, part_bytes=127)
        self.assertEqual(list(self.root.glob(base.name + ".part*")), [partial])
        self.assertEqual(partial.read_bytes(), b"existing partial upload")

    def test_ambiguous_single_and_split_files_are_rejected(self):
        base, _, _ = self.make_split()
        base.write_bytes(self.archive.read_bytes())
        for operation in (archive_files, archive_info, lambda path: unpack_database(path, self.restored)):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "Ambiguous"):
                operation(base)
        self.assertFalse(self.restored.exists())

    def test_missing_truncated_and_corrupted_parts_never_publish_a_database(self):
        base, _, _ = self.make_split()
        part = archive_files(base)[0]
        original = part.read_bytes()
        for replacement in (None, original[:-1], bytes([original[0] ^ 1]) + original[1:]):
            with self.subTest(replacement=None if replacement is None else len(replacement)):
                if replacement is None:
                    part.unlink()
                else:
                    part.write_bytes(replacement)
                with self.assertRaises(ValueError):
                    unpack_database(base, self.restored)
                self.assertFalse(self.restored.exists())
                self.assertEqual(list(self.root.glob(".sqlite-package-*")), [])
                part.write_bytes(original)

    def test_unsafe_or_invalid_descriptor_entries_are_rejected(self):
        base, descriptor, _ = self.make_split()
        path = Path(str(base) + ".parts.json")
        mutations = [
            lambda d: d.update(format="unknown"),
            lambda d: d.update(parts=[]),
            lambda d: d.update(archive_bytes=True),
            lambda d: d.update(archive_bytes=d["archive_bytes"] + 1),
            lambda d: d["parts"][0].update(name="../papers.sqlite3.xz"),
            lambda d: d["parts"][0].update(name=str(self.archive.resolve())),
            lambda d: d["parts"][0].update(name=base.name + ".part002"),
            lambda d: d["parts"][0].update(bytes=0),
            lambda d: d["parts"][0].update(bytes=True),
            lambda d: d["parts"][0].update(sha256="not a SHA-256"),
        ]
        for mutate in mutations:
            changed = json.loads(json.dumps(descriptor))
            mutate(changed)
            path.write_text(json.dumps(changed))
            with self.subTest(descriptor=changed), self.assertRaises(ValueError):
                unpack_database(base, self.restored)
            self.assertFalse(self.restored.exists())

    def test_overall_hash_is_checked_even_when_all_part_hashes_match(self):
        base, descriptor, _ = self.make_split()
        descriptor["archive_sha256"] = "0" * 64
        Path(str(base) + ".parts.json").write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(ValueError, "total size or SHA-256"):
            unpack_database(base, self.restored)
        self.assertFalse(self.restored.exists())

    def test_part_symlinks_cannot_redirect_archive_reads(self):
        base, _, _ = self.make_split()
        part = archive_files(base)[0]
        alternate = self.root / "outside-part"
        alternate.write_bytes(part.read_bytes())
        part.unlink()
        try:
            part.symlink_to(alternate)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are unavailable")
        with self.assertRaisesRegex(ValueError, "non-regular"):
            unpack_database(base, self.restored)
        self.assertFalse(self.restored.exists())

    def test_split_verification_failure_cleans_only_its_new_outputs(self):
        pack_database(self.source, self.archive)
        base = self.root / "split.sqlite3.xz"
        original = self.archive.read_bytes()
        with patch("reference_finder.database_package.archive_info", side_effect=ValueError("verification failed")):
            with self.assertRaisesRegex(ValueError, "verification failed"):
                split_archive(self.archive, base, part_bytes=127)
        self.assertFalse(base.exists())
        self.assertEqual(list(self.root.glob(base.name + ".part*")), [])
        self.assertEqual(self.archive.read_bytes(), original)

    def test_split_rejects_empty_input_and_invalid_chunk_sizes(self):
        self.archive.write_bytes(b"")
        base = self.root / "split.sqlite3.xz"
        for part_bytes in (0, -1, True, 1.5, 1):
            with self.subTest(part_bytes=part_bytes), self.assertRaises(ValueError):
                split_archive(self.archive, base, part_bytes=part_bytes)
        self.assertEqual(list(self.root.glob(base.name + ".part*")), [])

    def test_cli_accepts_the_missing_logical_base_of_a_split_archive(self):
        base, _, packed = self.make_split()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = package_database.main(["unpack", "--archive", str(base), "--output", str(self.restored)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["snapshot_sha256"], packed["snapshot_sha256"])


if __name__ == "__main__":
    unittest.main()
