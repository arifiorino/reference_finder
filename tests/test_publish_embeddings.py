"""Publish only verified, complete indexes linked to the repository's paper set."""
from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch

import publish_embeddings as publisher
from reference_finder.database_package import pack_database, unpack_database
from reference_finder.embeddings import encoder_config, open_index, pack_vector, read_corpus, sync_paper_state


class Encoder:
    model_id = "Qwen/Qwen3-Embedding-4B"
    revision = "a" * 40
    dimension = 2
    document_prompt = ""
    query_prompt = "Instruct: retrieve\nQuery:"


class PublishEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "papers.sqlite3"
        self.index = self.root / "embeddings.sqlite3"
        self.directory = self.root / "snapshots"
        self.directory.mkdir()
        self.archive = self.directory / "embeddings.sqlite3.xz"
        self.paper_archive = self.directory / "papers.sqlite3.xz"
        self.manifest_path = self.directory / "manifest.json"
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY,title TEXT,abstract TEXT)")
            conn.executemany("INSERT INTO papers VALUES (?,?,?)", [(1, "First research paper", "Its abstract."),
                                                                  (3, "Paper with no abstract", None)])
            conn.commit()
        self.rows = read_corpus(self.db)
        self.config = encoder_config(Encoder(), 2048, 64)
        self.make_index(self.index, self.rows)
        old_index = self.root / "old.sqlite3"
        self.make_index(old_index, self.rows[:1])
        old = pack_database(old_index, self.archive, page_size=1024)
        papers = pack_database(self.db, self.paper_archive, page_size=1024)
        self.paper_entry = {"archive": "papers.sqlite3.xz", "archive_sha256": papers["archive_sha256"],
                            "paper_id_text_hash_sha256": publisher.corpus_digest(self.db),
                            "coverage": {"papers": 2, "abstracts": 1, "title_only": 1}}
        self.manifest = {"updated_at": "old", "layout": "linked-papers-v1",
                         "validation": {"original_vectors_unchanged": True, "paper_and_embedding_digests_differ_by_design": True},
                         "collection_caveats": ["Current-year lists may change.",
                                                "The embedding archive is unchanged and covers the original 1 papers."],
                         "files": [self.paper_entry, {
                             "archive": "embeddings.sqlite3.xz", "archive_sha256": old["archive_sha256"],
                             "model_config": self.config, "coverage": {"papers": 2, "embedded": 1, "pending": 1},
                             "coverage_scope": "Only old papers", "paper_id_text_hash_scope": "Stored subset",
                             "link": {"database": "papers.sqlite3", "key": "paper_state.paper_id = papers.id",
                                      "digest_relationship": "Old subset"}}]}
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.before = {p: p.read_bytes() for p in (self.archive, self.paper_archive, self.manifest_path)}

    def make_index(self, path, rows):
        with closing(open_index(path, self.config)) as conn:
            sync_paper_state(conn, rows)
            with conn:
                for row in rows:
                    conn.execute("INSERT INTO chunks VALUES (?,?,?,?)", (row["id"], 0, 12, pack_vector([1, row["id"]], 2)))
                    conn.execute("UPDATE paper_state SET embedded=1 WHERE paper_id=?", (row["id"],))

    def publish(self):
        return publisher.publish_embeddings(self.db, self.index, self.directory)

    def assert_unchanged(self):
        for path, contents in self.before.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(list(self.directory.glob(".sqlite-package-*")), [])

    def test_complete_linked_index_updates_archive_and_metadata_only(self):
        inputs = {p: p.read_bytes() for p in (self.db, self.index)}
        result = self.publish()
        self.assertEqual((result["papers"], result["embedded"], result["pending"]), (2, 2, 0))
        for path, contents in inputs.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(self.paper_archive.read_bytes(), self.before[self.paper_archive])
        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(manifest["files"][0], self.paper_entry)
        saved = manifest["files"][1]
        self.assertEqual(saved["archive_sha256"], publisher.file_hash(self.archive))
        self.assertEqual(saved["paper_id_text_hash_sha256"], self.paper_entry["paper_id_text_hash_sha256"])
        self.assertEqual(saved["coverage"]["pending"], 0)
        self.assertEqual(saved["coverage"]["embedded"], 2)
        self.assertFalse(saved["contains_paper_text"])
        self.assertNotIn("coverage_scope", saved)
        self.assertNotIn("digest_relationship", saved["link"])
        self.assertNotIn("original_vectors_unchanged", manifest["validation"])
        self.assertEqual(manifest["collection_caveats"], ["Current-year lists may change."])
        restored = self.root / "restored.sqlite3"
        unpack_database(self.archive, restored)
        with closing(sqlite3.connect(self.index)) as source, closing(sqlite3.connect(restored)) as actual:
            publisher.verify_same_tables(source, actual)
            self.assertEqual(actual.execute("PRAGMA page_size").fetchone()[0], 1024)
        self.assertEqual(publisher.file_hash(restored), saved["sqlite_sha256"])
        self.assertEqual(list(self.directory.glob(".sqlite-package-*")), [])

    def test_incomplete_index_is_rejected_before_archive_changes(self):
        with sqlite3.connect(self.index) as conn:
            conn.execute("DELETE FROM chunks WHERE paper_id=3")
            conn.execute("UPDATE paper_state SET embedded=0 WHERE paper_id=3")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.publish()
        self.assert_unchanged()

    def test_working_papers_must_match_published_archive(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE papers SET abstract='Changed abstract' WHERE id=1")
        with self.assertRaisesRegex(ValueError, "Working paper database differs"):
            self.publish()
        self.assert_unchanged()

    def test_stale_index_is_rejected(self):
        with sqlite3.connect(self.index) as conn:
            conn.execute("UPDATE paper_state SET source_hash='old' WHERE paper_id=1")
        with self.assertRaisesRegex(ValueError, "incomplete or mismatched"):
            self.publish()
        self.assert_unchanged()

    def test_wrong_model_and_legacy_index_are_rejected(self):
        for changed, message in ((dict(self.config, model_id="nvidia/Nemotron-3-Embed-1B-BF16"), "Model/settings"),
                                 (dict(self.config, format_version=2), "format-3")):
            with self.subTest(config=changed), sqlite3.connect(self.index) as conn:
                conn.execute("UPDATE embedding_meta SET value=? WHERE key='config'", (json.dumps(changed),))
            with self.assertRaisesRegex(ValueError, message):
                self.publish()
            self.assert_unchanged()

    def test_text_bearing_index_cannot_be_published(self):
        with sqlite3.connect(self.index) as conn:
            conn.execute("ALTER TABLE chunks ADD COLUMN text TEXT")
            conn.execute("UPDATE chunks SET text='Copied abstract'")
        with self.assertRaisesRegex(ValueError, "no paper text"):
            self.publish()
        self.assert_unchanged()

    def test_corrupted_archive_manifest_pair_is_rejected(self):
        self.paper_archive.write_bytes(b"changed")
        expected = self.paper_archive.read_bytes()
        with self.assertRaisesRegex(ValueError, "Paper archive differs"):
            self.publish()
        self.assertEqual(self.paper_archive.read_bytes(), expected)
        for path in (self.archive, self.manifest_path):
            self.assertEqual(path.read_bytes(), self.before[path])

    def test_malformed_chunks_are_rejected_even_when_status_counts_them(self):
        original = self.index.read_bytes()
        cases = [
            ("UPDATE chunks SET chunk_index=1 WHERE paper_id=1", (), "contiguous"),
            ("UPDATE chunks SET token_count=0 WHERE paper_id=1", (), "token count"),
            ("UPDATE chunks SET vector=? WHERE paper_id=1", (pack_vector([1, 1, 1], 3),), "dimension"),
            ("UPDATE chunks SET vector=? WHERE paper_id=1", (struct.pack("<2e", float("inf"), 1),), "non-finite"),
            ("UPDATE chunks SET vector=? WHERE paper_id=1", (bytes(4),), "zero embedding"),
        ]
        for sql, values, message in cases:
            with self.subTest(message=message):
                self.index.write_bytes(original)
                with sqlite3.connect(self.index) as conn:
                    conn.execute(sql, values)
                with self.assertRaisesRegex(ValueError, message):
                    self.publish()
                self.assert_unchanged()

    def test_snapshot_input_collision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            publisher.publish_embeddings(self.db, self.archive, self.directory)
        self.assert_unchanged()

    def test_manifest_publish_failure_restores_old_archive(self):
        replace = publisher.os.replace
        def fail_manifest(source, destination):
            if Path(destination).resolve() == self.manifest_path.resolve():
                raise OSError("disk failure")
            return replace(source, destination)
        with patch.object(publisher.os, "replace", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.publish()
        self.assert_unchanged()

    def test_cli_reports_success_and_actionable_failure(self):
        args = ["--db", str(self.db), "--index", str(self.index), "--snapshot-dir", str(self.directory)]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(publisher.main(args), 0)
        self.assertEqual(json.loads(output.getvalue())["pending"], 0)
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(publisher.main(args + ["--index", str(self.root / "missing.sqlite3")]), 1)
        self.assertIn("required", error.getvalue())


if __name__ == "__main__":
    unittest.main()
