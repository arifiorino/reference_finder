"""Server preparation preserves local work and resumes a verified linked index."""
from contextlib import closing, redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_gpu_embeddings as runner
from publish_embeddings import corpus_digest
from reference_finder.database_package import pack_database, split_archive
from reference_finder.embeddings import encoder_config, open_index, pack_vector, read_corpus, sync_paper_state


class Encoder:
    model_id = "Qwen/Qwen3-Embedding-4B"
    revision = "a" * 40
    dimension = 2
    document_prompt = ""
    query_prompt = "Instruct: retrieve related papers\nQuery:"


class ServerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.snapshots = self.root / "snapshots"
        self.snapshots.mkdir()
        self.data = self.root / "data"
        self.data.mkdir()
        self.work = self.data / "server"
        self.index = self.data / "embeddings.sqlite3"
        self.source = self.root / "fixture-papers.sqlite3"
        with closing(sqlite3.connect(self.source)) as conn:
            conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY,title TEXT,abstract TEXT)")
            conn.executemany("INSERT INTO papers VALUES (?,?,?)", [
                (1, "Original paper", "Original abstract"),
                (3, "Newly collected paper", "New abstract"),
            ])
            conn.commit()
        self.rows = read_corpus(self.source)
        self.config = encoder_config(Encoder(), 2048, 64)
        self.make_index(self.index, self.rows[:1])
        paper = pack_database(self.source, self.snapshots / "papers.sqlite3.xz", page_size=1024)
        index = pack_database(self.index, self.snapshots / "embeddings.sqlite3.xz", page_size=1024)
        self.paper_entry = self.entry("papers.sqlite3.xz", paper)
        self.paper_entry.update(paper_id_text_hash_sha256=corpus_digest(self.source),
                                coverage={"papers": 2, "abstracts": 2, "title_only": 0})
        self.index_entry = self.entry("embeddings.sqlite3.xz", index)
        self.index_entry.update(model_config=self.config,
                                coverage={"papers": 2, "embedded": 1, "pending": 1})
        self.manifest = {"layout": "linked-papers-v1", "files": [self.paper_entry, self.index_entry]}
        self.write_manifest()
        self.snapshot_bytes = {p: p.read_bytes() for p in self.snapshots.iterdir()}

    @staticmethod
    def entry(name, packed):
        return {"archive": name, "archive_sha256": packed["archive_sha256"],
                "sqlite_sha256": packed["snapshot_sha256"],
                "archive_bytes": packed["archive_bytes"], "sqlite_bytes": packed["snapshot_bytes"]}

    def write_manifest(self):
        (self.snapshots / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def make_index(self, path, rows):
        with closing(open_index(path, self.config)) as conn:
            sync_paper_state(conn, rows)
            with conn:
                for row in rows:
                    conn.execute("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?)",
                                 (row["id"], 0, 12, pack_vector([1, row["id"]], 2)))
                    conn.execute("UPDATE paper_state SET embedded=1 WHERE paper_id=?", (row["id"],))

    def prepare(self):
        return runner.prepare(self.snapshots, self.work, self.index)

    def run_main(self, *args):
        with patch.object(runner, "ROOT", self.root), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.main(["--index", str(self.index), *args])

    def assert_snapshots_unchanged(self):
        self.assertEqual({p: p.read_bytes() for p in self.snapshots.iterdir()}, self.snapshot_bytes)

    def split_embedding_fixture(self):
        archive = self.snapshots / "embeddings.sqlite3.xz"
        source = self.root / "unsplit-embeddings.sqlite3.xz"
        archive.rename(source)
        descriptor = split_archive(source, archive, part_bytes=128)
        self.snapshot_bytes = {p: p.read_bytes() for p in self.snapshots.iterdir()}
        return [self.snapshots / part["name"] for part in descriptor["parts"]]

    def test_prepare_uses_current_snapshot_without_replacing_local_paper_database(self):
        local = self.data / "papers.sqlite3"
        with closing(sqlite3.connect(local)) as conn:
            conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY,title TEXT,abstract TEXT)")
            conn.execute("INSERT INTO papers VALUES (99,'Locally collected paper','Unsynced work')")
            conn.commit()
        old_local, old_index = local.read_bytes(), self.index.read_bytes()
        paper, status = self.prepare()
        self.assertEqual(paper, self.work / ("papers-" + self.paper_entry["sqlite_sha256"][:16] + ".sqlite3"))
        self.assertEqual(hashlib.sha256(paper.read_bytes()).hexdigest(), self.paper_entry["sqlite_sha256"])
        self.assertEqual((status["papers"], status["embedded"], status["pending"]), (2, 1, 1))
        self.assertEqual(local.read_bytes(), old_local)
        self.assertEqual(self.index.read_bytes(), old_index)
        self.assert_snapshots_unchanged()
        self.assertEqual(self.prepare(), (paper, status))
        self.assertEqual(self.index.read_bytes(), old_index)

    def test_missing_index_is_restored_from_verified_bundled_vectors(self):
        self.index.unlink()
        paper, status = self.prepare()
        self.assertTrue(paper.is_file())
        self.assertEqual(hashlib.sha256(self.index.read_bytes()).hexdigest(), self.index_entry["sqlite_sha256"])
        self.assertEqual((status["embedded"], status["pending"]), (1, 1))
        self.assert_snapshots_unchanged()

    def test_missing_index_restores_multipart_vectors_without_model_inference(self):
        parts = self.split_embedding_fixture()
        self.assertGreater(len(parts), 1)
        self.assertFalse((self.snapshots / "embeddings.sqlite3.xz").exists())
        self.index.unlink()
        with patch.object(runner, "require_cuda") as cuda, patch.object(runner.subprocess, "run") as build:
            self.assertEqual(self.run_main("--prepare-only"), 0)
        cuda.assert_not_called()
        build.assert_not_called()
        self.assertEqual(hashlib.sha256(self.index.read_bytes()).hexdigest(), self.index_entry["sqlite_sha256"])
        paper, status = self.prepare()
        self.assertEqual((status["papers"], status["embedded"], status["pending"]), (2, 1, 1))
        self.assertEqual(status["config"], self.config)
        self.assertTrue(paper.is_file())
        self.assert_snapshots_unchanged()

    def test_corrupt_or_missing_archive_part_never_publishes_new_index(self):
        part = self.split_embedding_fixture()[0]
        original = part.read_bytes()
        self.index.unlink()
        for problem in ("modified", "truncated", "missing"):
            with self.subTest(problem=problem):
                if problem == "modified":
                    part.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                elif problem == "truncated":
                    part.write_bytes(original[:-1])
                else:
                    part.unlink()
                with self.assertRaises((ValueError, OSError)):
                    self.prepare()
                self.assertFalse(self.index.exists())
                part.write_bytes(original)
        self.assert_snapshots_unchanged()

    def test_multipart_archive_must_match_repository_manifest_hash(self):
        self.split_embedding_fixture()
        self.index.unlink()
        self.index_entry["archive_sha256"] = "f" * 64
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.index.exists())

    def test_index_cannot_replace_archive_part_or_descriptor(self):
        parts = self.split_embedding_fixture()
        previous = self.index.read_bytes()
        descriptor = self.snapshots / "embeddings.sqlite3.xz.parts.json"
        for path in (parts[0], parts[-1], descriptor):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "separate"):
                runner.prepare(self.snapshots, self.work, path)
        self.assertEqual(self.index.read_bytes(), previous)
        self.assert_snapshots_unchanged()

    def test_bad_paper_archive_hash_does_not_touch_existing_index(self):
        previous = self.index.read_bytes()
        (self.snapshots / "papers.sqlite3.xz").write_bytes(b"damaged archive")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.index.read_bytes(), previous)

    def test_bad_restored_paper_hash_is_rejected_before_use(self):
        previous = self.index.read_bytes()
        self.paper_entry["sqlite_sha256"] = "0" * 64
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.index.read_bytes(), previous)

    def test_modified_cached_paper_snapshot_is_rejected(self):
        paper, _ = self.prepare()
        with sqlite3.connect(paper) as conn:
            conn.execute("UPDATE papers SET title='Changed cached title' WHERE id=3")
        previous = self.index.read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.index.read_bytes(), previous)

    def test_cached_paper_sidecars_are_rejected_even_when_main_file_hash_matches(self):
        paper, _ = self.prepare()
        previous = self.index.read_bytes()
        for suffix in ("-wal", "-journal"):
            with self.subTest(sidecar=suffix):
                sidecar = Path(str(paper) + suffix)
                sidecar.write_bytes(b"Unresolved SQLite writes must not be ignored")
                self.assertEqual(hashlib.sha256(paper.read_bytes()).hexdigest(), self.paper_entry["sqlite_sha256"])
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(self.index.read_bytes(), previous)
                sidecar.unlink()

    def test_bad_embedding_archive_cannot_seed_a_new_index(self):
        self.index.unlink()
        (self.snapshots / "embeddings.sqlite3.xz").write_bytes(b"damaged archive")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.index.exists())

    def test_bad_restored_embedding_hash_does_not_publish_new_index(self):
        self.index.unlink()
        self.index_entry["sqlite_sha256"] = "f" * 64
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.index.exists())

    def test_verified_archive_with_wrong_config_does_not_publish_new_index(self):
        bad_seed = self.root / "different-config.sqlite3"
        self.make_index(bad_seed, self.rows[:1])
        with sqlite3.connect(bad_seed) as conn:
            conn.execute("UPDATE embedding_meta SET value=? WHERE key='config'",
                         (json.dumps(dict(self.config, max_tokens=1024)),))
        archive = self.snapshots / "embeddings.sqlite3.xz"
        archive.unlink()
        packed = pack_database(bad_seed, archive, page_size=1024)
        self.index_entry.update(self.entry("embeddings.sqlite3.xz", packed))
        self.write_manifest()
        self.index.unlink()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.index.exists())

    def test_index_must_not_overwrite_paper_or_snapshot_paths(self):
        previous = self.index.read_bytes()
        paper = self.work / ("papers-" + self.paper_entry["sqlite_sha256"][:16] + ".sqlite3")
        for path in (paper, self.snapshots / "papers.sqlite3.xz",
                     self.snapshots / "embeddings.sqlite3.xz", self.snapshots / "manifest.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.prepare(self.snapshots, self.work, path)
        self.assertEqual(self.index.read_bytes(), previous)
        self.assertFalse(paper.exists())
        self.assert_snapshots_unchanged()

    def test_incompatible_model_and_legacy_index_are_preserved(self):
        original = self.index.read_bytes()
        for config in (dict(self.config, model_id="nvidia/Nemotron-3-Embed-1B-BF16"),
                       dict(self.config, format_version=2), dict(self.config, revision="b" * 40)):
            with self.subTest(config=config):
                self.index.write_bytes(original)
                with sqlite3.connect(self.index) as conn:
                    conn.execute("UPDATE embedding_meta SET value=? WHERE key='config'", (json.dumps(config),))
                previous = self.index.read_bytes()
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertEqual(self.index.read_bytes(), previous)

    def test_extra_local_vector_states_are_never_removed(self):
        extra = dict(self.rows[0], id=99, title="Local research", abstract="Not in snapshot")
        self.make_index(self.index, self.rows[:1] + [extra])
        previous = self.index.read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.index.read_bytes(), previous)

    def test_prepare_only_needs_no_cuda_and_never_runs_inference(self):
        previous = self.index.read_bytes()
        with patch.object(runner, "require_cuda") as cuda, patch.object(runner.subprocess, "run") as build:
            self.assertEqual(self.run_main("--prepare-only"), 0)
        cuda.assert_not_called()
        build.assert_not_called()
        self.assertEqual(self.index.read_bytes(), previous)

    def test_complete_index_skips_cuda_and_model_loading(self):
        self.make_index(self.index, self.rows)
        previous = self.index.read_bytes()
        with patch.object(runner, "require_cuda") as cuda, patch.object(runner.subprocess, "run") as build:
            self.assertEqual(self.run_main(), 0)
        cuda.assert_not_called()
        build.assert_not_called()
        self.assertEqual(self.index.read_bytes(), previous)

    def test_build_uses_current_python_cuda_and_requested_batch_settings(self):
        def complete(command, check):
            self.assertFalse(check)
            self.make_index(self.index, self.rows)
            return subprocess.CompletedProcess(command, 0)
        with patch.object(runner, "require_cuda") as cuda, \
                patch.object(runner.subprocess, "run", side_effect=complete) as build:
            self.assertEqual(self.run_main("--batch-size", "8", "--dtype", "float16"), 0)
        cuda.assert_called_once_with("float16")
        command = build.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, str(self.root / "embed_papers.py"), "build"])
        for option, expected in (("--device", "cuda"), ("--dtype", "float16"),
                                 ("--batch-size", "8"), ("--index", str(self.index))):
            self.assertEqual(command[command.index(option) + 1], expected)
        self.assertEqual(command[command.index("--db") + 1],
                         str(self.work / ("papers-" + self.paper_entry["sqlite_sha256"][:16] + ".sqlite3")))
        self.assertNotIn("--revision", command)
        self.assertNotIn("--limit", command)
        self.assert_snapshots_unchanged()

    def test_interrupted_build_returns_its_status_without_losing_prior_vectors(self):
        previous = self.index.read_bytes()
        with patch.object(runner, "require_cuda"), \
                patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 130)):
            self.assertEqual(self.run_main(), 130)
        self.assertEqual(self.index.read_bytes(), previous)

    def test_successful_subprocess_must_actually_complete_unlimited_build(self):
        with patch.object(runner, "require_cuda"), \
                patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(self.run_main(), 1)

    def test_explicit_limit_permits_pending_work_and_is_forwarded(self):
        with patch.object(runner, "require_cuda"), \
                patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as build:
            self.assertEqual(self.run_main("--limit", "1"), 0)
        command = build.call_args.args[0]
        self.assertEqual(command[command.index("--limit") + 1], "1")


if __name__ == "__main__":
    unittest.main()
