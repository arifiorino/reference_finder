"""Safe, model-free conversion of existing float32 embedding indexes."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys
import tempfile
import unittest

from reference_finder.embedding_conversion import convert_index
from reference_finder.embeddings import (
    build_index, encoder_config, load_config, search_index, unpack_vector,
)


class TestEncoder:
    model_id = "test/converter"
    revision = "a" * 40
    dimension = 3
    document_prompt = ""
    query_prompt = "query: "

    def document_chunks(self, text, overlap=64):
        return [{"text": text, "token_count": len(text.split())}]

    def encode_documents(self, texts, batch_size=32):
        return [[1, len(text), 2] for text in texts]

    def encode_queries(self, texts, batch_size=1):
        return [[1, 0, 0] for text in texts]


class EmbeddingConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.corpus = self.root / "corpus.sqlite3"
        self.index = self.root / "float32.sqlite3"
        self.output = self.root / "float16.sqlite3"
        with closing(sqlite3.connect(self.corpus)) as conn:
            conn.execute("CREATE TABLE papers (id INTEGER PRIMARY KEY, title TEXT, abstract TEXT)")
            conn.executemany("INSERT INTO papers VALUES (?,?,?)", [
                (1, "First paper", "Abstract one"), (2, "Second paper", None),
                (3, "A very long title for the third paper", "Abstract three"),
            ])
            conn.commit()
        self.encoder = TestEncoder()
        self.config = encoder_config(self.encoder, 1024, 32, vector_format="float32-le")
        build_index(self.corpus, self.index, self.encoder, self.config)
        with closing(sqlite3.connect(self.index)) as conn:
            conn.execute("INSERT INTO embedding_meta VALUES ('additional', 'keep this')")
            conn.commit()

    def rows(self, path, sql):
        with closing(sqlite3.connect(path)) as conn:
            return conn.execute(sql).fetchall()

    def test_conversion_preserves_source_metadata_and_rankings(self):
        original_digest = hashlib.sha256(self.index.read_bytes()).digest()
        result = convert_index(self.index, self.output)
        self.assertEqual(hashlib.sha256(self.index.read_bytes()).digest(), original_digest)
        self.assertEqual(result["converted_chunks"], 3)
        self.assertEqual(result["pending"], 0)
        self.assertEqual(result["original_vector_bytes"], result["vector_bytes"] * 2)
        self.assertEqual(load_config(self.index), self.config)
        expected = dict(self.config, format_version=2, vector_format="float16-le")
        self.assertEqual(load_config(self.output), expected)
        for sql in ("SELECT * FROM papers ORDER BY id",
                    "SELECT paper_id,chunk_index,text,token_count FROM chunks ORDER BY paper_id,chunk_index",
                    "SELECT * FROM embedding_meta WHERE key != 'config' ORDER BY key"):
            self.assertEqual(self.rows(self.index, sql), self.rows(self.output, sql))
        for (blob,) in self.rows(self.output, "SELECT vector FROM chunks"):
            self.assertEqual(len(blob), 6)
            self.assertEqual(len(unpack_vector(blob, "float16-le")), 3)
        before = search_index(self.index, self.encoder, "related paper", top_k=3)
        after = search_index(self.output, self.encoder, "related paper", top_k=3)
        self.assertEqual([r["paper_id"] for r in before], [r["paper_id"] for r in after])
        for left, right in zip(before, after):
            self.assertAlmostEqual(left["score"], right["score"], delta=0.001)

    def test_preserves_partial_index_and_committed_wal_contents(self):
        with closing(sqlite3.connect(self.index)) as writer:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("DELETE FROM chunks WHERE paper_id=3")
            writer.execute("UPDATE papers SET embedded=0 WHERE id=3")
            writer.execute("UPDATE papers SET title='New title committed in WAL' WHERE id=1")
            writer.commit()
            self.assertGreater(Path(str(self.index) + "-wal").stat().st_size, 0)
            result = convert_index(self.index, self.output)
            self.assertEqual(result["pending"], 1)
            self.assertEqual(result["converted_chunks"], 2)
            self.assertEqual(self.rows(self.output, "SELECT title FROM papers WHERE id=1"),
                             [("New title committed in WAL",)])
        self.assertFalse(Path(str(self.output) + "-wal").exists())

    def test_rejects_existing_output_and_same_path(self):
        self.output.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            convert_index(self.index, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep")
        with self.assertRaisesRegex(ValueError, "different files"):
            convert_index(self.index, self.index)

    def test_corrupt_vector_never_publishes_or_alters_source(self):
        for blob in (b"bad", struct.pack("<fff", 0, 0, 0),
                     struct.pack("<fff", float("nan"), 1, 2), struct.pack("<ff", 1, 2)):
            with self.subTest(blob=blob):
                with closing(sqlite3.connect(self.index)) as conn:
                    conn.execute("UPDATE chunks SET vector=? WHERE paper_id=3", (blob,))
                    conn.commit()
                original_digest = hashlib.sha256(self.index.read_bytes()).digest()
                with self.assertRaisesRegex(ValueError, "Invalid vector for paper 3 chunk 0"):
                    convert_index(self.index, self.output)
                self.assertFalse(self.output.exists())
                self.assertEqual(hashlib.sha256(self.index.read_bytes()).digest(), original_digest)
                self.assertFalse(list(self.root.glob(".sqlite-package-*")))

    def test_already_float16_is_rejected_without_rounding_again(self):
        convert_index(self.index, self.output)
        second = self.root / "second.sqlite3"
        with self.assertRaisesRegex(ValueError, "already uses float16"):
            convert_index(self.output, second)
        self.assertFalse(second.exists())

    def test_cli_works_without_site_packages_or_model_dependencies(self):
        command = [sys.executable, "-S", str(Path(__file__).resolve().parents[1] / "embed_papers.py"),
                   "convert", "--index", str(self.index), "--output", str(self.output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["config"]["vector_format"], "float16-le")


if __name__ == "__main__":
    unittest.main()
