"""Exercise CLI safety and checkpoint selection without loading model weights."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import embed_papers
from reference_finder.embeddings import encoder_config


class EmbeddingCliTests(unittest.TestCase):
    def test_export_works_without_model_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.sqlite3"
            destination = Path(directory) / "snapshot.sqlite3"
            with sqlite3.connect(source) as conn:
                conn.execute("CREATE TABLE papers (id INTEGER, title TEXT, abstract TEXT)")
                conn.execute("INSERT INTO papers VALUES (1, 'A title', NULL)")
            output = io.StringIO()
            with patch.object(embed_papers, "make_encoder") as load, contextlib.redirect_stdout(output):
                result = embed_papers.main(["export", "--db", str(source), "--output", str(destination)])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(output.getvalue())["papers"], 1)
            load.assert_not_called()

    def test_missing_corpus_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(embed_papers, "make_encoder") as load, self.assertLogs(level="ERROR"):
                result = embed_papers.main(["build", "--db", str(Path(directory) / "missing.sqlite3")])
            self.assertEqual(result, 1)
            load.assert_not_called()

    def test_same_input_and_output_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            db = str(Path(directory) / "corpus.sqlite3")
            with patch.object(embed_papers, "make_encoder") as load, self.assertLogs(level="ERROR"):
                result = embed_papers.main(["build", "--db", db, "--index", db])
            self.assertEqual(result, 1)
            load.assert_not_called()

    def test_resume_reuses_immutable_revision_without_resolving_main(self):
        encoder = SimpleNamespace(
            model_id="test/model", revision="a" * 40, dimension=2,
            document_prompt="passage: ", query_prompt="query: ",
            device="cpu", dtype="float32",
        )
        config = encoder_config(encoder, 1024, 32)
        args = embed_papers.parser().parse_args(["build"])
        with patch.object(embed_papers.sys, "version_info", (3, 11)), \
                patch("reference_finder.embedding_backend.resolve_revision") as resolve, \
                patch("reference_finder.embedding_backend.SentenceEncoder", return_value=encoder) as load:
            _, actual = embed_papers.make_encoder(args, config)
        self.assertEqual(actual, config)
        resolve.assert_not_called()
        load.assert_called_once_with("test/model", "a" * 40, max_tokens=1024, device="auto", dtype="auto")

    def test_changed_model_is_rejected_without_network_or_model_load(self):
        encoder = SimpleNamespace(model_id="test/model", revision="a" * 40, dimension=2,
                                  document_prompt="passage: ", query_prompt="query: ")
        config = encoder_config(encoder, 1024, 32)
        args = embed_papers.parser().parse_args(["build", "--model", "different/model"])
        with patch.object(embed_papers.sys, "version_info", (3, 11)), \
                patch("reference_finder.embedding_backend.resolve_revision") as resolve, \
                patch("reference_finder.embedding_backend.SentenceEncoder") as load:
            with self.assertRaisesRegex(ValueError, "settings differ"):
                embed_papers.make_encoder(args, config)
        resolve.assert_not_called()
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
