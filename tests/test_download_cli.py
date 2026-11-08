"""Publisher fallback must remain optional and report retryable failures."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import download_abstracts
from reference_finder.api import RequestLimit


class DownloadCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.args = ["--db", str(root / "papers.sqlite3"), "--cache-dir", str(root / "cache"),
                     "--venues", "tcc", "pkc", "--start-year", "2017", "--end-year", "2026",
                     "--providers", "springer", "--phase", "enrich"]

    def invoke(self, result=None, error=None):
        with patch("reference_finder.springer.enrich_springer", return_value=result, side_effect=error) as enrich, \
             patch("reference_finder.webfetch.WebFetcher") as fetch, \
             patch("download_abstracts.OpenAlex") as oa, \
             patch("download_abstracts.SemanticScholar") as s2, \
             contextlib.redirect_stdout(io.StringIO()):
            status = download_abstracts.main(self.args)
            oa.assert_not_called()
            s2.assert_not_called()
            self.assertEqual(enrich.call_args.args[1:4], (["tcc", "pkc"], 2017, 2026))
            self.assertIs(enrich.call_args.args[4], fetch.return_value)
            return status

    def test_publisher_only_enrichment_does_not_initialize_api_clients(self):
        self.assertEqual(self.invoke({"status": "complete", "filled": 2}), 0)

    def test_partial_publisher_retrieval_returns_nonzero(self):
        self.assertEqual(self.invoke({"status": "incomplete", "filled": 1}), 1)

    def test_request_budget_can_pause_publisher_enrichment(self):
        self.assertEqual(self.invoke(error=RequestLimit("Budget exhausted")), 2)

    def test_publisher_source_cannot_claim_paper_discovery(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            download_abstracts.main(self.args + ["--phase", "discover"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
