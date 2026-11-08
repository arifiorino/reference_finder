"""Exercise conference routing, persistence, and resumable source failures."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import build_corpus
from reference_finder.api import APIError


def roster(venue, year, abstract=None):
    return {"status": "complete", "note": "Fixture source", "urls": [], "records": [{
        "external_id": venue + "/" + str(year), "title": venue + " research paper",
        "year": year, "doi": "10.1234/" + venue + str(year), "abstract": abstract,
        "authors": ["A. Researcher"], "url": "https://example.org/" + venue,
    }]}


class BuildCorpusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.db = root / "papers.sqlite3"
        self.output = root / "report"
        self.args = ["--db", str(self.db), "--output", str(self.output),
                     "--cache-dir", str(root / "cache"), "--start-year", "2025", "--end-year", "2025"]
        self.fetch = self.patch("build_corpus.WebFetcher")
        self.dblp = self.patch("reference_finder.dblp.collect_dblp", side_effect=lambda v, y, f: roster(v, y))
        self.iacr = self.patch("reference_finder.iacr.collect_iacr", side_effect=lambda v, y, f: roster(v, y, "Official abstract."))
        self.pets = self.patch("reference_finder.pets.collect_pets", side_effect=lambda y, f: roster("pets", y, "Privacy abstract."))
        self.ndss = self.patch("reference_finder.ndss.collect_ndss", side_effect=lambda y, f: roster("ndss", y, "Network abstract."))
        self.usenix = self.patch("reference_finder.usenix.collect_usenix", side_effect=lambda y, f: roster("usenix-security", y))
        self.ccs = self.patch("reference_finder.ccs.collect_ccs")
        self.tcc = self.patch("reference_finder.tcc.collect_tcc", side_effect=lambda y, f: roster("tcc", y, "Accepted paper abstract."))

    def patch(self, target, **kwargs):
        patcher = patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def run_cli(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_corpus.main(self.args + list(extra))

    def test_default_sources_collect_new_venues_and_resume_without_duplicate_papers(self):
        venues = ["--venues", "pets", "ndss", "tcc", "pkc"]
        self.assertEqual(self.run_cli(*venues), 0)
        self.assertEqual([c.args[:2] for c in self.dblp.call_args_list],
                         [(v, 2025) for v in ("pets", "ndss", "tcc", "pkc")])
        self.assertEqual([c.args[:2] for c in self.iacr.call_args_list], [("tcc", 2025), ("pkc", 2025)])
        self.pets.assert_called_once()
        self.ndss.assert_called_once()
        self.usenix.assert_not_called()
        self.ccs.assert_not_called()
        self.tcc.assert_not_called()
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM roster_runs").fetchone()[0], 8)
        self.assertEqual(self.run_cli(*venues), 0)
        self.assertEqual(self.dblp.call_count, 4)
        self.assertEqual(self.iacr.call_count, 2)
        self.assertEqual(self.pets.call_count, 1)
        self.assertEqual(self.ndss.call_count, 1)

    def test_report_defaults_include_twelve_venues_without_network(self):
        self.assertEqual(self.run_cli("--report-only"), 0)
        report = json.loads((self.output / "coverage.json").read_text())
        self.assertEqual(len(report["coverage"]), 12)
        self.assertTrue({"pets", "ndss", "tcc", "pkc"}.issubset(r["venue"] for r in report["coverage"]))
        for collector in (self.dblp, self.iacr, self.pets, self.ndss, self.tcc):
            collector.assert_not_called()

    def test_tcc_accepted_papers_source_only_runs_for_tcc_2026_and_resumes(self):
        args = ("--venues", "tcc", "pkc", "--sources", "dblp", "tcc", "--end-year", "2026")
        self.assertEqual(self.run_cli(*args), 0)
        self.tcc.assert_called_once()
        self.assertEqual(self.tcc.call_args.args[0], 2026)
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT abstract FROM papers WHERE doi='10.1234/tcc2026'").fetchone()[0], "Accepted paper abstract.")
        self.assertEqual(self.run_cli(*args), 0)
        self.tcc.assert_called_once()

    def test_failed_official_source_retries_while_completed_roster_is_kept(self):
        self.pets.side_effect = APIError("Source temporarily unavailable")
        self.assertEqual(self.run_cli("--venues", "pets"), 1)
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.pets.side_effect = lambda y, f: roster("pets", y, "Recovered abstract.")
        self.assertEqual(self.run_cli("--venues", "pets"), 0)
        self.dblp.assert_called_once()
        self.assertEqual(self.pets.call_count, 2)
        with sqlite3.connect(self.db) as db:
            self.assertEqual(db.execute("SELECT abstract FROM papers").fetchone()[0], "Recovered abstract.")


if __name__ == "__main__":
    unittest.main()
