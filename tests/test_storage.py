import json
import unittest

from reference_finder.storage import Database, normalize_doi, reconstruct_abstract


def openalex_record(identifier="https://openalex.org/W1", doi=None, title="A secure protocol", author="Alice Example", abstract=None):
    return {
        "id": identifier, "doi": doi, "display_name": title,
        "publication_year": 2024, "publication_date": "2024-08-01",
        "authorships": [{"author": {"display_name": author}}] if author else [],
        "abstract_inverted_index": abstract,
    }


def scholar_record(identifier="s2-1", doi=None, title="A secure protocol", author="Alice Example", abstract=None):
    return {
        "paperId": identifier, "externalIds": {"DOI": doi} if doi else {},
        "title": title, "year": 2024, "publicationDate": "2024-08-01",
        "authors": [{"name": author}] if author else [], "abstract": abstract,
    }


class NormalizationTests(unittest.TestCase):
    def test_abstract_reconstructs_positions_and_repeated_tokens(self):
        self.assertEqual(
            reconstruct_abstract({"protocol.": [6], "We": [0, 3], "a": [5], "propose": [4], "study": [1], "security.": [2]}),
            "We study security. We propose a protocol.",
        )

    def test_missing_abstract_and_invalid_offsets(self):
        self.assertIsNone(reconstruct_abstract(None))
        self.assertIsNone(reconstruct_abstract({}))
        self.assertEqual(reconstruct_abstract({"valid": [2], "invalid": [-1, "0", 1.5], "ignored": None}), "valid")

    def test_doi_normalization(self):
        for value in ("10.1145/ABC.123", " https://doi.org/10.1145/ABC.123 ", "HTTP://DX.DOI.ORG/10.1145/ABC.123", "doi: 10.1145/ABC.123", "https://doi.org/10.1145%2FABC.123"):
            with self.subTest(value=value):
                self.assertEqual(normalize_doi(value), "10.1145/abc.123")
        for value in (None, "", "not-a-doi", "https://example.org/10.1145/abc", "10.1145/has spaces"):
            with self.subTest(value=value):
                self.assertIsNone(normalize_doi(value))


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)

    def paper(self, paper_id):
        return self.db.conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()

    def test_doi_deduplication_retains_both_provider_records(self):
        oa = openalex_record(doi="https://doi.org/10.1145/ABC")
        s2 = scholar_record(doi="10.1145/abc", title="A secure protocol: revised metadata")
        first = self.db.upsert("openalex", oa, "ccs", 2024)
        second = self.db.upsert("semantic_scholar", s2, "ccs", 2024)
        self.assertEqual(first, second)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        records = self.db.conn.execute("SELECT provider, raw_json FROM source_records ORDER BY provider").fetchall()
        self.assertEqual([(row["provider"], json.loads(row["raw_json"])) for row in records], [("openalex", oa), ("semantic_scholar", s2)])
        self.assertEqual(self.db.coverage("ccs", 2024)["papers"], 1)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM paper_venues").fetchone()[0], 2)

    def test_missing_doi_merges_matching_title_and_first_author(self):
        first = self.db.upsert("openalex", openalex_record(title="A Secure Protocol!", author="ALICE EXAMPLE"), "usenix-security", 2024)
        second = self.db.upsert("semantic_scholar", scholar_record(title="A secure protocol", author="Alice Example"), "usenix-security", 2024)
        self.assertEqual(first, second)

    def test_title_fallback_rejects_different_or_missing_authors(self):
        first = self.db.upsert("openalex", openalex_record(), "usenix-security", 2024)
        different = self.db.upsert("semantic_scholar", scholar_record(identifier="different", author="Bob Example"), "usenix-security", 2024)
        missing = self.db.upsert("semantic_scholar", scholar_record(identifier="missing", author=None), "usenix-security", 2024)
        self.assertEqual(len({first, different, missing}), 3)

    def test_title_fallback_requires_same_venue_and_year(self):
        first = self.db.upsert("openalex", openalex_record(), "ccs", 2024)
        different_venue = self.db.upsert("semantic_scholar", scholar_record(identifier="venue"), "sp", 2024)
        different_year = self.db.upsert("semantic_scholar", scholar_record(identifier="year"), "ccs", 2023)
        self.assertEqual(len({first, different_venue, different_year}), 3)

    def test_title_fallback_rejects_blank_first_author_names(self):
        oa = openalex_record()
        oa["authorships"] = [{"author": {}}]
        s2 = scholar_record()
        s2["authors"] = [{"name": "   "}]
        first = self.db.upsert("openalex", oa, "usenix-security", 2024)
        second = self.db.upsert("semantic_scholar", s2, "usenix-security", 2024)
        self.assertNotEqual(first, second)

    def test_title_fallback_rejects_ambiguous_candidates(self):
        first = self.db.upsert("openalex", openalex_record(identifier="W1", doi="10.1145/first"), "ccs", 2024)
        second = self.db.upsert("openalex", openalex_record(identifier="W2", doi="10.1145/second"), "ccs", 2024)
        ambiguous = self.db.upsert("semantic_scholar", scholar_record(), "ccs", 2024)
        self.assertEqual(len({first, second, ambiguous}), 3)

    def test_same_title_different_dois_do_not_merge(self):
        first = self.db.upsert("openalex", openalex_record(doi="10.1145/first"), "ccs", 2024)
        second = self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/second"), "ccs", 2024)
        self.assertNotEqual(first, second)

    def test_null_abstract_never_erases_existing_text(self):
        first = self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract={"Original": [0], "abstract.": [1]}), "ccs", 2024)
        self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/abc", abstract=None), "ccs", 2024)
        self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract=None), "ccs", 2024)
        self.assertEqual(self.paper(first)["abstract"], "Original abstract.")
        self.assertEqual(self.paper(first)["abstract_source"], "openalex")

    def test_semantic_scholar_abstract_is_preferred(self):
        first = self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract={"OpenAlex": [0]}), "crypto", 2024)
        self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/abc", abstract="  Semantic Scholar text.  "), "crypto", 2024)
        self.db.upsert("openalex", openalex_record(doi="10.1145/abc", abstract={"Later": [0], "OpenAlex": [1]}), "crypto", 2024)
        self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/abc", abstract="  "), "crypto", 2024)
        self.assertEqual(self.paper(first)["abstract"], "Semantic Scholar text.")
        self.assertEqual(self.paper(first)["abstract_source"], "semantic_scholar")

    def test_later_doi_bridges_records_and_preserves_memberships_and_provenance(self):
        first = self.db.upsert("openalex", openalex_record(title="Original title", abstract={"OpenAlex": [0]}), "crypto", 2024)
        second = self.db.upsert("semantic_scholar", scholar_record(doi="10.1145/bridge", title="Different indexed title", abstract="Preferred abstract"), "eurocrypt", 2023)
        self.assertNotEqual(first, second)
        bridge = openalex_record(title="Original title", doi="10.1145/bridge")
        merged = self.db.upsert("openalex", bridge, "crypto", 2024)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(self.paper(merged)["doi"], "10.1145/bridge")
        self.assertEqual(self.paper(merged)["abstract"], "Preferred abstract")
        self.assertEqual(self.paper(merged)["abstract_source"], "semantic_scholar")
        memberships = self.db.conn.execute("SELECT venue, year, provider FROM paper_venues WHERE paper_id=? ORDER BY venue", (merged,)).fetchall()
        self.assertEqual([tuple(row) for row in memberships], [("crypto", 2024, "openalex"), ("eurocrypt", 2023, "semantic_scholar")])
        records = self.db.conn.execute("SELECT provider, paper_id, raw_json FROM source_records ORDER BY provider").fetchall()
        self.assertEqual([(row["provider"], row["paper_id"]) for row in records], [("openalex", merged), ("semantic_scholar", merged)])
        self.assertEqual(json.loads(records[0]["raw_json"]), bridge)
        self.assertEqual(json.loads(records[1]["raw_json"])["abstract"], "Preferred abstract")
        self.assertEqual(self.db.conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_page_data_and_checkpoint_roll_back_together(self):
        job = '{"provider":"openalex","venue":"ccs","year":2024}'
        with self.db.conn:
            first = self.db.upsert("openalex", openalex_record(), "ccs", 2024)
            self.db.save_checkpoint(job, "first-cursor", False, 10, 1)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with self.db.conn:
                self.db.upsert("semantic_scholar", scholar_record(identifier="new", title="A different paper"), "ccs", 2024)
                self.db.save_checkpoint(job, "second-cursor", True, 20, 1)
                raise RuntimeError("interrupted")
        self.assertEqual(self.db.conn.execute("SELECT id FROM papers").fetchall()[0][0], first)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM source_records").fetchone()[0], 1)
        checkpoint = self.db.checkpoint(job)
        self.assertEqual((checkpoint["cursor"], checkpoint["done"], checkpoint["fetched"], checkpoint["accepted"]), ("first-cursor", 0, 10, 1))
        with self.db.conn:
            self.db.save_checkpoint(job, None, True, 20, 2)
        checkpoint = self.db.checkpoint(job)
        self.assertEqual((checkpoint["cursor"], checkpoint["done"], checkpoint["fetched"], checkpoint["accepted"]), (None, 1, 30, 3))


if __name__ == "__main__":
    unittest.main()
