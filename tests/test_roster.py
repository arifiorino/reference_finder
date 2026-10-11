"""Exercise roster/API identity and snapshot safety using real SQLite state."""
import json
import unittest

from reference_finder.roster import RosterStore
from reference_finder.storage import Database


def roster_record(identifier="proceedings/paper1", **changes):
    record = {
        "external_id": identifier, "title": "A practical security result", "year": 2025,
        "doi": None, "authors": ["Ada Researcher"], "abstract": None,
        "url": "https://proceedings.example/paper/" + identifier,
    }
    record.update(changes)
    return record


def api_record(identifier="api1", **changes):
    record = {
        "paperId": identifier, "title": "A practical security result", "year": 2025,
        "externalIds": {}, "authors": [{"name": "Ada Researcher"}], "abstract": None,
    }
    record.update(changes)
    return record


def snapshot(*records, status="complete"):
    return {"records": list(records), "status": status, "urls": ["https://proceedings.example/2025"], "note": "Test roster"}


class RosterStoreTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)

    def store(self):
        return RosterStore(self.db)

    def rows(self):
        return self.db.conn.execute("SELECT * FROM papers ORDER BY id").fetchall()

    def test_doi_links_api_to_event_despite_title_and_publication_year_difference(self):
        paper_id = self.db.upsert("semantic_scholar", api_record(
            year=2026, title="Earlier API title", externalIds={"DOI": "10.1007/BOOK_1"},
            abstract="API abstract"), "asiacrypt", 2026)
        store = self.store()
        store.save("dblp", "asiacrypt", 2025, snapshot(roster_record(doi="https://doi.org/10.1007/book_1")))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(store.coverage("asiacrypt", 2025)["abstracts"], 1)
        paper = list(store.papers(["asiacrypt"], 2025, 2025))[0]
        self.assertEqual(paper["id"], paper_id)
        self.assertEqual(paper["event_year"], 2025)
        self.assertEqual(paper["publication_year"], 2026)
        self.assertEqual(paper["title"], "A practical security result")

    def test_dblp_external_id_links_api_without_doi_across_publication_years(self):
        paper_id = self.db.upsert("semantic_scholar", api_record(
            title="API wording", year=2026, externalIds={"DBLP": "conf/asiacrypt/Example25"},
            abstract="The indexed abstract."), "asiacrypt", 2026)
        store = self.store()
        store.save("dblp", "asiacrypt", 2025, snapshot(roster_record("conf/asiacrypt/Example25")))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(list(store.papers(["asiacrypt"], 2025, 2025))[0]["id"], paper_id)
        self.assertEqual(store.coverage("asiacrypt", 2025)["abstracts"], 1)

    def test_exact_title_links_non_doi_api_and_official_abstract_survives_later_api_updates(self):
        paper_id = self.db.upsert("semantic_scholar", api_record(authors=[]), "usenix-security", 2025)
        store = self.store()
        store.save("usenix", "usenix-security", 2025, snapshot(roster_record(abstract="Official abstract.")))
        for abstract in (None, "Lower-priority API version."):
            with self.subTest(abstract=abstract):
                self.db.upsert("semantic_scholar", api_record(abstract=abstract), "usenix-security", 2025)
                paper = self.rows()[0]
                self.assertEqual(paper["id"], paper_id)
                self.assertEqual(paper["abstract"], "Official abstract.")
                self.assertEqual(paper["abstract_source"], "usenix")
                self.assertEqual(len(self.rows()), 1)

    def test_same_title_in_adjacent_editions_is_not_identity_without_identifier(self):
        self.db.upsert("semantic_scholar", api_record(year=2024), "crypto", 2024)
        store = self.store()
        store.save("iacr", "crypto", 2025, snapshot(roster_record()))
        self.assertEqual(len(self.rows()), 2)
        api_id = self.db.conn.execute("SELECT paper_id FROM source_records WHERE provider='semantic_scholar'").fetchone()[0]
        roster_id = self.db.conn.execute("SELECT paper_id FROM roster_entries").fetchone()[0]
        self.assertNotEqual(api_id, roster_id)

    def test_different_titles_in_same_venue_and_year_remain_distinct(self):
        self.db.upsert("semantic_scholar", api_record(title="A different result"), "crypto", 2025)
        store = self.store()
        store.save("iacr", "crypto", 2025, snapshot(roster_record()))
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(store.coverage("crypto", 2025)["unverified_api"], 1)

    def test_matching_titles_with_conflicting_dois_are_not_merged(self):
        self.db.upsert("semantic_scholar", api_record(externalIds={"DOI": "10.1007/first"}), "crypto", 2025)
        store = self.store()
        store.save("dblp", "crypto", 2025, snapshot(roster_record(doi="10.1007/second")))
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual({r["doi"] for r in self.rows()}, {"10.1007/first", "10.1007/second"})

    def test_conflicting_doi_disqualifies_bad_api_dblp_cross_reference(self):
        self.db.upsert("semantic_scholar", api_record(externalIds={
            "DOI": "10.1007/first", "DBLP": "conf/crypto/Example25"}), "crypto", 2025)
        store = self.store()
        store.save("dblp", "crypto", 2025, snapshot(roster_record("conf/crypto/Example25", doi="10.1007/second")))
        self.assertEqual(len(self.rows()), 2)
        roster = list(store.papers(["crypto"], 2025, 2025))[0]
        self.assertEqual(roster["doi"], "10.1007/second")

    def test_later_identifier_merge_preserves_roster_references_and_official_abstract(self):
        api_id = self.db.upsert("semantic_scholar", api_record(
            title="API preprint title", externalIds={"DOI": "10.1007/linked"}, abstract="API version"), "crypto", 2025)
        store = self.store()
        store.save("iacr", "crypto", 2025, snapshot(roster_record(abstract="Official version")))
        roster_id = self.db.conn.execute("SELECT paper_id FROM roster_entries").fetchone()[0]
        self.assertNotEqual(api_id, roster_id)
        self.db.upsert("iacr", roster_record(doi="10.1007/linked", abstract="Official version"), "crypto", 2025)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.db.conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(self.db.conn.execute("SELECT paper_id FROM roster_entries").fetchone()[0], api_id)
        self.assertEqual(self.rows()[0]["abstract_source"], "iacr")
        self.assertEqual(store.coverage("crypto", 2025)["listed"], 1)
        self.assertEqual(store.coverage("crypto", 2025)["abstracts"], 1)

    def test_cached_dblp_identity_survives_a_database_merge_during_the_import(self):
        api_id = self.db.upsert("semantic_scholar", api_record(
            "with-doi", title="Earlier wording", externalIds={"DOI": "10.1007/linked"}), "crypto", 2025)
        self.db.upsert("semantic_scholar", api_record(
            "with-dblp", externalIds={"DBLP": "conf/crypto/Example25"}), "crypto", 2025)
        store = self.store()
        # A subsequent API record bridges identities after RosterStore built its
        # DBLP lookup; source_records now points at the surviving paper ID.
        self.db.upsert("semantic_scholar", api_record(
            "with-dblp", externalIds={"DOI": "10.1007/linked", "DBLP": "conf/crypto/Example25"}), "crypto", 2025)
        store.save("dblp", "crypto", 2025, snapshot(roster_record("conf/crypto/Example25", doi="10.1007/linked")))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(list(store.papers(["crypto"], 2025, 2025))[0]["id"], api_id)

    def test_failed_snapshot_preserves_evidence_and_successful_snapshot_removes_stale_entries(self):
        store = self.store()
        first = roster_record("first", title="First paper")
        second = roster_record("second", title="Second paper")
        store.save("dblp", "crypto", 2025, snapshot(first, second))
        self.assertTrue(store.finished("dblp", "crypto", 2025))
        store.save("dblp", "crypto", 2025, snapshot(status="incomplete"))
        self.assertFalse(store.finished("dblp", "crypto", 2025))
        self.assertEqual(store.coverage("crypto", 2025)["listed"], 2)
        store.save("dblp", "crypto", 2025, snapshot(second))
        self.assertTrue(store.finished("dblp", "crypto", 2025))
        self.assertEqual(store.coverage("crypto", 2025)["listed"], 1)
        self.assertEqual([r["title"] for r in store.papers(["crypto"], 2025, 2025)], ["Second paper"])
        self.assertEqual(len(self.rows()), 2, "Old corpus records must survive roster reconciliation")

    def test_empty_claimed_complete_snapshot_cannot_erase_previous_roster(self):
        store = self.store()
        store.save("dblp", "crypto", 2025, snapshot(roster_record()))
        store.save("dblp", "crypto", 2025, snapshot())
        self.assertFalse(store.finished("dblp", "crypto", 2025))
        self.assertEqual(store.coverage("crypto", 2025)["listed"], 1)

    def test_shared_accepted_papers_page_is_not_a_paper_identity(self):
        store = self.store()
        url = "https://www.sigsac.org/ccs/CCS2026/program/accepted-papers.html"
        store.save("ccs", "ccs", 2026, snapshot(
            roster_record("a", title="First research paper", year=2026, url=url),
            roster_record("b", title="Second research paper", year=2026, url=url)))
        self.assertEqual(store.coverage("ccs", 2026)["listed"], 2)

    def test_matching_roster_url_avoids_duplicates_when_api_titles_are_ambiguous(self):
        for ident, author in (("one", "A Author"), ("two", "B Author")):
            self.db.upsert("semantic_scholar", api_record(ident, authors=[{"name": author}]), "usenix-security", 2025)
        store = self.store()
        url = "https://www.usenix.org/conference/usenixsecurity25/presentation/example"
        store.save("dblp", "usenix-security", 2025, snapshot(roster_record("conf/uss/Example25", url=url)))
        store.save("usenix", "usenix-security", 2025, snapshot(roster_record(url, url=url, abstract="Official.")))
        self.assertEqual(store.coverage("usenix-security", 2025)["listed"], 1)
        self.assertEqual(store.coverage("usenix-security", 2025)["abstracts"], 1)

    def test_reviewed_alias_bridges_a_changed_title(self):
        store = self.store()
        store.save("dblp", "crypto", 2025, snapshot(roster_record("conf/crypto/Published25", title="Published title", doi="10.1007/example")))
        store.aliases[("iacr", "123", "crypto", 2025)] = "conf/crypto/Published25"
        store.save("iacr", "crypto", 2025, snapshot(roster_record("123", title="Earlier title", abstract="Official abstract.")))
        self.assertEqual(store.coverage("crypto", 2025)["listed"], 1)
        self.assertEqual(store.coverage("crypto", 2025)["abstracts"], 1)

    def test_reviewed_schedule_only_exclusions_adjust_counts_without_removing_api_data(self):
        store = self.store()
        exclusions = [("conf/uss/Landau18", 2018), ("conf/uss/Capkun20", 2020),
                      ("conf/uss/Rubin20", 2020), ("conf/uss/000121", 2021)]
        for identifier, year in exclusions:
            with self.subTest(identifier=identifier):
                api = api_record(identifier, year=year, title="Schedule-only contribution " + identifier)
                paper_id = self.db.upsert("semantic_scholar", api, "usenix-security", year)
                original_json = self.db.conn.execute("SELECT raw_json FROM source_records WHERE provider='semantic_scholar' AND external_id=?", (identifier,)).fetchone()[0]
                source = snapshot(roster_record(identifier, year=year, title=api["title"]),
                                  roster_record("research/" + identifier, year=year, title="Research paper " + identifier))
                adjusted = store.save("dblp", "usenix-security", year, source)
                self.assertEqual(len(source["records"]), 2, "Do not mutate collector/cached source data")
                self.assertEqual(len(adjusted["records"]), 1)
                self.assertEqual(adjusted["status"], "complete")
                run = store.coverage("usenix-security", year)["runs"][0]
                self.assertEqual(run["record_count"], 1)
                self.assertIn(identifier, run["note"])
                self.assertIn("Reviewed exclusions (1)", run["note"])
                self.assertIn(store.reviews[("dblp", identifier, "usenix-security", year)]["evidence_urls"][0], json.loads(run["urls_json"]))
                preserved = self.db.conn.execute("SELECT paper_id,raw_json FROM source_records WHERE provider='semantic_scholar' AND external_id=?", (identifier,)).fetchone()
                self.assertEqual(preserved["paper_id"], paper_id)
                self.assertEqual(preserved["raw_json"], original_json)
                self.assertFalse(self.db.conn.execute("SELECT 1 FROM roster_entries WHERE provider='dblp' AND external_id=?", (identifier,)).fetchone())

    def test_uncertain_usenix_entries_are_retained_with_persistent_review_evidence(self):
        store = self.store()
        for identifier, year, evidence_url in [
                ("conf/uss/ChoiKS24", 2024, "https://www.usenix.org/sites/default/files/sec24-contents.pdf"),
                ("conf/uss/Myung0L25", 2025, "https://www.usenix.org/sites/default/files/sec25_contents.pdf")]:
            with self.subTest(identifier=identifier):
                adjusted = store.save("dblp", "usenix-security", year, snapshot(roster_record(identifier, year=year)))
                self.assertEqual(len(adjusted["records"]), 1)
                self.assertEqual(adjusted["records"][0]["roster_review"]["action"], "annotate")
                for table in ("source_records", "roster_entries"):
                    raw = self.db.conn.execute("SELECT raw_json FROM " + table + " WHERE provider='dblp' AND external_id=?", (identifier,)).fetchone()[0]
                    review = json.loads(raw)["roster_review"]
                    self.assertIn("withdrawal has not been confirmed", review["note"])
                    self.assertIn(evidence_url, review["evidence_urls"])
                papers = list(store.papers(["usenix-security"], year, year))
                self.assertEqual(len(papers), 1)
                self.assertIn("withdrawal has not been confirmed", papers[0]["evidence"][0]["review_note"])
                self.assertIn(evidence_url, papers[0]["evidence"][0]["review_evidence_urls"])
                self.assertEqual(store.coverage("usenix-security", year)["runs"][0]["record_count"], 1)

    def test_exclusion_is_scoped_to_reviewed_event_year(self):
        store = self.store()
        adjusted = store.save("dblp", "usenix-security", 2019, snapshot(
            roster_record("conf/uss/Landau18", year=2019)))
        self.assertEqual(len(adjusted["records"]), 1)
        self.assertNotIn("roster_review", adjusted["records"][0])
        self.assertEqual(store.coverage("usenix-security", 2019)["listed"], 1)

    def test_reviewed_exclusion_removes_old_roster_membership_on_partial_rescan(self):
        store = self.store()
        review = store.reviews.pop(("dblp", "conf/uss/Landau18", "usenix-security", 2018))
        records = [roster_record("conf/uss/Landau18", year=2018, title="Invited talk"),
                   roster_record("research/one", year=2018, title="Research paper")]
        store.save("dblp", "usenix-security", 2018, snapshot(*records))
        self.assertEqual(store.coverage("usenix-security", 2018)["listed"], 2)
        store.reviews[("dblp", "conf/uss/Landau18", "usenix-security", 2018)] = review
        adjusted = store.save("dblp", "usenix-security", 2018, snapshot(records[0], status="incomplete"))
        self.assertEqual(adjusted["records"], [])
        self.assertEqual(adjusted["status"], "incomplete")
        self.assertEqual(store.coverage("usenix-security", 2018)["listed"], 1)
        self.assertEqual(len(self.rows()), 2, "Keep old corpus records and their provenance")


if __name__ == "__main__":
    unittest.main()
