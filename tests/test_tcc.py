import json
import unittest

from reference_finder.api import APIError, RequestLimit
from reference_finder.tcc import ACCEPTED_URL, PAPERS_URL, collect_tcc


PAGE = '''<html><head><title>TCC 2026 Accepted Papers</title></head><body>
<h1>TCC 2026</h1><h2>Accepted Papers</h2>
<div id="notYetAvailable">This information is not yet available.</div>
<script>fetch('json/papers.json', {credentials: 'same-origin'})
 .then(response => response.json()).then(data => render(data.acceptedPapers));</script>
</body></html>'''


def paper(identifier=2, **changes):
    entry = {"paperId": identifier, "pubkey": 38101 + identifier,
             "title": "Event Algebras and Applications to Cryptography " + str(identifier),
             "authors": ["Konstantin Gegier", "Ueli Maurer"],
             "abstract": "A monotone predicate obeys a < b,\r\n including a proof.",
             "affiliations": ["ETH Zurich"], "pages": 33}
    entry.update(changes)
    return entry


class TccTests(unittest.TestCase):
    def collect(self, entries=None, page=PAGE, payload=None):
        calls = []
        if payload is None:
            payload = json.dumps({"_source": "IACR/hotcrp v2", "acceptedPapers":
                                  [paper()] if entries is None else entries,
                                  "invitedTalks": [{"title": "Not a research paper"}]})
        def fetch(url):
            calls.append(url)
            return page if url == ACCEPTED_URL else payload
        result = collect_tcc(2026, fetch)
        self.assertEqual(result["urls"], calls)
        return result

    def test_accepted_json_with_static_unavailable_placeholder_is_complete(self):
        result = self.collect(entries=[paper(), paper(5)])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 2)
        self.assertIn("2/2 accepted research papers and 2 abstracts", result["note"])
        record = result["records"][0]
        self.assertEqual(record["external_id"], "38103")
        self.assertEqual(record["url"], "https://www.iacr.org/cryptodb/data/paper.php?pubkey=38103")
        self.assertEqual(record["year"], 2026)
        self.assertIsNone(record["doi"])
        self.assertEqual(record["authors"], ["Konstantin Gegier", "Ueli Maurer"])
        self.assertEqual(record["abstract"], "A monotone predicate obeys a < b, including a proof.")

    def test_uses_year_scoped_id_if_publication_key_is_not_yet_assigned(self):
        result = self.collect(entries=[paper(pubkey=None)])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["records"][0]["external_id"], "tcc:2026:paper:2")
        self.assertEqual(result["records"][0]["url"], ACCEPTED_URL)

    def test_missing_abstract_is_valid_roster_not_fabricated_from_other_fields(self):
        result = self.collect(entries=[paper(abstract=None, description="Not the abstract")])
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["records"][0]["abstract"])
        self.assertIn("0 abstracts", result["note"])

    def test_wrong_conference_year_missing_link_or_truncated_html_cannot_import(self):
        for html in (PAGE.replace("2026", "2025"), PAGE.replace("json/papers.json", "foreign.json"),
                     PAGE.replace("Accepted Papers</h2>", "Program</h2>"), PAGE.replace("</html>", "")):
            with self.subTest(html=html):
                result = self.collect(page=html)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["records"], [])

    def test_invalid_json_and_wrong_schema_are_retryable(self):
        for payload in ('{"acceptedPapers":[', '{}', '[]', '{"acceptedPapers":{}}'):
            with self.subTest(payload=payload):
                result = self.collect(payload=payload)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["records"], [])

    def test_empty_accepted_roster_is_unavailable(self):
        result = self.collect(entries=[])
        self.assertEqual(result["status"], "unavailable")

    def test_duplicate_paper_or_publication_ids_do_not_create_conflicting_records(self):
        for entry in (paper(title="A conflicting title"), paper(5, pubkey=38103)):
            result = self.collect(entries=[paper(), entry])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["records"]), 1)

    def test_malformed_entries_retain_other_papers_without_completed_checkpoint(self):
        for entry in (None, {}, paper(paperId=True), paper(title=123)):
            result = self.collect(entries=[paper(5), entry])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["records"]), 1)

    def test_malformed_optional_fields_keep_title_and_mark_incomplete(self):
        for changes in ({"authors": "Alice"}, {"authors": [None]}, {"abstract": {}}, {"pubkey": False}):
            result = self.collect(entries=[paper(**changes)])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(len(result["records"]), 1)

    def test_network_errors_and_request_limits_propagate_at_both_steps(self):
        for failed_url in (ACCEPTED_URL, PAPERS_URL):
            for error in (APIError("HTTP 503"), RequestLimit("Stop")):
                with self.subTest(url=failed_url, error=error):
                    def fetch(url):
                        if url == failed_url:
                            raise error
                        return PAGE
                    with self.assertRaises(type(error)):
                        collect_tcc(2026, fetch)

    def test_other_years_are_not_claimed_to_use_this_layout(self):
        for year in (2025, 2027, True, "2026"):
            with self.assertRaises(ValueError):
                collect_tcc(year, lambda _: self.fail("Unexpected fetch"))


if __name__ == "__main__":
    unittest.main()
