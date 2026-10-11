import unittest

from reference_finder.ccs import ACCEPTED_URLS, collect_ccs


def row(title, authors="Alice Example (Example University)<br>Bob Smith (Research Institute)", attrs=""):
    return '<tr {}><td>{}</td><td>{}</td></tr>'.format(attrs, title, authors)


def table(cycle, body):
    return '<h3>{}</h3><table class="accepted-papers-table"><tr><th>Title</th><th>Author</th></tr>{}</table>'.format(cycle, body)


def page(first=None, second=None, year=2026):
    return '<html><head><title>ACM CCS {}</title></head><body><h2>ACCEPTED PAPERS</h2>{}{}</body></html>'.format(
        year, table("First Cycle", row("First Paper") if first is None else first),
        table("Second Cycle", row("Second Paper") if second is None else second))


class CcsTests(unittest.TestCase):
    def collect(self, html):
        urls = []
        def fetch(url):
            urls.append(url)
            return html
        result = collect_ccs(2026, fetch)
        self.assertEqual(urls, [ACCEPTED_URLS[2026]])
        self.assertEqual(result["urls"], urls)
        return result

    def test_both_cycles_titles_authors_and_absent_abstracts(self):
        result = self.collect(page())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual([r["section"] for r in result["records"]], ["First Cycle", "Second Cycle"])
        for record in result["records"]:
            self.assertEqual(record["authors"], ["Alice Example", "Bob Smith"])
            self.assertIsNone(record["abstract"])
            self.assertIsNone(record["doi"])
            self.assertEqual(record["url"], ACCEPTED_URLS[2026])
        self.assertIn("Provisional", result["note"])

    def test_nested_affiliations_and_inline_names_remain_one_author(self):
        result = self.collect(page(first=row("First", "Alice <em>Example</em> (Institute (Lab))<br>Bob (Bobby) Smith (University, Department)")))
        self.assertEqual(result["records"][0]["authors"], ["Alice Example", "Bob (Bobby) Smith"])

    def test_identity_is_stable_under_order_and_whitespace_changes(self):
        a = self.collect(page(first=row("Alpha Paper") + row("Beta Paper")))
        b = self.collect(page(first=row("Beta Paper") + row("Alpha   Paper")))
        ids_a = sorted(r["external_id"] for r in a["records"])
        ids_b = sorted(r["external_id"] for r in b["records"])
        self.assertEqual(ids_a, ids_b)

    def test_paper_link_preferred_and_existing_row_fragment_preserved(self):
        result = self.collect(page(first=row('<a href="https://doi.org/10.1234/paper">Linked Paper</a>'),
                                   second=row("Paper with fragment", attrs='id="paper-2"')))
        self.assertEqual(result["records"][0]["url"], "https://doi.org/10.1234/paper")
        self.assertEqual(result["records"][1]["url"], ACCEPTED_URLS[2026] + "#paper-2")

    def test_unsupported_year_never_guesses_or_fetches_a_url(self):
        def fail(_):
            self.fail("Unsupported year made a network call")
        result = collect_ccs(2025, fail)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["records"], [])
        self.assertEqual(result["urls"], [])

    def test_wrong_year_or_missing_accepted_heading_is_incomplete(self):
        for html in (page(year=2025), page().replace("ACCEPTED PAPERS", "POSTERS")):
            result = self.collect(html)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["records"], [])

    def test_missing_cycle_cannot_claim_complete(self):
        result = self.collect(page(second=""))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["records"]), 1)
        self.assertIn("Second Cycle", result["note"])

    def test_malformed_row_does_not_lose_valid_records_or_claim_complete(self):
        for malformed in ('<tr><td>Only a title</td></tr>', row(""), row("No authors", "")):
            with self.subTest(malformed=malformed):
                result = self.collect(page(first=row("Valid Paper") + malformed))
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(len(result["records"]), 2)
                self.assertIn("malformed", result["note"])

    def test_duplicate_between_cycles_is_deduplicated_but_flagged(self):
        result = self.collect(page(first=row("Same Paper"), second=row("Same Paper")))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["records"]), 1)

    def test_unknown_cycle_table_and_invalid_header_are_flagged(self):
        for html in (page().replace("Second Cycle", "Poster Session"),
                     page().replace("<th>Author</th>", "<th>Location</th>")):
            self.assertEqual(self.collect(html)["status"], "incomplete")

    def test_truncated_page_is_incomplete(self):
        self.assertEqual(self.collect(page().replace("</html>", ""))["status"], "incomplete")

    def test_fetch_error_propagates(self):
        def fail(_):
            raise OSError("HTTP 503")
        with self.assertRaises(OSError):
            collect_ccs(2026, fail)


if __name__ == "__main__":
    unittest.main()
