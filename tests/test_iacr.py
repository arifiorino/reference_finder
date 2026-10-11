"""Offline fixtures use the structure of CryptoDB's venue/event pages."""
import unittest

from reference_finder.iacr import INDEX_URL, collect_iacr


def index_html(venue="crypto", year=2025, count=2):
    return '''<h3>Publications by venue in each year</h3>
    <table id="data-table"><thead><tr><th>Venue</th><th>Year</th><th>Count</th></tr></thead>
    <tbody><tr><td><a href="conf.php?year=%d&amp;venue=%s">%s</a></td>
    <td>%d</td><td>%d</td><td class="numbar"></td></tr></tbody></table>''' % (
        year, venue, venue.upper(), year, count)


def paper_html(identifier, title="A research paper", abstract="An abstract.", invited=False):
    # CryptoDB really repeats the class attribute on this container.  Selecting
    # .author instead of .authors keeps authors with html.parser's recovery.
    return '''<div class="row py-2 border-top"><div class="col col-auto">2025</div>
    <div class="col-3 col-md-2">CRYPTO</div><div class="col-12 col-md-9">
    <span class="pub-title"><a href="paper.php?pubkey=%s">%s</a></span>%s
    <div class="authors" class="ml-2"><span class="author"><a href="author.php?authorkey=1">Ada Researcher</a></span>
    <span class="author"><a href="author.php?authorkey=2">Zoë Scientist</a></span></div>
    %s</div></div>''' % (
        identifier, title,
        '<div class="award" title="Invited talk/paper"><span class="star">★</span>Invited talk</div>' if invited else '',
        ('<span class="abstract-toggle">Abstract</span><div class="mt-2 collapse abstract" id="abstract-%s">%s</div>'
         % (identifier, abstract)) if abstract is not None else '',
    )


def event_html(*papers, venue="crypto", year=2025):
    return '<nav><a href="invited.php">Invited talks/papers</a></nav><h3 class="mb-3">Papers from %s %d</h3>%s' % (
        venue.upper(), year, "".join(papers))


class IacrTests(unittest.TestCase):
    def collect(self, index, event, venue="crypto", year=2025):
        self.urls = []

        def fetch(url):
            self.urls.append(url)
            return index if url == INDEX_URL else event

        return collect_iacr(venue, year, fetch)

    def test_collects_inline_abstract_authors_and_missing_abstract_without_per_paper_requests(self):
        result = self.collect(index_html(), event_html(
            paper_html(123, "Cryptography &amp; privacy", "First paragraph.\n<p>Second <em>paragraph</em>.</p>"),
            paper_html(124, abstract=None)))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(self.urls), 2)
        self.assertEqual(self.urls, result["urls"])
        self.assertEqual(self.urls[1], "https://www.iacr.org/cryptodb/data/conf.php?year=2025&venue=crypto")
        first, second = result["records"]
        self.assertEqual(first["external_id"], "123")
        self.assertEqual(first["title"], "Cryptography & privacy")
        self.assertEqual(first["authors"], ["Ada Researcher", "Zoë Scientist"])
        self.assertEqual(first["abstract"], "First paragraph. Second paragraph .")
        self.assertEqual(first["url"], "https://www.iacr.org/cryptodb/data/paper.php?pubkey=123")
        self.assertIsNone(first["doi"])
        self.assertEqual(first["year"], 2025)
        self.assertIsNone(second["abstract"])

    def test_excludes_only_explicit_invited_tag_and_checks_count_before_exclusion(self):
        result = self.collect(index_html(), event_html(
            paper_html(1, abstract="This work was discussed in an invited talk."),
            paper_html(2, invited=True)))
        self.assertEqual(result["status"], "complete")
        self.assertEqual([r["external_id"] for r in result["records"]], ["1"])
        self.assertIn("1 invited entries", result["note"])
        self.assertIn("2/2 CryptoDB entries parsed", result["note"])

    def test_old_year_can_have_complete_roster_without_any_abstracts(self):
        result = self.collect(index_html(year=2017, count=1), event_html(
            paper_html(1, abstract=None), year=2017), year=2017)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["records"]), 1)
        self.assertIsNone(result["records"][0]["abstract"])

    def test_proceedings_books_are_counted_for_validation_but_not_as_papers(self):
        result = self.collect(index_html(count=2), event_html(
            paper_html(1, title="Advances in Cryptology - ASIACRYPT 2017, Proceedings, Part I"),
            paper_html(2)))
        self.assertEqual(result["status"], "complete")
        self.assertEqual([r["external_id"] for r in result["records"]], ["2"])
        self.assertIn("1 proceedings volumes excluded", result["note"])

    def test_missing_year_is_unavailable_and_never_zero_complete(self):
        result = self.collect(index_html(), "", year=2026)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["records"], [])
        self.assertEqual(self.urls, [INDEX_URL])

    def test_empty_or_challenge_index_is_incomplete(self):
        for index in ("<h1>Please wait</h1>", '<table id="data-table"><tbody></tbody></table>'):
            with self.subTest(index=index):
                result = self.collect(index, "")
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(self.urls, [INDEX_URL])

    def test_count_mismatch_preserves_recoverable_records(self):
        result = self.collect(index_html(count=3), event_html(paper_html(1), paper_html(2)))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["records"]), 2)
        self.assertIn("2/3", result["note"])

    def test_duplicate_paper_is_not_reported_complete(self):
        result = self.collect(index_html(), event_html(paper_html(1), paper_html(1)))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["records"]), 1)

    def test_unparseable_link_is_not_silently_lost(self):
        result = self.collect(index_html(count=1), event_html(
            paper_html(1).replace("paper.php?pubkey=1", "paper.php?pubkey=broken")))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["records"], [])

    def test_wrong_conference_heading_is_incomplete(self):
        result = self.collect(index_html(), event_html(paper_html(1), paper_html(2), venue="eurocrypt"))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["records"], [])

    def test_missing_event_body_is_incomplete(self):
        result = self.collect(index_html(), "<h1>Forbidden</h1>")
        self.assertEqual(result["status"], "incomplete")

    def test_index_external_domain_links_are_not_followed(self):
        index = index_html().replace("conf.php?", "https://other.example/cryptodb/data/conf.php?")
        result = self.collect(index, "")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(self.urls, [INDEX_URL])

    def test_eurocrypt_and_asiacrypt_use_exact_index_links(self):
        for venue in ("eurocrypt", "asiacrypt"):
            with self.subTest(venue=venue):
                result = self.collect(index_html(venue=venue, count=1), event_html(
                    paper_html(9), venue=venue), venue=venue)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(self.urls[1].endswith("venue=" + venue))

    def test_recognizes_optional_doi_link(self):
        event = event_html(paper_html(1)).replace(
            '<span class="pub-title">',
            '<a href="https://doi.org/10.1007/978-3-000_1">Publisher</a><span class="pub-title">')
        result = self.collect(index_html(count=1), event)
        self.assertEqual(result["records"][0]["doi"], "10.1007/978-3-000_1")

    def test_network_errors_propagate_for_caller_retry(self):
        def fetch(url):
            raise TimeoutError("network failure")
        with self.assertRaises(TimeoutError):
            collect_iacr("crypto", 2025, fetch)

    def test_unsupported_venue_rejected(self):
        with self.assertRaises(ValueError):
            collect_iacr("ccs", 2025, lambda url: "")


if __name__ == "__main__":
    unittest.main()
