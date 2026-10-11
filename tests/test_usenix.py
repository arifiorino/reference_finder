import unittest

from reference_finder.usenix import collect_usenix


def page(body):
    return "<html><body>" + body + "</body></html>"


def paper(year=2025, slug="example", title="A research paper", abstract="<p>An abstract.</p>",
          authors="Alice Example and Bob Smith, <em>First University;</em> Carol Jones, <em>Second University</em>",
          media=True):
    path = "technical-sessions/presentation/" if year == 2017 else "presentation/"
    return '''<article class="node node-paper view-mode-schedule">
      <h2><a href="/conference/usenixsecurity{yy}/{path}{slug}">{title}</a></h2>
      <div class="content"><div class="field-name-field-paper-people-text"><p>{authors}</p></div>
      {media}<div class="group-schedule-accordion">
      <div class="field-name-field-paper-description-long">{abstract}</div></div></div></article>'''.format(
          yy=str(year)[2:], path=path, slug=slug, title=title, authors=authors,
          media='<span class="usenix-schedule-media pdf"><a href="x"><img src="pdf.svg"></a></span>' if media else "",
          abstract=abstract)


class UsenixTests(unittest.TestCase):
    def collect(self, body, year=2025):
        calls = []
        def fetch(url):
            calls.append(url)
            return page(body)
        result = collect_usenix(year, fetch)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["urls"], calls)
        return result

    def test_2025_layout_abstract_and_affiliation_removal(self):
        result = self.collect('<article class="node-session"><h2>Systems Security</h2>' +
                              paper(abstract="<p>First <em>important</em> paragraph.</p><p>Second paragraph.</p>") + "</article>")
        self.assertEqual(result["status"], "complete")
        r, = result["records"]
        self.assertEqual(r["abstract"], "First important paragraph.\n\nSecond paragraph.")
        self.assertEqual(r["authors"], ["Alice Example", "Bob Smith", "Carol Jones"])
        self.assertEqual(r["section"], "Systems Security")
        self.assertEqual(r["external_id"], "https://www.usenix.org/conference/usenixsecurity25/presentation/example")

    def test_2017_layout_uses_older_presentation_path(self):
        result = self.collect(paper(year=2017), year=2017)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["records"][0]["url"],
                         "https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/example")

    def test_excludes_enigma_and_invited_talks_not_their_biographies(self):
        invited = '''<article class="node-paper"><h2><a href="/conference/usenixsecurity25/presentation/talk">Invited talk</a></h2>
          <div class="field-name-field-presented-by">A Speaker</div>
          <div class="field-name-field-paper-description-long">Talk abstract</div>
          <article class="node-speaker"><div class="field-name-body">Biography</div></article></article>'''
        result = self.collect(paper() + invited + '<article class="node-session"><h2>Enigma Track 1</h2>' +
                              paper(slug="enigma") + "</article>")
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["status"], "complete")
        self.assertIn("excluded 2", result["note"])

    def test_keeps_research_without_abstract_or_doi(self):
        result = self.collect(paper(abstract="", media=False))
        r, = result["records"]
        self.assertIsNone(r["abstract"])
        self.assertIsNone(r["doi"])
        self.assertEqual(result["status"], "complete")

    def test_unclassified_entry_does_not_claim_complete(self):
        result = self.collect(paper() + '<article class="node-paper"><h2>Unrecognized entry</h2></article>')
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("1 entries", result["note"])

    def test_duplicate_presentation_deduplicated_with_abstract_filled(self):
        result = self.collect(paper(abstract="") + paper())
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["abstract"], "An abstract.")

    def test_empty_or_unavailable_source_never_complete(self):
        for body in ("<h1>New layout</h1>", "<h1>Page not found</h1>"):
            with self.subTest(body=body):
                result = self.collect(body)
                self.assertNotEqual(result["status"], "complete")
                self.assertEqual(result["records"], [])

    def test_truncated_response_is_incomplete(self):
        result = collect_usenix(2025, lambda _: "<html>" + paper())
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("truncated", result["note"])

    def test_fetch_failure_propagates(self):
        def fetch(_):
            raise OSError("HTTP 503")
        with self.assertRaises(OSError):
            collect_usenix(2025, fetch)

    def test_wrong_year_link_or_external_link_is_not_silently_accepted(self):
        for html in (paper(year=2017), paper().replace('/conference/usenixsecurity25/', 'https://other.example/conference/usenixsecurity25/')):
            result = self.collect(html)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["records"], [])

    def test_explicit_citation_doi(self):
        html = paper().replace('<div class="content">', '<div class="content"><meta name="citation_doi" content="https://doi.org/10.1234/test">')
        self.assertEqual(self.collect(html)["records"][0]["doi"], "10.1234/test")

    def test_artifact_doi_in_abstract_is_not_the_papers_identifier(self):
        # The actual 2025 Ghosh schedule entry links its research artifact in
        # the abstract; treating that Zenodo DOI as the paper breaks matching.
        abstract = '<p>Our artifact is available at <a href="https://doi.org/10.5281/zenodo.15614029">Zenodo</a>.</p>'
        record = self.collect(paper(abstract=abstract))["records"][0]
        self.assertIsNone(record["doi"])
        self.assertIn("Our artifact is available", record["abstract"])

    def test_citation_doi_wins_over_unrelated_abstract_links(self):
        html = paper(abstract='<p>Dataset: <a href="https://doi.org/10.5281/zenodo.15614029">data</a></p>')
        html = html.replace('<div class="content">', '<div class="content"><meta name="citation_doi" content="https://doi.org/10.1234/PAPER">')
        self.assertEqual(self.collect(html)["records"][0]["doi"], "10.1234/paper")

    def test_invalid_citation_doi_does_not_fall_back_to_dataset(self):
        html = paper(abstract='<p><a href="https://doi.org/10.5281/zenodo.15614029">data</a></p>')
        html = html.replace('<div class="content">', '<div class="content"><meta name="citation_doi" content="not a DOI">')
        self.assertIsNone(self.collect(html)["records"][0]["doi"])

    def test_no_affiliation_markup_does_not_invent_authors(self):
        self.assertEqual(self.collect(paper(authors="Alice, University X"))["records"][0]["authors"], [])

    def test_oxford_comma_does_not_prefix_author_with_and(self):
        result = self.collect(paper(authors="Alice Example, Bob Smith, and Carol Jones, <em>University</em>"))
        self.assertEqual(result["records"][0]["authors"], ["Alice Example", "Bob Smith", "Carol Jones"])

    def test_2026_nonstandard_research_url_is_retained(self):
        html = paper(year=2026).replace("/presentation/example", "/example")
        result = self.collect(html, year=2026)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["records"][0]["url"], "https://www.usenix.org/conference/usenixsecurity26/example")

    def test_test_of_time_reprise_does_not_become_current_year_paper(self):
        html = paper(year=2026) + '<article class="node-session"><h2>Discussion with Winners of the Test of Time Award</h2>' + paper(year=2026, slug="tot-discussion", authors="Originally presented in 2016.", abstract="", media=False) + '</article>'
        result = self.collect(html, year=2026)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["status"], "complete")

    def test_empty_lightning_talk_container_does_not_invalidate_roster(self):
        html = paper(year=2019) + '<article class="node-session"><h2>Lightning Talks</h2><article class="node-paper"><h2><a href="/conference/usenixsecurity19/presentation/lightning-talks"></a></h2><div class="field-name-field-paper-description-long"><ul><li>A talk</li></ul></div></article></article>'
        result = self.collect(html, year=2019)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
