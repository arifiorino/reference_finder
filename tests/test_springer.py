import json
from pathlib import Path
import unittest

from reference_finder.api import RequestLimit
from reference_finder.springer import enrich_springer, parse_springer_abstract
from reference_finder.storage import Database
from reference_finder.webfetch import FetchError


DOI = "10.1007/978-3-662-54365-8_9"
TITLE = "Tightly Secure IBE Under Constant-Size Master Public Key"


def page(doi=DOI, title=TITLE, abstract="<p>A complete abstract.</p>", authors=True):
    author_meta = '<meta name="citation_author" content="Chen, Jie">' if authors else ""
    section = ('''<section><div class="c-article-section" id="Abs1-section">
      <h2>Abstract</h2><div class="c-article-section__content" id="Abs1-content">%s</div>
      </div></section>''' % abstract) if abstract is not None else ""
    return '''<html><head><title>%s | Springer Nature Link</title>
      <meta name="citation_doi" content="%s"><meta name="citation_title" content="%s">
      <meta name="citation_publication_date" content="2017">%s
      <meta name="description" content="This metadata is a truncated summary, not the abstract.">
      </head><body><h1>%s</h1>%s<section><h2>References</h2>Unrelated text.</section></body></html>''' % (
        title, doi, title, author_meta, title, section)


class SpringerParserTests(unittest.TestCase):
    def test_extracts_only_explicit_abstract_and_preserves_paragraphs_math(self):
        html = page(abstract='<p>We prove <em>privacy</em> for <img alt="x + y" src="math.svg">.</p>'
                    '<p>Then <script type="math/tex">z=1</script> follows.</p><script>tracking()</script>')
        result = parse_springer_abstract(html, DOI, TITLE)
        self.assertEqual(result["abstract"], "We prove privacy for x + y .\n\nThen z=1 follows.")
        self.assertEqual(result["authors"], ["Chen, Jie"])
        self.assertNotIn("References", result["abstract"])
        self.assertNotIn("truncated summary", result["abstract"])

    def test_doi_and_title_normalization_accepts_only_typography_changes(self):
        result = parse_springer_abstract(page(doi="https://doi.org/" + DOI.upper(), title=TITLE + "."), DOI, TITLE.lower())
        self.assertTrue(result["abstract"])
        result = parse_springer_abstract(page(title=TITLE.replace("Constant", "Con\u00adstant")), DOI, TITLE)
        self.assertTrue(result["abstract"])

    def test_legacy_dblp_duplicate_math_rendering_is_matched_narrowly(self):
        prefix = "Near-Optimal Secret Sharing and Error Correcting Codes in "
        published = prefix + r"$$\mathsf {AC}^0$$"
        for existing in (prefix + r"\mathsf AC^0 AC 0.", prefix + r"\mathsf {AC}^{0}."):
            with self.subTest(existing=existing):
                self.assertTrue(parse_springer_abstract(page(title=published), DOI, existing)["abstract"])
        invalid_titles = [
            prefix + r"\mathsf AC^1 AC 1.",
            prefix + r"\mathsf TC^0 TC 0.",
            prefix + r"\mathsf AC^0 AC 1.",
            prefix + r"\mathrm AC^0 AC 0.",
            prefix + r"\mathsf AC^0 + 1 AC 0.",
            "Different research in " + r"\mathsf AC^0 AC 0.",
        ]
        for existing in invalid_titles:
            with self.subTest(existing=existing), self.assertRaises(ValueError):
                parse_springer_abstract(page(title=published), DOI, existing)
        with self.assertRaises(ValueError):
            parse_springer_abstract(page(doi="10.1007/other", title=published), DOI, prefix + r"\mathsf AC^0 AC 0.")

    def test_explicit_chapter_subtitle_completes_citation_title(self):
        main = "Exploring Crypto Dark Matter:"
        subtitle = "New Simple PRF Candidates and Their Applications"
        full = main + " - " + subtitle + "."
        heading = '<h1 class="c-article-title" data-test="chapter-title">' + main + '</h1>'
        sub = '<p class="c-chapter-category" lang="en">' + subtitle + '</p>'
        html = page(title=main).replace('<h1>' + main + '</h1>', heading + sub)
        self.assertTrue(parse_springer_abstract(html, DOI, full)["abstract"])

        invalid_pages = [
            html.replace(heading, heading.replace(main, "A different chapter:")),
            html.replace(sub, sub.replace(subtitle, "An unrelated subtitle")),
            html.replace(sub, '<p>' + subtitle + '</p>'),
            html.replace(heading + sub, heading + '<div>Intervening content</div>' + sub),
            html.replace(heading + sub, heading + '<section>' + sub + '</section>'),
            html.replace(heading + sub, heading).replace('</body>', '<section>' + sub + '</section></body>'),
            html.replace(heading, heading + heading),
            html.replace('name="citation_doi" content="' + DOI + '"', 'name="citation_doi" content="10.1007/other"'),
        ]
        for bad in invalid_pages:
            with self.subTest(html=bad[:80]), self.assertRaises(ValueError):
                parse_springer_abstract(bad, DOI, full)

    def test_final_math_identifier_preserves_font_and_identifier(self):
        prefix = "Attribute Based Encryption for Deterministic Finite Automata from "
        published = prefix + r"$$\mathsf{DLIN}$$"
        self.assertTrue(parse_springer_abstract(page(title=published), DOI, prefix + r"\mathsfDLIN.")["abstract"])
        for tail in (r"\mathsfXDLIN.", r"\mathrmDLIN.", r"\mathsfDLIN^0.", "DLIN."):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                parse_springer_abstract(page(title=published), DOI, prefix + tail)

    def test_lost_math_structure_is_not_guessed(self):
        prefix = "Efficient Information-Theoretic Secure Multiparty Computation over "
        published = prefix + r"$$\mathbb {Z}/p^k\mathbb {Z}$$ via Galois Rings"
        with self.assertRaises(ValueError):
            parse_springer_abstract(page(title=published), DOI, prefix + "Z/pkZ via Galois Rings.")

    def test_reviewed_math_alias_requires_exact_doi_and_both_exact_titles(self):
        path = Path(__file__).parents[1] / "reference_finder" / "springer_title_aliases.json"
        alias = json.loads(path.read_text(encoding="utf-8"))[0]
        doi, existing, published = alias["doi"], alias["inventory_title"], alias["publisher_title"]
        self.assertTrue(parse_springer_abstract(page(doi=doi, title=published), doi, existing)["abstract"])
        for requested, stored, citation in (
            ("10.1007/another-paper", existing, published),
            (doi, existing.replace("pkZ", "pjZ"), published),
            (doi, existing, published.replace("p^k", "p^j")),
            (doi, existing, published.replace("\u00a0", " ")),
        ):
            with self.subTest(doi=requested, stored=stored, citation=citation), self.assertRaises(ValueError):
                parse_springer_abstract(page(doi=requested, title=citation), requested, stored)
        with self.assertRaises(ValueError):
            parse_springer_abstract(page(doi="10.1007/another-paper", title=published), doi, existing)

    def test_reviewed_identifier_aliases_preserve_identifiers(self):
        cases = [
            ("10.1007/978-3-030-64381-2_22", "On Average-Case Hardness in TFNP from One-Way Functions.",
             r"On Average-Case Hardness in $$\mathsf {TFNP}$$ from One-Way Functions", "TFNP", "TFP"),
            ("10.1007/978-3-030-64381-2_2", "Security Analysis of itSPAKE2+.",
             r"Security Analysis of $$\textit{SPAKE2}+$$", "SPAKE2", "SPAKE3"),
            ("10.1007/978-3-031-22318-1_19", "Fully Succinct Batch Arguments for sfNP from Indistinguishability Obfuscation.",
             "Fully Succinct Batch Arguments for\u00a0 $$\\textsf{NP}$$ from\u00a0Indistinguishability Obfuscation", "NP", "RP"),
            ("10.1007/978-3-032-12293-3_10", "(Multi-input) sfFE for Randomized Functionalities, Revisited.",
             "(Multi-input) $$\\textsf{FE}$$ for\u00a0Randomized Functionalities, Revisited", "FE", "FHE"),
            ("10.1007/978-3-032-12296-4_11", "Dimensional esfROSion: Improving the sfROS Attack with Decomposition in Higher Bases.",
             "Dimensional e $$\\textsf{ROS}$$ ion: Improving the\u00a0 $$\\textsf{ROS}$$ Attack with\u00a0Decomposition in\u00a0Higher Bases", "ROS", "SOS"),
            ("10.1007/978-3-031-57718-5_8", "sfReSolveD: Shorter Signatures from Regular Syndrome Decoding and VOLE-in-the-Head.",
             "$$\\textsf{ReSolveD} $$ : Shorter Signatures from\u00a0Regular Syndrome Decoding and\u00a0VOLE-in-the-Head", "ReSolveD", "OtherProtocol"),
            ("10.1007/978-3-031-91823-0_5", "Security Analysis of Signal's PQXDH Handshake.",
             "Security Analysis of\u00a0Signal’s $$\\textsf{PQXDH} $$ Handshake", "PQXDH", "X3DH"),
            ("10.1007/978-3-032-12290-2_4", "sfJustvengers: Batched VOLE ZK Disjunctions in 풪(R+B+C) Communication.",
             "$$\\textsf{Justvengers}$$ : Batched VOLE ZK Disjunctions in\u00a0 $${\\mathcal {O}}(R{+}B{+}C)$$ Communication", "Justvengers", "OtherProtocol"),
        ]
        for doi, existing, published, term, other in cases:
            with self.subTest(doi=doi):
                self.assertTrue(parse_springer_abstract(page(doi=doi, title=published), doi, existing)["abstract"])
                with self.assertRaises(ValueError):
                    parse_springer_abstract(page(doi=doi, title=published.replace(term, other)), doi, existing)
                with self.assertRaises(ValueError):
                    parse_springer_abstract(page(doi=doi, title=published), doi, existing.replace(term, other))
                with self.assertRaises(ValueError):
                    parse_springer_abstract(page(doi="10.1007/unrelated", title=published), "10.1007/unrelated", existing)
        with self.assertRaises(ValueError):
            parse_springer_abstract(page(doi=cases[1][0], title=cases[1][2].replace("+", "")), cases[1][0], cases[1][1])
        with self.assertRaises(ValueError):
            parse_springer_abstract(page(doi=cases[-1][0], title=cases[-1][2].replace("{+}", "{-}", 1)), cases[-1][0], cases[-1][1])

    def test_no_abstract_or_no_authors_are_not_invented_from_other_content(self):
        result = parse_springer_abstract(page(abstract=None, authors=False), DOI, TITLE)
        self.assertIsNone(result["abstract"])
        self.assertEqual(result["authors"], [])
        self.assertIsNone(parse_springer_abstract(page(abstract="<p>No abstract available.</p>"), DOI, TITLE)["abstract"])

    def test_reviewed_trademark_alias_requires_exact_doi_and_titles(self):
        doi = "10.1007/978-3-319-76578-5_13"
        existing = "A Practical Cryptanalysis of WalnutDSA TM."
        published = r"A Practical Cryptanalysis of WalnutDSA $$^{\text {TM}}$$"
        self.assertTrue(parse_springer_abstract(page(doi=doi, title=published), doi, existing)["abstract"])
        for requested, stored, citation in (
            ("10.1007/another-paper", existing, published),
            (doi, existing.replace("WalnutDSA", "AnotherDSA"), published),
            (doi, existing, published.replace("WalnutDSA", "AnotherDSA")),
            (doi, existing.replace("TM", "R"), published),
            (doi, existing, published.replace("TM", "R")),
        ):
            with self.subTest(doi=requested, stored=stored, citation=citation), self.assertRaises(ValueError):
                parse_springer_abstract(page(doi=requested, title=citation), requested, stored)

    def test_reviewed_ibe_abe_alias_preserves_identifiers_and_slash(self):
        doi = "10.1007/978-3-030-97131-1_8"
        existing = "Leakage-Resilient IBE/ ABE with Optimal Leakage Rates from Lattices."
        published = r"Leakage-Resilient $$\mathsf {IBE}$$ / $$\mathsf {ABE}$$ with Optimal Leakage Rates from Lattices"
        self.assertTrue(parse_springer_abstract(page(doi=doi, title=published), doi, existing)["abstract"])
        for term, changed in (("IBE", "IBS"), ("ABE", "ABS"), ("/", "+")):
            with self.subTest(term=term):
                with self.assertRaises(ValueError):
                    parse_springer_abstract(page(doi=doi, title=published.replace(term, changed)), doi, existing)
                with self.assertRaises(ValueError):
                    parse_springer_abstract(page(doi=doi, title=published), doi, existing.replace(term, changed))
        with self.assertRaises(ValueError):
            parse_springer_abstract(page(doi="10.1007/unrelated", title=published), "10.1007/unrelated", existing)

    def test_reviewed_pi_alias_preserves_the_protocol_symbol(self):
        doi = "10.1007/978-3-031-91829-2_4"
        existing = "Π: A Unified Framework for Computational Verifiable Secret Sharing."
        published = "$${\\Pi }$$ : A Unified Framework for\u00a0Computational Verifiable Secret Sharing"
        self.assertTrue(parse_springer_abstract(page(doi=doi, title=published), doi, existing)["abstract"])
        for requested, stored, citation in (
            ("10.1007/another-paper", existing, published),
            (doi, existing.replace("Π", "Σ"), published),
            (doi, existing, published.replace(r"\Pi", r"\Sigma")),
            (doi, existing, published.replace(r"\Pi", r"\pi")),
            (doi, existing.replace("Computational", "Statistical"), published),
        ):
            with self.subTest(doi=requested, stored=stored, citation=citation), self.assertRaises(ValueError):
                parse_springer_abstract(page(doi=requested, title=citation), requested, stored)

    def test_reviewed_rlwe_alias_preserves_assumptions_and_direction(self):
        doi = "10.1007/978-3-032-26731-3_6"
        existing = "Improved Reduction from sfRLWE to sfMP -sfLWE."
        published = "Improved Reduction from\u00a0 $$\\textsf{RLWE}$$ to\u00a0 $$\\textsf{MP}\\text {-}\\textsf{LWE}$$"
        self.assertTrue(parse_springer_abstract(page(doi=doi, title=published), doi, existing)["abstract"])
        for term, changed in (("RLWE", "MLWE"), ("MP", "SP"), ("LWE", "SIS"), ("from", "to"), ("-", "+")):
            with self.subTest(term=term):
                with self.assertRaises(ValueError):
                    parse_springer_abstract(page(doi=doi, title=published.replace(term, changed)), doi, existing)
                with self.assertRaises(ValueError):
                    parse_springer_abstract(page(doi=doi, title=published), doi, existing.replace(term, changed))
        with self.assertRaises(ValueError):
            parse_springer_abstract(page(doi="10.1007/unrelated", title=published), "10.1007/unrelated", existing)

    def test_wrong_or_missing_doi_title_and_truncated_pages_are_rejected(self):
        bad_pages = [
            page(doi="10.1007/other"), page(title="An entirely unrelated publication"),
            page().replace('name="citation_doi"', 'name="other_doi"'),
            page().replace('name="citation_title"', 'name="other_title"'), page()[:-7],
            page().replace("<head>", '<head><meta name="citation_doi" content="10.1007/other">'),
            page().replace("<head>", '<head><meta name="citation_title" content="Another title">'),
            page().replace("<title>", "<title>Access denied "),
        ]
        for html in bad_pages:
            with self.subTest(html=html[:80]), self.assertRaises(ValueError):
                parse_springer_abstract(html, DOI, TITLE)

    def test_unsupported_doi_rejected(self):
        for doi in ("10.1145/123", "not a doi", None):
            with self.subTest(doi=doi), self.assertRaises(ValueError):
                parse_springer_abstract(page(), doi, TITLE)

    def test_abstract_id_without_explicit_heading_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_springer_abstract(page().replace("<h2>Abstract</h2>", "<h2>Biography</h2>"), DOI, TITLE)

    def test_visible_abstract_with_unknown_container_is_retryable(self):
        html = page().replace('id="Abs1-content"', 'id="new-abstract-content"')
        html = html.replace('class="c-article-section__content"', 'class="new-section-layout"')
        with self.assertRaisesRegex(ValueError, "unrecognized content layout"):
            parse_springer_abstract(html, DOI, TITLE)


class SpringerEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.addCleanup(self.db.conn.close)

    def add(self, doi=DOI, title=TITLE, year=2017, venue="pkc", authors=None, abstract=None):
        with self.db.conn:
            return self.db.upsert("dblp", {
                "external_id": "conf/" + venue + "/" + doi,
                "doi": doi, "title": title, "year": year,
                "authors": ["Existing Author"] if authors is None else authors,
                "abstract": abstract,
            }, venue, year)

    def row(self, paper_id):
        return self.db.conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()

    def attempts(self):
        return self.db.conn.execute("SELECT doi,outcome FROM enrichment_attempts WHERE provider='springer' ORDER BY doi").fetchall()

    def test_enriches_known_paper_preserving_inventory_and_without_new_memberships(self):
        paper_id = self.add()
        visited = []

        def fetch(url):
            visited.append(url)
            return page(title=TITLE + ".")

        result = enrich_springer(self.db, ["pkc"], 2017, 2026, fetch)
        self.assertEqual(result["status"], "complete")
        self.assertEqual((result["checked"], result["filled"]), (1, 1))
        self.assertEqual(visited, ["https://link.springer.com/chapter/" + DOI])
        row = self.row(paper_id)
        self.assertEqual(row["abstract"], "A complete abstract.")
        self.assertEqual(row["abstract_source"], "springer")
        self.assertEqual(row["title"], TITLE)
        self.assertEqual(json.loads(row["authors_json"]), ["Existing Author"])
        self.assertEqual(row["publication_year"], 2017)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 1)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM paper_venues").fetchone()[0], 1)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM roster_entries").fetchone()[0], 0)
        self.assertEqual([tuple(r) for r in self.attempts()], [(DOI, "abstract")])

    def test_missing_publisher_authors_preserves_existing_names(self):
        paper_id = self.add()
        enrich_springer(self.db, ["pkc"], 2017, 2026, lambda url: page(authors=False))
        self.assertEqual(json.loads(self.row(paper_id)["authors_json"]), ["Existing Author"])

    def test_publisher_authors_fill_empty_inventory_authors(self):
        paper_id = self.add(authors=[])
        enrich_springer(self.db, ["pkc"], 2017, 2026, lambda url: page())
        self.assertEqual(json.loads(self.row(paper_id)["authors_json"]), ["Chen, Jie"])

    def test_valid_page_without_abstract_is_checkpointed_without_source_record(self):
        self.add()
        result = enrich_springer(self.db, ["pkc"], 2017, 2026, lambda url: page(abstract=None))
        self.assertEqual((result["filled"], result["no_abstract"]), (0, 1))
        self.assertEqual([tuple(r) for r in self.attempts()], [(DOI, "no_abstract")])
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM source_records WHERE provider='springer'").fetchone()[0], 0)
        resumed = enrich_springer(self.db, ["pkc"], 2017, 2026, lambda url: self.fail("Should not repeat completed lookup"))
        self.assertEqual(resumed["checked"], 0)

    def test_identity_errors_do_not_change_paper_and_remain_retryable(self):
        for bad in (page(doi="10.1007/other"), page(title="Different paper"), page()[:-7]):
            with self.subTest(bad=bad[:30]):
                paper_id = self.add()
                result = enrich_springer(self.db, ["pkc"], 2017, 2026, lambda url: bad)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(len(result["errors"]), 1)
                self.assertIsNone(self.row(paper_id)["abstract"])
                self.assertEqual(self.row(paper_id)["title"], TITLE)
                self.assertEqual(self.attempts(), [])

    def test_partial_progress_survives_block_and_remaining_papers_resume(self):
        self.add()
        self.add(doi="10.1007/second")
        self.add(doi="10.1007/third")
        visited = []

        def fetch(url):
            visited.append(url)
            if len(visited) > 1:
                raise FetchError("link.springer.com: HTTP 429; resume later")
            return page()

        result = enrich_springer(self.db, ["pkc"], 2017, 2026, fetch)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual((result["checked"], result["filled"]), (2, 1))
        self.assertEqual(len(visited), 2)
        self.assertEqual([tuple(r) for r in self.attempts()], [(DOI, "abstract")])
        self.assertEqual(len(self.db.missing_for("springer", ["pkc"], 2017, 2026)), 2)

    def test_isolated_fetch_failure_preserves_progress_without_false_checkpoint(self):
        self.add()
        second = self.add(doi="10.1007/second")

        def fetch(url):
            if url.endswith(DOI):
                raise FetchError("link.springer.com: HTTP 404 (/chapter/missing)")
            return page(doi="10.1007/second")

        result = enrich_springer(self.db, ["pkc"], 2017, 2026, fetch)
        self.assertEqual((result["checked"], result["filled"]), (2, 1))
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(self.row(second)["abstract"])
        self.assertEqual([tuple(r) for r in self.attempts()], [("10.1007/second", "abstract")])

    def test_html_challenges_stop_provider_without_finished_checkpoints(self):
        self.add()
        self.add(doi="10.1007/second")
        responses = [
            FetchError("link.springer.com: bot challenge; page was not parsed or cached."),
            FetchError("link.springer.com: CAPTCHA verification required"),
            FetchError("link.springer.com: Access denied"),
            page().replace("<title>", "<title>Access denied "),
            page().replace("<title>", "<title>CAPTCHA "),
            page().replace("<title>", "<title>Just a moment "),
        ]
        for response in responses:
            with self.subTest(response=str(response)[:70]):
                visited = []

                def fetch(url):
                    visited.append(url)
                    if isinstance(response, Exception):
                        raise response
                    return response

                result = enrich_springer(self.db, ["pkc"], 2017, 2026, fetch)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(len(visited), 1)
                self.assertEqual(result["filled"], 0)
                self.assertEqual(self.attempts(), [])

    def test_unknown_abstract_layout_is_not_checkpointed_and_other_papers_continue(self):
        first = self.add()
        second = self.add(doi="10.1007/second")
        unknown = page().replace('id="Abs1-content"', 'id="new-abstract-content"')
        unknown = unknown.replace('class="c-article-section__content"', 'class="new-section-layout"')

        def fetch(url):
            return unknown if url.endswith(DOI) else page(doi="10.1007/second")

        result = enrich_springer(self.db, ["pkc"], 2017, 2026, fetch)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual((result["checked"], result["filled"]), (2, 1))
        self.assertIsNone(self.row(first)["abstract"])
        self.assertTrue(self.row(second)["abstract"])
        self.assertEqual([tuple(r) for r in self.attempts()], [("10.1007/second", "abstract")])

    def test_unsupported_dois_do_not_block_later_supported_records(self):
        for n in range(101):
            self.add(doi="10.1145/%d" % n)
        self.add()
        result = enrich_springer(self.db, ["pkc"], 2017, 2026, lambda url: page())
        self.assertEqual(result["skipped_unsupported"], 101)
        self.assertEqual(result["filled"], 1)
        self.assertEqual(len(self.attempts()), 1)

    def test_only_selected_missing_venue_year_records_are_enriched(self):
        self.add(doi="10.1007/existing", abstract="Already available.")
        self.add(doi="10.1007/old", year=2016)
        self.add(doi="10.1007/other", venue="crypto")
        self.add()
        result = enrich_springer(self.db, ["pkc"], 2017, 2026, lambda url: page())
        self.assertEqual((result["candidates"], result["filled"]), (1, 1))

    def test_budget_stop_propagates_after_committed_progress(self):
        first = self.add()
        self.add(doi="10.1007/second")

        def fetch(url):
            if url.endswith(DOI):
                return page()
            raise RequestLimit("stop")

        with self.assertRaises(RequestLimit):
            enrich_springer(self.db, ["pkc"], 2017, 2026, fetch)
        self.assertTrue(self.row(first)["abstract"])
        self.assertEqual(len(self.attempts()), 1)

    def test_empty_selection_performs_no_fetches(self):
        self.add()
        self.assertEqual(enrich_springer(self.db, [], 2017, 2026, lambda url: self.fail("Unexpected fetch"))["checked"], 0)


if __name__ == "__main__":
    unittest.main()
