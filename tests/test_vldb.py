import copy
import json
import unittest
from unittest.mock import patch

from reference_finder.api import APIError, RequestLimit
from reference_finder.vldb import collect_vldb, parse_volume


def entry(volume=19, number=1, title=None, authors="Alice Example, Bob Example"):
    return {"title": title or "Research paper %d" % number, "authors": authors,
            "issue": 1, "pdf": "https://www.vldb.org/pvldb/vol%d/p%d-example.pdf" % (volume, number)}


def summary(volume=19, number=1, title=None, authors="Alice Example, Bob Example", abstract="An abstract.", paper_id=None):
    return {"Paper Title": title or "Research paper %d" % number, "Author Names": authors,
            "Paper ID": paper_id or "vol%d/p%d-example" % (volume, number), "Abstract": abstract}


def payload(volume=19, entries=None, summaries=None):
    return {"page": "/volumes/[volume]", "query": {"volume": str(volume)},
            "props": {"pageProps": {"volume": str(volume),
                      "groupedIssues": {"1": [entry(volume)] if entries is None else entries},
                      "volumeSummaries": [summary(volume)] if summaries is None else summaries}}}


def html(data):
    return '<html><body><script id="__NEXT_DATA__" type="application/json">' + json.dumps(data) + '</script></body></html>'


def record(year=2026, volume=19, number=1, **changes):
    result = {"external_id": "journals/pvldb/Example%d" % number, "dblp_key": "journals/pvldb/Example%d" % number,
              "title": "Research paper %d" % number, "authors": ["Alice Example", "Bob Example"],
              "year": year, "publication_year": year, "volume": str(volume), "issue": "1",
              "doi": "10.14778/test.%d" % number, "url": entry(volume, number)["pdf"], "abstract": None}
    result.update(changes)
    return result


class VldbTests(unittest.TestCase):
    def collect(self, records=None, volumes=None, status="complete", year=2026):
        indexed = {"records": [record(year)] if records is None else records,
                   "status": status, "urls": ["https://sparql.dblp.org/?query=test"], "note": "DBLP note"}
        volumes = volumes or {}
        calls = []
        def fetch(url):
            calls.append(url)
            volume = int(url.rsplit("/", 1)[1])
            answer = volumes.get(volume, html(payload(volume)))
            if isinstance(answer, BaseException):
                raise answer
            return answer
        with patch("reference_finder.vldb.collect_dblp", return_value=indexed) as dblp:
            result = collect_vldb(year, fetch)
            dblp.assert_called_once_with("vldb", year, fetch)
        self.assertEqual(result["urls"], indexed["urls"] + calls)
        return result

    def test_official_abstract_and_dblp_metadata_provenance(self):
        result = self.collect()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["year_basis"], "publication")
        paper = result["records"][0]
        self.assertEqual(paper["external_id"], entry()["pdf"])
        self.assertEqual(paper["dblp_key"], "journals/pvldb/Example1")
        self.assertEqual(paper["doi"], "10.14778/test.1")
        self.assertEqual(paper["abstract"], "An abstract.")
        self.assertIn("publication year 2026", paper["section"])
        self.assertIn("not VLDB presentation years", result["note"])

    def test_publication_year_can_span_two_volumes(self):
        result = self.collect(records=[record(volume=19), record(volume=20, number=2)],
                              volumes={20: html(payload(20, [entry(20, 2)], [summary(20, 2)]))})
        self.assertEqual(result["status"], "complete")
        self.assertEqual([r["volume"] for r in result["records"]], ["19", "20"])
        self.assertEqual([r["year"] for r in result["records"]], [2026, 2026])

    def test_official_other_year_papers_cannot_create_membership(self):
        result = self.collect(volumes={19: html(payload(entries=[entry(), entry(number=2)],
                                                       summaries=[summary(), summary(number=2)]))})
        self.assertEqual(len(result["records"]), 1)

    def test_frontmatter_editorial_excluded_stale_summaries_do_not_create_papers(self):
        data = payload(entries=[entry(), {"title": "Front Matter"}, {"title": "Editorial"}],
                       summaries=[summary(), summary(number=2)])
        parsed = parse_volume(html(data), 19)
        self.assertEqual(len(parsed["papers"]), 1)
        self.assertEqual(parsed["excluded"], 2)

    def test_old_summary_numeric_ids_and_affiliations_require_full_identity(self):
        data = payload(14, [entry(14, authors="Alice Example (Example, University), Bob Example (Other)*")],
                       [summary(14, authors="Alice Example (Example, University); Bob Example (Other)*", paper_id=17)])
        parsed = parse_volume(html(data), 14)
        self.assertEqual(parsed["papers"][(14, "p1-example")]["abstract"], "An abstract.")
        data["props"]["pageProps"]["volumeSummaries"][0]["Author Names"] = "Another Author"
        self.assertIsNone(parse_volume(html(data), 14)["papers"][(14, "p1-example")]["abstract"])

    def test_missing_summary_id_and_bare_pdf_basename_are_supported(self):
        for identity in (None, "p1-example"):
            data = payload()
            data["props"]["pageProps"]["volumeSummaries"][0]["Paper ID"] = identity
            self.assertEqual(parse_volume(html(data), 19)["papers"][(19, "p1-example")]["abstract"], "An abstract.")

    def test_missing_abstract_is_not_an_incomplete_paper_inventory(self):
        result = self.collect(volumes={19: html(payload(summaries=[]))})
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["records"][0]["abstract"])
        self.assertIn("0 have official HTML abstracts", result["note"])

    def test_preserves_scientific_plaintext(self):
        abstract = r"When x < 1 and y > 0, use \nu rather than nu."
        result = self.collect(volumes={19: html(payload(summaries=[summary(abstract=abstract)]))})
        self.assertEqual(result["records"][0]["abstract"], abstract)

    def test_doi_url_requires_unique_exact_title_and_full_author_list(self):
        good = record(url="https://doi.org/10.14778/test.1", authors=["Alice Example 0001", "Bob Example"])
        result = self.collect(records=[good])
        self.assertEqual(result["status"], "complete")
        for changes in ({"authors": ["Alice Example"]}, {"title": "Different paper"}):
            bad = dict(good, **changes)
            result = self.collect(records=[bad])
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["records"], [])

    def test_ambiguous_title_author_match_is_not_guessed(self):
        result = self.collect(records=[record(url="https://doi.org/test")],
            volumes={19: html(payload(entries=[entry(), entry(number=2, title="Research paper 1")], summaries=[]))})
        self.assertEqual(result["records"], [])
        self.assertEqual(result["status"], "incomplete")

    def test_valid_but_wrong_publisher_pdf_does_not_fallback_to_title(self):
        result = self.collect(records=[record(url=entry(number=8)["pdf"])])
        self.assertEqual(result["records"], [])

    def test_mismatched_pdf_title_and_authors_are_not_imported(self):
        result = self.collect(records=[record(title="Unrelated paper", authors=["Another Author"])])
        self.assertEqual(result["records"], [])
        self.assertEqual(result["status"], "incomplete")

    def test_summary_conflict_cannot_transfer_abstract(self):
        data = payload(summaries=[summary(title="Unrelated paper", authors="Another Author")])
        with self.assertRaises(ValueError):
            parse_volume(html(data), 19)

    def test_failed_volume_retains_only_other_officially_verified_records(self):
        result = self.collect(records=[record(), record(volume=20, number=2)],
                              volumes={19: APIError("Unavailable"), 20: html(payload(20, [entry(20, 2)], []))})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual([r["volume"] for r in result["records"]], ["20"])

    def test_request_budget_propagates(self):
        with self.assertRaises(RequestLimit):
            self.collect(volumes={19: RequestLimit("Stop")})

    def test_dblp_partial_status_is_preserved(self):
        result = self.collect(status="incomplete")
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["status"], "incomplete")
        result = self.collect(records=[], status="unavailable")
        self.assertEqual(result["status"], "unavailable")

    def test_missing_volume_or_wrong_publication_year_is_not_guessed(self):
        for changes in ({"volume": None}, {"publication_year": 2025}):
            result = self.collect(records=[dict(record(), **changes)])
            self.assertEqual(result["records"], [])
            self.assertEqual(result["status"], "incomplete")

    def test_bad_volume_identity_and_truncated_html_rejected(self):
        cases = [html(payload(18)), html(payload()).removesuffix("</html>"), "<html>Blocked</html>"]
        data = payload()
        data["query"]["volume"] = "18"
        cases.append(html(data))
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    parse_volume(case, 19)

    def test_duplicates_and_wrong_issue_are_rejected(self):
        base = payload()
        for field in ("groupedIssues", "volumeSummaries"):
            data = copy.deepcopy(base)
            target = data["props"]["pageProps"][field]
            if field == "groupedIssues":
                target = target["1"]
            target.append(copy.deepcopy(target[0]))
            with self.assertRaises(ValueError):
                parse_volume(html(data), 19)
        data = copy.deepcopy(base)
        data["props"]["pageProps"]["groupedIssues"]["1"][0]["issue"] = 2
        with self.assertRaises(ValueError):
            parse_volume(html(data), 19)

    def test_external_pdf_link_is_rejected(self):
        data = payload()
        data["props"]["pageProps"]["groupedIssues"]["1"][0]["pdf"] = "https://example.org/pvldb/vol19/p1-example.pdf"
        with self.assertRaises(ValueError):
            parse_volume(html(data), 19)

    def test_legacy_publisher_pdf_urls_match(self):
        result = self.collect(records=[record(url="http://vldb.org/pvldb_wp/archive/vol19/p1-example.pdf")])
        self.assertEqual(result["status"], "complete")

    def test_invalid_year(self):
        for year in (2016, "2026", True, 2100):
            with self.assertRaises(ValueError):
                collect_vldb(year, lambda _: self.fail("Unexpected fetch"))


if __name__ == "__main__":
    unittest.main()
