import unittest

from reference_finder.api import APIError, RequestLimit
from reference_finder.pets import collect_pets


def page(body, head=""):
    return '<html><head>' + head + '</head><body>' + body + '</body></html>'


def paper_link(number, year=2026, title=None, href=None):
    title = title or 'Research paper {}'.format(number)
    href = href or '../{0}/popets-{0}-{1:04d}.php'.format(year, number)
    return '<li><b><a href="{}">{}</a></b> [<a href="other.pdf">PDF</a>]<i>Alice (University)</i></li>'.format(href, title)


def volume(year=2026, issue_numbers=(1, 2, 3, 4), extra=""):
    body = '<div class="content accepted-list"><h2>Volume {}</h2>'.format(year)
    for number in issue_numbers:
        body += '<h3 id="issue{0}">Issue {0}</h3><ul>'.format(number)
        body += '<li><b><a href="editor.pdf">Editors’ Introduction</a></b></li>'
        body += paper_link(number, year) + '</ul>'
    return page(body + extra + '</div>')


def detail(number=1, year=2026, title=None, abstract="The <em>paper</em> abstract.", doi=None):
    title = title or 'Research paper {}'.format(number)
    doi = doi or '10.56553/popets-{}-{:04d}'.format(year, number)
    head = '''<meta name="citation_title" content="{title}" />
      <meta name="citation_author" content="Alice Example" />
      <meta name="citation_author" content="Bob Example" />
      <meta name="citation_publication_volume" content="{year}" />
      <meta name="citation_publication_issue" content="{number}" />'''.format(title=title, year=year, number=number)
    body = '''<div class="content"><h2>{title}</h2><p><b>Authors:</b> Alice (University), Bob (Other University)</p>
      <p><b>Volume:</b> {year}<br /><b>Issue:</b> {number}<br /><b>Pages:</b> 6–25<br />
      <b>DOI:</b> <a href="https://doi.org/{doi}">https://doi.org/{doi}</a></p>
      <p><a href="https://doi.org/10.5281/zenodo.artifact">Artifact: Available</a></p>
      <p><b>Abstract:</b> {abstract}</p><p><b>Keywords:</b> Not abstract text</p>
      <p>Copyright and license, not abstract text.</p></div>'''.format(
          title=title, year=year, number=number, doi=doi, abstract=abstract)
    return page(body, head)


class PetsTests(unittest.TestCase):
    def collect(self, year=2026, roster=None, details=None):
        calls = []
        details = details or {}
        def fetch(url):
            calls.append(url)
            if url.endswith('/'):
                return volume(year) if roster is None else roster
            number = int(url.rsplit('-', 1)[1].removesuffix('.php'))
            result = details.get(number, detail(number, year))
            if isinstance(result, BaseException):
                raise result
            return result
        result = collect_pets(year, fetch)
        self.assertEqual(result['urls'], calls)
        return result

    def test_reads_all_four_issues_and_excludes_editors(self):
        result = self.collect()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['records']), 4)
        self.assertIn('excluded 4', result['note'])
        self.assertEqual([r['section'] for r in result['records']], ['Issue 1', 'Issue 2', 'Issue 3', 'Issue 4'])
        record = result['records'][0]
        self.assertEqual(record['external_id'], 'https://petsymposium.org/popets/2026/popets-2026-0001.php')
        self.assertEqual(record['abstract'], 'The paper abstract.')
        self.assertEqual(record['authors'], ['Alice Example', 'Bob Example'])
        self.assertEqual(record['doi'], '10.56553/popets-2026-0001')

    def test_2017_layout_and_old_doi_prefix(self):
        result = self.collect(year=2017, details={1: detail(1, 2017, doi='10.1515/popets-2017-0001')})
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['records'][0]['doi'], '10.1515/popets-2017-0001')

    def test_partial_volume_retains_papers_without_claiming_complete(self):
        result = self.collect(roster=volume(issue_numbers=(1, 2)))
        self.assertEqual(len(result['records']), 2)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('3, 4', result['note'])

    def test_wrong_volume_and_layout_return_no_records(self):
        for roster in (volume(2025), page('<h2>Volume 2026</h2>')):
            with self.subTest(roster=roster):
                result = self.collect(roster=roster)
                self.assertEqual(result['records'], [])
                self.assertEqual(result['status'], 'incomplete')

    def test_failed_detail_keeps_roster_title_and_other_abstracts(self):
        result = self.collect(details={2: APIError('HTTP 503')})
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['records']), 4)
        self.assertIsNone(result['records'][1]['abstract'])
        self.assertEqual(result['records'][2]['abstract'], 'The paper abstract.')
        self.assertIn('1 paper pages failed', result['note'])

    def test_initial_network_failure_and_budget_limit_propagate(self):
        with self.assertRaises(APIError):
            collect_pets(2026, lambda _: (_ for _ in ()).throw(APIError('HTTP 404')))
        with self.assertRaises(RequestLimit):
            self.collect(details={2: RequestLimit('Stop')})

    def test_wrong_paper_title_year_or_issue_cannot_contaminate_abstracts(self):
        for wrong in (detail(1, title='An unrelated paper'), detail(1, 2025),
                      detail(1).replace('content="1"', 'content="2"')):
            with self.subTest(wrong=wrong):
                result = self.collect(details={1: wrong})
                self.assertEqual(result['status'], 'incomplete')
                self.assertIsNone(result['records'][0]['abstract'])

    def test_artifact_doi_is_not_used_as_paper_doi(self):
        no_doi = detail(1).replace('<b>DOI:</b>', '<b>Other:</b>')
        result = self.collect(details={1: no_doi})
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['records'][0]['doi'])

    def test_citation_metadata_is_used_if_present(self):
        html = detail(1).replace('</head>', '<meta name="citation_doi" content="10.1234/citation" /></head>')
        self.assertEqual(self.collect(details={1: html})['records'][0]['doi'], '10.1234/citation')

    def test_truncated_volume_and_detail_never_complete(self):
        for kwargs in ({'roster': volume().removesuffix('</html>')},
                       {'details': {1: detail(1).removesuffix('</html>')}}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.collect(**kwargs)['status'], 'incomplete')

    def test_absent_abstract_label_is_retryable_but_empty_abstract_is_allowed(self):
        absent = detail(1).replace('<b>Abstract:</b>', '<b>Description:</b>')
        self.assertEqual(self.collect(details={1: absent})['status'], 'incomplete')
        result = self.collect(details={1: detail(1, abstract='')})
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['records'][0]['abstract'])

    def test_repeated_url_is_only_fetched_once_and_www_query_fragment_normalized(self):
        duplicate = paper_link(4, href='http://www.petsymposium.org/popets/2026/popets-2026-0004.php?x=1#details')
        result = self.collect(roster=volume(extra='<ul>' + duplicate + '</ul>'))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['records']), 4)
        self.assertEqual(len(result['urls']), 5)

    def test_external_and_wrong_year_links_are_not_fetched(self):
        for href in ('https://other.example/popets/2026/popets-2026-0005.php',
                     '../2025/popets-2025-0005.php', 'popets-2026-0005.pdf'):
            with self.subTest(href=href):
                result = self.collect(roster=volume(extra='<ul>' + paper_link(5, href=href) + '</ul>'))
                self.assertEqual(result['status'], 'incomplete')
                self.assertEqual(len(result['urls']), 5)

    def test_malformed_research_entry_and_repeated_issue_are_incomplete(self):
        for extra in ('<ul><li>Unrecognized paper</li></ul>', '<h3>Issue 4</h3>'):
            self.assertEqual(self.collect(roster=volume(extra=extra))['status'], 'incomplete')

    def test_conflicting_titles_with_same_doi_are_not_silently_merged(self):
        result = self.collect(details={2: detail(2, doi='10.56553/popets-2026-0001')})
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['records']), 4)
        self.assertIsNone(result['records'][0]['doi'])
        self.assertIsNone(result['records'][1]['doi'])

    def test_matching_title_and_doi_deduplicate_papers(self):
        roster = volume().replace('Research paper 2', 'Research paper 1')
        result = self.collect(roster=roster, details={2: detail(2, title='Research paper 1', doi='10.56553/popets-2026-0001')})
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['records']), 3)

    def test_visible_fields_work_without_optional_citation_metadata(self):
        html = detail(1)
        html = html[:html.index('<head>')] + '<head></head>' + html[html.index('<body>'):]
        result = self.collect(details={1: html})
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['records'][0]['abstract'], 'The paper abstract.')
        self.assertEqual(result['records'][0]['authors'], [])

    def test_invalid_year(self):
        for year in (2014, '2026', True, 2100):
            with self.assertRaises(ValueError):
                collect_pets(year, lambda _: self.fail('Unexpected fetch'))


if __name__ == '__main__':
    unittest.main()
