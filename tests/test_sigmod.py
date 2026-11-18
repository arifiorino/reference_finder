import unittest

from reference_finder.api import APIError, RequestLimit
from reference_finder.sigmod import CONFERENCE_ISSUES, OPEN_TOCS, collect_sigmod, parse_sigmod_toc


def article(number, title=None, abstract='<p>First paragraph.</p><p>Second paragraph.</p>', href=None):
    href = href or 'https://dl.acm.org/doi/10.1145/%s' % number
    return ('<h3><a class="DLtitleLink" href="%s">%s</a></h3>'
            '<ul class="DLauthors"><li class="nameList">Alice Example</li><li>Bob Smith</li></ul>%s') % (
                href, title or 'Paper %s' % number,
                '<div class="DLabstract">%s</div>' % abstract if abstract is not None else '')


def issue(volume, number, articles=None, publication_year=None):
    return ('<h1>Proceedings of the ACM on Management of Data: PACMMOD: '
            'Vol. %d, No. %d. %d</h1><div id="DLcontent">%s</div>') % (
                volume, number, publication_year or volume + 2022,
                article(volume * 100 + number) if articles is None else articles)


def document(*issues):
    return '<html><body>%s</body></html>' % ''.join(issues)


def pages(year):
    base = 'https://%d.sigmod.org/' % year
    landing, tocs = OPEN_TOCS[year]
    values = {base + landing: '<html><head><title>The %d ACM SIGMOD/PODS Conference</title></head><body>%s</body></html>' % (
        year, ''.join('<a href="%s">OpenTOC</a>' % name for name in tocs))}
    if year == 2024:
        values[base + tocs[0]] = document(*(issue(*pair) for pair in sorted(CONFERENCE_ISSUES[year])))
    else:
        for name, pair in zip(tocs, [(2, 6), (3, 1), (3, 3)]):
            values[base + name] = document(issue(*pair))
    return values


class SigmodTests(unittest.TestCase):
    def collect(self, year, values):
        calls = []

        def fetch(url):
            calls.append(url)
            value = values[url]
            if isinstance(value, Exception):
                raise value
            return value

        result = collect_sigmod(year, fetch)
        self.assertEqual(result['urls'], calls)
        return result

    def test_four_issues_and_publication_year_are_distinct_from_event_year(self):
        result = self.collect(2024, pages(2024))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['records']), 4)
        self.assertEqual([r['year'] for r in result['records']], [2024] * 4)
        self.assertEqual([r['publication_year'] for r in result['records']], [2023, 2023, 2024, 2024])
        record = result['records'][0]
        self.assertEqual(record['doi'], '10.1145/103')
        self.assertEqual(record['external_id'], record['doi'])
        self.assertEqual(record['url'], 'https://doi.org/10.1145/103')
        self.assertEqual(record['authors'], ['Alice Example', 'Bob Smith'])
        self.assertEqual(record['abstract'], 'First paragraph.\n\nSecond paragraph.')
        self.assertEqual(record['section'], 'SIGMOD research; PACMMOD 1(3)')

    def test_three_2025_tocs_do_not_claim_full_coverage(self):
        result = self.collect(2025, pages(2025))
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['records']), 3)
        self.assertIn('3/4 conference issues', result['note'])
        self.assertIn('Missing official OpenTOC issues: 2(4)', result['note'])

    def test_editorials_are_excluded_but_research_about_editors_is_retained(self):
        parsed = parse_sigmod_toc(document(issue(2, 1,
            article(1, 'PACMMOD Editorial: Volume 2, Issue 1') + article(2, 'Editorial recommendation systems'))), 2024)
        self.assertEqual(parsed['excluded'], 1)
        self.assertEqual([r['doi'] for r in parsed['records']], ['10.1145/2'])

    def test_proofpoint_links_are_decoded_locally_into_exact_acm_dois(self):
        href = 'https://urldefense.proofpoint.com/v2/url?u=https-3A__dl.acm.org_doi_10.1145_3725256&amp;d=unneeded'
        parsed = parse_sigmod_toc(document(issue(3, 3, article(1, href=href))), 2025)
        self.assertEqual(parsed['records'][0]['doi'], '10.1145/3725256')
        self.assertEqual(parsed['errors'], [])

    def test_unrecognized_or_ambiguous_doi_links_cannot_establish_membership(self):
        for href in ('https://example.org/doi/10.1145/1',
                     'https://dl.acm.org/doi/10.1145/1?extra=1',
                     'https://dl.acm.org/doi/10.1145/1#fragment',
                     'https://urldefense.proofpoint.com/v2/url?u=https-3A__dl.acm.org_doi_10.1145_1&amp;u=https-3A__dl.acm.org_doi_10.1145_2',
                     'https://urldefense.proofpoint.com/v2/url?u=https-3A__example.org_doi_10.1145_1'):
            with self.subTest(href=href):
                parsed = parse_sigmod_toc(document(issue(2, 1, article(1, href=href) + article(2))), 2024)
                self.assertEqual([r['doi'] for r in parsed['records']], ['10.1145/2'])
                self.assertTrue(parsed['errors'])

    def test_pods_next_event_and_wrong_publication_year_are_rejected(self):
        for html in (document(issue(2, 2)), document(issue(2, 4)), document(issue(2, 1, publication_year=2025))):
            with self.subTest(html=html):
                with self.assertRaisesRegex(ValueError, 'outside'):
                    parse_sigmod_toc(html, 2024)

    def test_missing_abstract_does_not_borrow_next_papers_abstract(self):
        parsed = parse_sigmod_toc(document(issue(2, 1, article(1, abstract=None) + article(2))), 2024)
        self.assertIsNone(parsed['records'][0]['abstract'])
        self.assertEqual(parsed['records'][1]['abstract'], 'First paragraph.\n\nSecond paragraph.')
        self.assertTrue(parsed['errors'])

    def test_empty_author_block_and_missing_abstract_mark_result_incomplete(self):
        for body in (article(1, abstract=None), article(1).replace('<li class="nameList">Alice Example</li><li>Bob Smith</li>', '')):
            values = pages(2024)
            values['https://2024.sigmod.org/toc.html'] = document(
                *(issue(*pair, articles=body if pair == (1, 3) else None) for pair in sorted(CONFERENCE_ISSUES[2024])))
            result = self.collect(2024, values)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(len(result['records']), 4)
            self.assertIn('Missing or malformed author/abstract', result['note'])

    def test_math_and_paragraph_text_are_preserved(self):
        body = '<p>A <img alt="x+y"> formula and <script type="math/tex">z^2</script>.</p><p>More <em>text</em>.</p><script>bad()</script>'
        parsed = parse_sigmod_toc(document(issue(2, 1, article(1, abstract=body))), 2024)
        abstract = parsed['records'][0]['abstract']
        self.assertIn('x+y', abstract)
        self.assertIn('z^2', abstract)
        self.assertIn('\n\nMore text', abstract)
        self.assertNotIn('bad()', abstract)

    def test_truncated_unknown_empty_and_repeated_issue_layouts_are_rejected(self):
        for html in (document(issue(2, 1)).replace('</html>', ''),
                     document(issue(2, 1)).replace('Vol. 2', 'Volume two'),
                     document(issue(2, 1, articles='')),
                     document(issue(2, 1), issue(2, 1))):
            with self.subTest(html=html):
                with self.assertRaises(ValueError):
                    parse_sigmod_toc(html, 2024)

    def test_duplicate_doi_deduplicates_and_conflicts_are_incomplete(self):
        values = pages(2024)
        values['https://2024.sigmod.org/toc.html'] = document(
            issue(1, 3, article(1) + article(1)), issue(1, 4), issue(2, 1), issue(2, 3))
        result = self.collect(2024, values)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['records']), 4)
        values['https://2024.sigmod.org/toc.html'] = values['https://2024.sigmod.org/toc.html'].replace('Paper 203', 'Changed title').replace('10.1145/203', '10.1145/1')
        result = self.collect(2024, values)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['records']), 3)
        self.assertIn('Conflicting duplicate DOI', result['note'])

    def test_wrong_landing_page_identity_or_absent_links_prevents_fetching_tocs(self):
        landing = 'https://2024.sigmod.org/sigmod_proceedings.shtml'
        for page in ('<html><title>The 2025 ACM SIGMOD/PODS Conference</title></html>',
                     '<html><title>The 2024 ACM SIGMOD/PODS Conference</title></html>',
                     '<html><title>The 2024 ACM SIGMOD/PODS Conference</title>'):
            result = self.collect(2024, {landing: page})
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(result['records'], [])

    def test_toc_failure_retains_other_successful_issues(self):
        for error in (OSError('HTTP 503'), APIError('HTTP 503')):
            values = pages(2025)
            values['https://2025.sigmod.org/toc-2-6.html'] = error
            result = self.collect(2025, values)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(len(result['records']), 2)
            self.assertIn('HTTP 503', result['note'])

    def test_request_limit_propagates(self):
        values = pages(2025)
        values['https://2025.sigmod.org/toc-2-6.html'] = RequestLimit('stop')
        with self.assertRaises(RequestLimit):
            self.collect(2025, values)

    def test_landing_fetch_failure_propagates(self):
        with self.assertRaises(APIError):
            self.collect(2024, {'https://2024.sigmod.org/sigmod_proceedings.shtml': APIError('503')})

    def test_unsupported_year_does_not_guess_source_urls(self):
        for year in (2017, 2023, 2026):
            result = self.collect(year, {})
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(result['records'], [])
        for year in (True, '2024', 2024.0):
            with self.assertRaises(ValueError):
                self.collect(year, {})


if __name__ == '__main__':
    unittest.main()
