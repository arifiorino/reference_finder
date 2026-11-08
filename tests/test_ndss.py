import unittest

from reference_finder.api import RequestLimit
from reference_finder.ndss import BASE, collect_ndss
from reference_finder.roster import RosterStore
from reference_finder.storage import Database
from reference_finder.webfetch import FetchError


def page(body):
    return '<html><body><main>' + body + '</main></body></html>'


def accepted_item(slug='example', title='An example paper', year=2026):
    return '''<div class="pt-cv-content-item"><h2 class="pt-cv-title">
      <a href="{base}/ndss-paper/{slug}/">{title}</a></h2>
      <div class="pt-cv-content"><p>Alice Example (University A), Bob Author (Department, University B)</p></div></div>'''.format(base=BASE,slug=slug,title=title)


def accepted(body=None, year=2026):
    return page('<h1>NDSS Symposium %d Accepted Papers</h1>' % year + (accepted_item() if body is None else body))


def program_item(slug='example', title='An example paper', abstract='<p>First <em>important</em> paragraph.</p><p>Second paragraph.</p>'):
    return '''<li class="list-group-item"><div class="row"><div class="col-10">
      <strong><a href="/ndss-paper/{slug}/">{title}</a></strong>
      <div><i><p>Alice Example (University A), Bob Author (Department, University B)</p></i></div>
      <div class="abstract-btn"><a data-target="#pid-{slug}">Abstract</a></div></div></div>
      <div id="pid-{slug}" class="collapse">{abstract}</div></li>'''.format(slug=slug,title=title,abstract=abstract)


def program(body=None, year=2026, section='Session 1: Network Security'):
    return page('<h1>NDSS Symposium %d Program</h1><a class="card-subheading-session">%s</a><ul class="list-group-session">' % (year,section) + (program_item() if body is None else body) + '</ul>')


def detail(title='An example paper', year=2026, abstract='<p>The full abstract.</p>', extra=''):
    return page('''<h1 class="entry-title">{title}</h1><article class="ndss-paper category-ndss-{year}">
      <div class="paper-data"><p><strong><p>Alice Example (University A), Bob Author (Department, University B)</p></strong></p>
      <p>{abstract}</p></div></article>{extra}'''.format(title=title,year=year,abstract=abstract,extra=extra))


class NdssTests(unittest.TestCase):
    def run_collector(self, year=2026, roster=None, schedule=None, details=None, wrapper=None):
        accepted_url = BASE + '/ndss%d/accepted-papers/' % year
        wrapper_url = BASE + '/ndss%d/program/' % year
        program_url = BASE + '/ndss-program/symposium-%d/' % year
        pages = {accepted_url: accepted(year=year) if roster is None else roster,
                 wrapper_url: page('<h1>NDSS %d and Co-located Event Programs</h1><a href="%s">Read More Symposium Program</a><a href="/ndss-program/workshop-%d/">Workshop Program</a>' % (year,program_url,year)) if wrapper is None else wrapper,
                 program_url: program(year=year) if schedule is None else schedule}
        pages.update(details or {})
        calls=[]
        def fetch(url):
            calls.append(url)
            value = pages[url]
            if isinstance(value, Exception):
                raise value
            return value
        result=collect_ndss(year,fetch)
        self.assertEqual(result['urls'], list(dict.fromkeys(calls)))
        return result,calls

    def test_modern_program_supplies_inline_abstract_and_authors(self):
        result,calls = self.run_collector()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(calls),3)
        r,=result['records']
        self.assertEqual(r['title'],'An example paper')
        self.assertEqual(r['abstract'],'First important paragraph.\n\nSecond paragraph.')
        self.assertEqual(r['authors'],['Alice Example','Bob Author'])
        self.assertEqual(r['external_id'],BASE+'/ndss-paper/example/')
        self.assertIsNone(r['doi'])

    def test_accepted_paper_absent_from_program_is_retrieved(self):
        roster=accepted(accepted_item()+accepted_item('fall','Fall paper'))
        result,calls=self.run_collector(roster=roster,details={BASE+'/ndss-paper/fall/':detail('Fall paper')})
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(result['records']),2)
        self.assertEqual(result['records'][1]['abstract'],'The full abstract.')
        self.assertEqual(len(calls),4)

    def test_2019_truncated_card_repaired_from_program(self):
        roster=page('''<h1>NDSS Symposium 2019 Accepted Papers</h1><div class="paper-list"><div class="rel-paper">
          <h3>An example...</h3><p>Alice (University...)</p><a class="paper-link-abs" href="/ndss-paper/example/">More Details</a></div></div>''')
        result,calls=self.run_collector(year=2019,roster=roster)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['records'][0]['title'],'An example paper')
        self.assertEqual(len(calls),3)

    def test_2018_pdf_roster_links_join_to_modal_abstract(self):
        pdf='/wp-content/uploads/2018/02/ndss2018_01A-1_paper.pdf'
        roster=page('''<h1>NDSS Symposium 2018 Accepted Papers</h1><div class="accepted-papers-section"><div class="single-paper">
          <h3>An example paper</h3><p>Alice Example and Bob Author</p><p class="paper-link-abs"><a href="%s">Read More</a></p></div></div>''' % pdf)
        schedule=page('''<h1>NDSS Symposium 2018 Programme</h1><div class="row"><div class="col-lg-8"><p><b>An example paper.</b>
          Alice Example <em>(University A)</em>, and Bob Author <em>(University B)</em>.</p>
          <div class="modal-body">An inline abstract.</div></div><div class="col-lg-4"><a href="%s">Paper</a></div></div>''' % pdf)
        result,calls=self.run_collector(year=2018,roster=roster,schedule=schedule)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['records'][0]['abstract'],'An inline abstract.')
        self.assertEqual(result['records'][0]['authors'],['Alice Example','Bob Author'])
        self.assertFalse(any(url.endswith('.pdf') for url in calls))

    def test_2017_legacy_program_abstract_and_author_lines(self):
        url=BASE+'/ndss2017/ndss-2017-programme/'
        html=page('''<h1>NDSS 2017 Programme</h1><h2>Session 1: Privacy</h2>
          <h3><a href="/ndss2017/ndss-2017-programme/example/">An example paper</a></h3>
          <p>First paragraph.</p><p>Second paragraph.</p><p>Authors:<br>Alice Example (University A)<br>Bob Author (University B)</p>
          <h3>Session Chair: Someone Else</h3><p>Not an abstract.</p>''')
        calls=[]
        def fetch(u):
            calls.append(u)
            return html
        result=collect_ndss(2017,fetch)
        self.assertEqual(calls,[url])
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['records'][0]['abstract'],'First paragraph.\n\nSecond paragraph.')
        self.assertEqual(result['records'][0]['authors'],['Alice Example','Bob Author'])

    def test_program_keynote_not_added_to_accepted_roster(self):
        schedule=program(program_item()+program_item('talk','Keynote on Security'))
        result,_=self.run_collector(schedule=schedule)
        self.assertEqual(len(result['records']),1)
        self.assertEqual(result['status'],'complete')

    def test_research_title_words_are_not_a_publication_type_filter(self):
        title='Automated Posters in Social Networks'
        result,_=self.run_collector(roster=accepted(accepted_item(title=title)),schedule=program(program_item(title=title)))
        self.assertEqual(result['status'],'complete')
        self.assertIsNotNone(result['records'][0]['abstract'])

    def test_unexpected_external_paper_link_makes_roster_incomplete(self):
        roster=accepted(accepted_item().replace(BASE+'/ndss-paper/', 'https://other.example/ndss-paper/'))
        result,calls=self.run_collector(roster=roster)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['records'],[])
        self.assertEqual(len(calls),1)

    def test_accepted_declared_count_mismatch_is_incomplete(self):
        result,_=self.run_collector(roster=accepted('<p>2 papers were accepted.</p>'+accepted_item()))
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(len(result['records']),1)

    def test_pagination_is_not_claimed_complete(self):
        result,_=self.run_collector(roster=accepted(accepted_item()+'<a rel="next" href="/page/2/">Next</a>'))
        self.assertEqual(result['status'],'incomplete')

    def test_wrong_year_heading_rejected(self):
        result,calls=self.run_collector(roster=accepted(year=2025))
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['records'],[])
        self.assertEqual(len(calls),1)

    def test_truncated_roster_preserves_records_but_is_incomplete(self):
        result,_=self.run_collector(roster=accepted().replace('</html>',''))
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(len(result['records']),1)

    def test_detail_failure_keeps_known_title_and_retries(self):
        result,_=self.run_collector(schedule=program(''),details={BASE+'/ndss-paper/example/':FetchError('HTTP 503')})
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['records'][0]['title'],'An example paper')
        self.assertIsNone(result['records'][0]['abstract'])
        self.assertIn('retrieval failures',result['note'])

    def test_program_failure_recovers_details_but_stays_retryable(self):
        result,_=self.run_collector(wrapper=FetchError('HTTP 503'),details={BASE+'/ndss-paper/example/':detail()})
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['records'][0]['abstract'],'The full abstract.')

    def test_unresolved_truncated_title_not_saved(self):
        result,_=self.run_collector(roster=accepted(accepted_item(title='An example...')),schedule=program(''),
                                    details={BASE+'/ndss-paper/example/':FetchError('HTTP 503')})
        self.assertEqual(result['records'],[])
        self.assertEqual(result['status'],'incomplete')

    def test_detail_wrong_year_or_title_rejected(self):
        for html in [detail(year=2025),detail(title='A different paper')]:
            with self.subTest(html=html):
                result,_=self.run_collector(schedule=program(''),details={BASE+'/ndss-paper/example/':html})
                self.assertEqual(result['status'],'incomplete')
                self.assertIsNone(result['records'][0]['abstract'])

    def test_related_paper_text_is_not_part_of_detail_abstract(self):
        html=detail(extra='<section class="rel-paper-wrap"><h2>View More Papers</h2><p>Unrelated paper text.</p></section>')
        result,_=self.run_collector(schedule=program(''),details={BASE+'/ndss-paper/example/':html})
        self.assertEqual(result['records'][0]['abstract'],'The full abstract.')
        self.assertEqual(result['records'][0]['authors'],['Alice Example','Bob Author'])

    def test_older_detail_category_omits_hyphen_before_year(self):
        html=detail(year=2022).replace('category-ndss-2022','category-ndss2022')
        result,_=self.run_collector(year=2022,schedule=program('',year=2022),details={BASE+'/ndss-paper/example/':html})
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['records'][0]['abstract'],'The full abstract.')

    def test_abstract_artifact_doi_is_not_assigned_to_paper(self):
        html=detail(abstract='<p>Our artifact <a href="https://doi.org/10.5281/zenodo.123">data</a>.</p>')
        result,_=self.run_collector(schedule=program(''),details={BASE+'/ndss-paper/example/':html})
        self.assertIsNone(result['records'][0]['doi'])

    def test_explicit_citation_doi_is_kept(self):
        html=detail(extra='<meta name="citation_doi" content="https://doi.org/10.14722/NDSS.2026.123">')
        result,_=self.run_collector(schedule=program(''),details={BASE+'/ndss-paper/example/':html})
        self.assertEqual(result['records'][0]['doi'],'10.14722/ndss.2026.123')

    def test_duplicate_roster_urls_are_deduplicated(self):
        result,_=self.run_collector(roster=accepted(accepted_item()+accepted_item()))
        self.assertEqual(len(result['records']),1)
        self.assertEqual(result['status'],'complete')

    def test_duplicate_landing_pages_share_pdf_and_prefer_program_identity(self):
        pdf=BASE+'/wp-content/uploads/2026-123-paper.pdf'
        roster=accepted(accepted_item('new-slug','Deep Learning Network') + accepted_item('example','Deep LearningNetwork'))
        schedule=program(program_item(title='Deep LearningNetwork').replace('</li>','<a href="'+pdf+'">Paper</a></li>'))
        html=detail(title='Deep Learning Network').replace('</article>','<div class="paper-buttons"><a href="'+pdf+'">Paper</a></div></article>')
        result,_=self.run_collector(roster=roster,schedule=schedule,details={BASE+'/ndss-paper/new-slug/':html})
        self.assertEqual(result['status'],'complete')
        self.assertEqual(len(result['records']),1)
        self.assertEqual(result['records'][0]['external_id'],BASE+'/ndss-paper/example/')
        self.assertNotIn('_pdf_url',result['records'][0])
        self.assertIn('1 duplicate landing pages',result['note'])

    def test_budget_exhaustion_propagates(self):
        with self.assertRaises(RequestLimit):
            self.run_collector(schedule=program(''),details={BASE+'/ndss-paper/example/':RequestLimit('budget')})

    def test_missing_or_future_roster_is_never_complete(self):
        result,calls=self.run_collector(year=2029,roster=page('<h1>NDSS 2029 Accepted Papers</h1><p>Coming soon.</p>'))
        self.assertEqual(result['records'],[])
        self.assertNotEqual(result['status'],'complete')
        self.assertEqual(len(calls),1)

    def test_roster_fetch_error_propagates(self):
        with self.assertRaises(FetchError):
            self.run_collector(roster=FetchError('HTTP 404'))

    def test_invalid_year(self):
        for year in (2016,2100,True,'2026'):
            with self.subTest(year=year), self.assertRaises(ValueError):
                collect_ndss(year,lambda _: '')

    def test_reviewed_2017_title_alias_preserves_dblp_paper_id(self):
        db=Database(':memory:')
        self.addCleanup(db.conn.close)
        store=RosterStore(db)
        url=BASE+'/ndss2017/ndss-2017-programme/stack-object-protection-low-fat-pointers/'
        record={'external_id':'conf/ndss/DuckYC17','title':'Stack Bounds Protection with Low Fat Pointers.',
                'year':2017,'url':url,'authors':['Gregory J. Duck','Roland H. C. Yap','Lorenzo Cavallaro']}
        store.save('dblp','ndss',2017,{'records':[record],'status':'complete'})
        paper_id=db.conn.execute('SELECT id FROM papers').fetchone()[0]
        official=dict(record,external_id=url,title='Stack Object Protection with Low Fat Pointers',abstract='The official abstract.')
        store.save('ndss','ndss',2017,{'records':[official],'status':'complete'})
        self.assertEqual(db.conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0],1)
        self.assertEqual(db.conn.execute('SELECT id,abstract FROM papers').fetchone()[:],(paper_id,'The official abstract.'))

    def test_reviewed_dblp_keynotes_do_not_become_research_papers(self):
        db=Database(':memory:')
        self.addCleanup(db.conn.close)
        store=RosterStore(db)
        for year,key,title in [(2019,'conf/ndss/Frincke19','Keynote: Modern Challenges for Cyber Defense.'),
                               (2020,'conf/ndss/Forney20','Overcoming the Evil Twins Attack: Lessons Learned from the Industrial Battlefield.')]:
            record={'external_id':key,'title':title,'year':year}
            adjusted=store.save('dblp','ndss',year,{'records':[record],'status':'complete'})
            self.assertEqual(adjusted['records'],[])
        self.assertEqual(db.conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0],0)

    def test_reviewed_2021_reprises_retain_original_2020_membership(self):
        db=Database(':memory:')
        self.addCleanup(db.conn.close)
        store=RosterStore(db)
        slugs=['phantomcache-obfuscating-cache-conflicts-with-localized-randomization',
               'soda-a-generic-online-detection-framework-for-smart-contracts',
               'uiscope-accurate-instrumentation-free-and-visible-attack-investigation-for-gui-applications']
        for slug in slugs:
            record={'external_id':BASE+'/ndss-paper/'+slug+'/','title':slug,'year':2020,'abstract':'Original abstract.'}
            store.save('ndss','ndss',2020,{'records':[record],'status':'incomplete'})
            adjusted=store.save('ndss','ndss',2021,{'records':[dict(record,year=2021)],'status':'incomplete'})
            self.assertEqual(adjusted['records'],[])
        self.assertEqual(db.conn.execute('SELECT COUNT(*) FROM papers').fetchone()[0],3)
        self.assertEqual(db.conn.execute('SELECT COUNT(*) FROM roster_entries WHERE year=2020').fetchone()[0],3)
        self.assertEqual(db.conn.execute('SELECT COUNT(*) FROM roster_entries WHERE year=2021').fetchone()[0],0)
        self.assertEqual({r[0] for r in db.conn.execute('SELECT publication_year FROM papers')},{2020})


if __name__=='__main__':
    unittest.main()
