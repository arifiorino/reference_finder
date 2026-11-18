"""Enumerate main-conference papers from DBLP's public SPARQL API.

DBLP's HTML site currently blocks automated access.  Online collection therefore
uses its documented public API, not a browser challenge or mirror workaround.
The BeautifulSoup helpers below also accept HTML saved by a user.  "complete"
means all records in DBLP's indexed main proceedings were read and counted; it
does not certify that DBLP has indexed every accepted paper for an ongoing year.

API documentation: https://blog.dblp.org/2024/09/09/introducing-our-public-sparql-query-service/
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode, urljoin, urlsplit

from bs4 import BeautifulSoup

from .api import RequestLimit
from .storage import normalize_doi


STREAMS = {
    "ccs": "ccs", "usenix-security": "uss", "sp": "sp",
    "crypto": "crypto", "eurosp": "eurosp", "asiaccs": "asiaccs",
    "eurocrypt": "eurocrypt", "asiacrypt": "asiacrypt",
    "pets": "popets", "ndss": "ndss", "tcc": "tcc", "pkc": "pkc",
    "sigmod": "sigmod", "vldb": "pvldb", "icde": "icde",
}
SPARQL_ENDPOINT = "https://sparql.dblp.org/sparql"
PAGE_SIZE = 500
_PREFIX = "PREFIX dblp: <https://dblp.org/rdf/schema#>\n"
_REC = "https://dblp.org/rec/"
_SEPARATOR = "\x1f"

# PACMMOD includes PODS issues, and publication years straddle SIGMOD events.
# These explicit issue memberships are listed in the DBLP SIGMOD series index
# https://dblp.org/db/conf/sigmod/index.html and the official annual programs.
# For example https://2026.sigmod.org/sigmod_papers.shtml lists all four 2026
# issues, while https://2024.sigmod.org/toc.html supplies the 2024 OpenTOC.
SIGMOD_ISSUES = {
    2023: ((1, 1), (1, 2)),
    2024: ((1, 3), (1, 4), (2, 1), (2, 3)),
    2025: ((2, 4), (2, 6), (3, 1), (3, 3)),
    2026: ((3, 4), (3, 6), (4, 1), (4, 3)),
}


def _scope(venue_key, year):
    if venue_key not in STREAMS:
        raise ValueError("Unknown conference: " + str(venue_key))
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
        raise ValueError("year must be an integer between 1900 and 2100")
    stream = STREAMS[venue_key]
    if venue_key in {"pets", "vldb"} or (venue_key == "sigmod" and year >= 2023):
        # Journal-backed venues need issue/year scopes, not proceedings keys.
        return stream, None
    if venue_key == "asiaccs":
        # Older AsiaCCS proceedings and paper keys lived under conf/ccs.
        return stream, rf"^https://dblp[.]org/rec/conf/(asiaccs/{year}|ccs/{year}asia)(-[0-9]+)?$"
    # The terminal anchor excludes 2024w, 2024cset, companions and other events.
    return stream, rf"^https://dblp[.]org/rec/conf/{stream}/{year}(-[0-9]+)?$"


def _query_url(query):
    # `format=json` works without a caller-specific Accept header.
    return SPARQL_ENDPOINT + "?" + urlencode({"query": query, "format": "json"})


def _bindings(raw):
    payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    if not isinstance(payload, dict) or payload.get("exception") or payload.get("error"):
        raise ValueError("DBLP API did not return a successful result")
    rows = payload.get("results", {}).get("bindings")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("DBLP API response is missing result bindings")
    return rows


def _value(row, name, default=""):
    value = row.get(name, {})
    if not isinstance(value, dict) or not isinstance(value.get("value", default), str):
        raise ValueError("Invalid DBLP binding for " + name)
    return value.get("value", default)


def _books_query(stream, pattern):
    return _PREFIX + """SELECT ?book (SAMPLE(?bookTitle) AS ?title)
       (SAMPLE(?tocPage) AS ?toc) (COUNT(DISTINCT ?paper) AS ?count)
WHERE {
  ?book dblp:bibtexType <http://purl.org/net/nknouf/ns/bibtex#Proceedings> ;
        dblp:publishedInStream <https://dblp.org/streams/conf/%s> .
  FILTER(REGEX(STR(?book), %s))
  OPTIONAL { ?book dblp:title ?bookTitle . }
  OPTIONAL { ?book dblp:listedOnTocPage ?tocPage . }
  OPTIONAL { ?paper a dblp:Inproceedings ; dblp:publishedAsPartOf ?book . }
} GROUP BY ?book ORDER BY ?book""" % (stream, json.dumps(pattern))


def _papers_query(books, offset):
    values = " ".join("<" + book + ">" for book in books)
    return _PREFIX + """SELECT ?paper (SAMPLE(?paperTitle) AS ?title)
       (SAMPLE(?paperDoi) AS ?doi) (SAMPLE(?document) AS ?url)
       (SAMPLE(?paperYear) AS ?publicationYear)
       (GROUP_CONCAT(DISTINCT CONCAT(STR(?authorOrdinal), ":", ?authorName); separator="\\u001f") AS ?authors)
       (GROUP_CONCAT(DISTINCT STR(?book); separator="\\u001f") AS ?books)
WHERE {
  VALUES ?book { %s }
  ?paper a dblp:Inproceedings ; dblp:publishedAsPartOf ?book ;
         dblp:title ?paperTitle .
  OPTIONAL { ?paper dblp:doi ?paperDoi . }
  OPTIONAL { ?paper dblp:primaryDocumentPage ?document . }
  OPTIONAL { ?paper dblp:yearOfPublication ?paperYear . }
  OPTIONAL {
    ?paper dblp:hasSignature ?signature .
    ?signature a dblp:AuthorSignature ; dblp:signatureOrdinal ?authorOrdinal ;
               dblp:signatureDblpName ?authorName .
  }
} GROUP BY ?paper ORDER BY ?paper LIMIT %d OFFSET %d""" % (values, PAGE_SIZE, offset)


def _ordered_authors(value):
    authors = []
    for item in filter(None, value.split(_SEPARATOR)):
        ordinal, separator, name = item.partition(":")
        if not separator or not ordinal.isdigit() or not name:
            raise ValueError("DBLP returned an invalid author signature")
        authors.append((int(ordinal), name))
    return [name for ordinal, name in sorted(authors)]


def _popets_scope(year):
    return """?paper a dblp:Article ;
         dblp:publishedInStream <https://dblp.org/streams/journals/popets> ;
         dblp:publishedInJournalVolume ?volume .
  FILTER(STR(?volume) = %s)""" % json.dumps(str(year))


def _popets_query(year, offset=None):
    scope = _popets_scope(year)
    if offset is None:
        return _PREFIX + "SELECT (COUNT(DISTINCT ?paper) AS ?count) WHERE { " + scope + " }"
    return _PREFIX + """SELECT ?paper ?volume (SAMPLE(?paperTitle) AS ?title)
       (SAMPLE(?paperDoi) AS ?doi) (SAMPLE(?document) AS ?url)
       (SAMPLE(?paperYear) AS ?publicationYear)
       (GROUP_CONCAT(DISTINCT CONCAT(STR(?authorOrdinal), ":", ?authorName); separator="\\u001f") AS ?authors)
WHERE {
  %s
  ?paper dblp:title ?paperTitle .
  OPTIONAL { ?paper dblp:doi ?paperDoi . }
  OPTIONAL { ?paper dblp:primaryDocumentPage ?document . }
  OPTIONAL { ?paper dblp:yearOfPublication ?paperYear . }
  OPTIONAL {
    ?paper dblp:hasSignature ?signature .
    ?signature a dblp:AuthorSignature ; dblp:signatureOrdinal ?authorOrdinal ;
               dblp:signatureDblpName ?authorName .
  }
} GROUP BY ?paper ?volume ORDER BY ?paper LIMIT %d OFFSET %d""" % (scope, PAGE_SIZE, offset)


def _popets_editorial(title):
    return bool(re.match(r"^editors?['’]?\s+introduction\b", title.strip(), re.I))


def _collect_popets(year, fetch):
    """Count and enumerate all issues of one PoPETs volume, excluding editorials."""
    result = {"records": [], "urls": [], "status": "incomplete", "note": ""}
    observed = set()
    excluded = 0
    try:
        url = _query_url(_popets_query(year))
        result["urls"].append(url)
        counts = _bindings(fetch(url))
        if len(counts) != 1:
            raise ValueError("DBLP did not return one PoPETs volume count")
        expected = int(_value(counts[0], "count"))
        if expected < 0:
            raise ValueError("DBLP returned a negative volume count")
        if not expected:
            result.update(status="unavailable", note=(
                "No PoPETs articles are indexed in DBLP for this volume year; "
                "this is unknown coverage, not a verified zero-paper conference."
            ))
            return result
        result["urls"].append("https://dblp.org/db/journals/popets/popets%d.html" % year)
        offset = 0
        while offset < expected:
            url = _query_url(_popets_query(year, offset))
            result["urls"].append(url)
            rows = _bindings(fetch(url))
            if not rows:
                break
            for row in rows:
                uri = _value(row, "paper")
                title = _value(row, "title").strip()
                if (not uri.startswith(_REC + "journals/popets/") or not title
                        or _value(row, "volume") != str(year)):
                    raise ValueError("DBLP returned a PoPETs article outside the requested volume or without a key/title")
                if uri in observed:
                    raise ValueError("DBLP pagination repeated a paper; roster may have changed")
                observed.add(uri)
                if _popets_editorial(title):
                    excluded += 1
                    continue
                key = uri[len(_REC):]
                paper = {
                    "external_id": key, "dblp_key": key, "title": title,
                    "year": year, "abstract": None,
                    "doi": normalize_doi(_value(row, "doi")),
                    "url": _value(row, "url") or uri,
                    "authors": _ordered_authors(_value(row, "authors")),
                    "section": "PoPETs volume %d" % year,
                }
                publication_year = _value(row, "publicationYear")
                if publication_year.isdigit():
                    paper["publication_year"] = int(publication_year)
                result["records"].append(paper)
            offset += len(rows)
            if len(rows) < PAGE_SIZE:
                break
        result["note"] = "Read %d of %d DBLP-indexed PoPETs articles; %d editors' introductions excluded." % (
            len(observed), expected, excluded)
        if len(observed) == expected and result["records"]:
            result["status"] = "complete"
            result["note"] += " Complete relative to DBLP's volume; indexing itself may still be incomplete."
    except RequestLimit:
        raise
    except Exception as exc:
        result["note"] = "DBLP PoPETs collection incomplete: " + str(exc)
    return result


def _pacmmod_query(year, offset=None):
    issues = " ".join('("%s" "%s")' % pair for pair in SIGMOD_ISSUES[year])
    values = "VALUES (?volume ?issue) { " + issues + " }"
    membership = """?paper a dblp:Article ;
        dblp:publishedInStream <https://dblp.org/streams/journals/pacmmod> ;
        dblp:publishedInJournalVolume ?volume ;
        dblp:publishedInJournalVolumeIssue ?issue ."""
    if offset is None:
        return _PREFIX + "SELECT ?volume ?issue (COUNT(DISTINCT ?paper) AS ?count) WHERE { " + values + " OPTIONAL { " + membership + " } } GROUP BY ?volume ?issue ORDER BY ?volume ?issue"
    return _PREFIX + """SELECT ?paper ?volume ?issue (SAMPLE(?paperTitle) AS ?title)
        (SAMPLE(?paperDoi) AS ?doi) (SAMPLE(?document) AS ?url)
        (SAMPLE(?paperYear) AS ?publicationYear)
        (GROUP_CONCAT(DISTINCT CONCAT(STR(?authorOrdinal), ":", ?authorName); separator="\\u001f") AS ?authors)
WHERE {
  %s
  %s
  ?paper dblp:title ?paperTitle .
  OPTIONAL { ?paper dblp:doi ?paperDoi . }
  OPTIONAL { ?paper dblp:primaryDocumentPage ?document . }
  OPTIONAL { ?paper dblp:yearOfPublication ?paperYear . }
  OPTIONAL {
    ?paper dblp:hasSignature ?signature .
    ?signature a dblp:AuthorSignature ; dblp:signatureOrdinal ?authorOrdinal ;
               dblp:signatureDblpName ?authorName .
  }
} GROUP BY ?paper ?volume ?issue ORDER BY ?paper LIMIT %d OFFSET %d""" % (values, membership, PAGE_SIZE, offset)


def _pacmmod_editorial(title):
    return bool(re.match(r"^(?:PACMMOD\b.*\bEditorial\b|Editorial\s+for\s+(?:PACMMOD|SIGMOD)\b)", title.strip(), re.I))


def _collect_sigmod_journal(year, fetch):
    result = {"records": [], "urls": [], "status": "incomplete", "note": ""}
    if year not in SIGMOD_ISSUES:
        result.update(status="unavailable", note="SIGMOD's PACMMOD issue membership has not been verified for this event year; do not infer it from the publication year.")
        return result
    scope = {(str(volume), str(issue)) for volume, issue in SIGMOD_ISSUES[year]}
    observed = {pair: set() for pair in scope}
    seen, excluded = set(), 0
    try:
        url = _query_url(_pacmmod_query(year))
        result["urls"].append(url)
        expected = {}
        for row in _bindings(fetch(url)):
            pair = (_value(row, "volume"), _value(row, "issue"))
            count = int(_value(row, "count"))
            if pair not in scope or pair in expected or count < 0:
                raise ValueError("DBLP returned invalid or out-of-scope SIGMOD issue counts")
            expected[pair] = count
        if set(expected) != scope:
            raise ValueError("DBLP omitted a requested SIGMOD issue count")
        if not sum(expected.values()):
            result.update(status="unavailable", note="No articles are indexed in the verified SIGMOD PACMMOD issues; this is unknown coverage, not a verified zero-paper conference.")
            return result
        result["urls"].append("https://dblp.org/db/conf/sigmod/index.html")
        for volume in sorted({v for v, _ in scope}):
            result["urls"].append("https://dblp.org/db/journals/pacmmod/pacmmod%s.html" % volume)
        offset = 0
        while offset < sum(expected.values()):
            url = _query_url(_pacmmod_query(year, offset))
            result["urls"].append(url)
            rows = _bindings(fetch(url))
            if not rows:
                break
            for row in rows:
                uri, title = _value(row, "paper"), _value(row, "title").strip()
                pair = (_value(row, "volume"), _value(row, "issue"))
                if not uri.startswith(_REC + "journals/pacmmod/") or not title or pair not in scope:
                    raise ValueError("DBLP returned a PACMMOD record outside SIGMOD's verified issues or without a key/title")
                if uri in seen:
                    raise ValueError("DBLP pagination repeated a PACMMOD article")
                seen.add(uri)
                observed[pair].add(uri)
                if _pacmmod_editorial(title):
                    excluded += 1
                    continue
                publication_year = _value(row, "publicationYear")
                record = {"external_id": uri[len(_REC):], "dblp_key": uri[len(_REC):],
                          "title": title, "year": year, "abstract": None,
                          "volume": pair[0], "issue": pair[1],
                          "doi": normalize_doi(_value(row, "doi")), "url": _value(row, "url") or uri,
                          "authors": _ordered_authors(_value(row, "authors")),
                          "section": "SIGMOD %d: PACMMOD %s(%s)" % (year, pair[0], pair[1])}
                if publication_year.isdigit():
                    record["publication_year"] = int(publication_year)
                result["records"].append(record)
            offset += len(rows)
            if len(rows) < PAGE_SIZE:
                break
        missing = [pair for pair, count in expected.items() if count == 0 or len(observed[pair]) != count]
        result["note"] = "Read %d of %d DBLP-indexed articles in SIGMOD %d's verified PACMMOD issues; %d editorials excluded. PODS issues and companion proceedings are excluded." % (len(seen), sum(expected.values()), year, excluded)
        if not missing and result["records"]:
            result["status"] = "complete"
            result["note"] += " Complete relative to DBLP's issue indexing, which may still be incomplete."
        elif missing:
            result["note"] += " Incomplete issues: " + "; ".join("%s(%s): read %d of %d" % (v, i, len(observed[(v, i)]), expected[(v, i)]) for v, i in sorted(missing))
    except RequestLimit:
        raise
    except Exception as exc:
        result["note"] = "DBLP SIGMOD collection incomplete: " + str(exc)
    return result


def _pvldb_query(year, offset=None):
    scope = """?paper a dblp:Article ;
        dblp:publishedInStream <https://dblp.org/streams/journals/pvldb> ;
        dblp:yearOfPublication ?publicationYear .
    FILTER(STR(?publicationYear) = %s)""" % json.dumps(str(year))
    if offset is None:
        return _PREFIX + "SELECT (COUNT(DISTINCT ?paper) AS ?count) WHERE { " + scope + " }"
    return _PREFIX + """SELECT ?paper ?publicationYear (SAMPLE(?paperTitle) AS ?title)
        (SAMPLE(?paperDoi) AS ?doi) (SAMPLE(?document) AS ?url)
        (SAMPLE(?journalVolume) AS ?volume) (SAMPLE(?journalIssue) AS ?issue)
        (GROUP_CONCAT(DISTINCT CONCAT(STR(?authorOrdinal), ":", ?authorName); separator="\\u001f") AS ?authors)
WHERE {
  %s
  ?paper dblp:title ?paperTitle .
  OPTIONAL { ?paper dblp:doi ?paperDoi . }
  OPTIONAL { ?paper dblp:primaryDocumentPage ?document . }
  OPTIONAL { ?paper dblp:publishedInJournalVolume ?journalVolume . }
  OPTIONAL { ?paper dblp:publishedInJournalVolumeIssue ?journalIssue . }
  OPTIONAL {
    ?paper dblp:hasSignature ?signature .
    ?signature a dblp:AuthorSignature ; dblp:signatureOrdinal ?authorOrdinal ;
               dblp:signatureDblpName ?authorName .
  }
} GROUP BY ?paper ?publicationYear ORDER BY ?paper LIMIT %d OFFSET %d""" % (scope, PAGE_SIZE, offset)


def _pvldb_editorial(title):
    return bool(re.match(r"^(?:(?:Welcome )?Message|Letter) from the .*(?:Editor|Chair)|^(?:Editorial|Preface|Foreword|(?:Front|Back) Matter)[.:]?\s*$", title.strip(), re.I))


def _proceedings_front_matter(title):
    return bool(re.fullmatch(
        r"(?:(?:A |Welcome )?Message from the (?:General |Program |Conference )?Chairs?|"
        r"(?:Front|Back) Matter|Preface|Foreword|Editorial)[.:]?", title.strip(), re.I,
    ))


def _collect_pvldb(year, fetch):
    """Enumerate PVLDB by publication year, never guess presentation years."""
    result = {"records": [], "urls": [], "status": "incomplete", "note": ""}
    seen, excluded = set(), 0
    try:
        url = _query_url(_pvldb_query(year))
        result["urls"].append(url)
        counts = _bindings(fetch(url))
        if len(counts) != 1:
            raise ValueError("DBLP did not return one PVLDB publication-year count")
        expected = int(_value(counts[0], "count"))
        if expected < 0:
            raise ValueError("DBLP returned a negative article count")
        if expected == 0:
            result.update(status="unavailable", note="No PVLDB articles are indexed for this publication year; this is unknown coverage, not a verified zero-paper conference.")
            return result
        offset = 0
        while offset < expected:
            url = _query_url(_pvldb_query(year, offset))
            result["urls"].append(url)
            rows = _bindings(fetch(url))
            if not rows:
                break
            for row in rows:
                uri, title = _value(row, "paper"), _value(row, "title").strip()
                if (not uri.startswith(_REC + "journals/pvldb/") or not title
                        or _value(row, "publicationYear") != str(year)):
                    raise ValueError("DBLP returned a PVLDB article outside the publication year or without a key/title")
                if uri in seen:
                    raise ValueError("DBLP pagination repeated a PVLDB article")
                seen.add(uri)
                if _pvldb_editorial(title):
                    excluded += 1
                    continue
                volume, issue = _value(row, "volume"), _value(row, "issue")
                result["records"].append({
                    "external_id": uri[len(_REC):], "dblp_key": uri[len(_REC):],
                    "title": title, "year": year, "publication_year": year,
                    "volume": volume or None, "issue": issue or None,
                    "doi": normalize_doi(_value(row, "doi")), "url": _value(row, "url") or uri,
                    "authors": _ordered_authors(_value(row, "authors")), "abstract": None,
                    "section": "PVLDB publication year %d: volume %s, issue %s" % (year, volume or "unknown", issue or "unknown"),
                })
            offset += len(rows)
            if len(rows) < PAGE_SIZE:
                break
        result["note"] = "Read %d of %d DBLP-indexed PVLDB articles published in %d; %d editorials excluded. The year is the publication year, not a claim about the VLDB presentation year." % (len(seen), expected, year, excluded)
        if len(seen) == expected and result["records"]:
            result["status"] = "complete"
            result["note"] += " Complete relative to DBLP's indexing, which may still be incomplete."
    except RequestLimit:
        raise
    except Exception as exc:
        result["note"] = "DBLP PVLDB collection incomplete: " + str(exc)
    return result


def collect_dblp(venue_key, year, fetch):
    """Return records, source URLs, status and note for an event's main roster.

    ``fetch(url)`` returns response text and must raise on HTTP errors. It is
    supplied by the runner so caching, request pacing and retries are shared.
    Publication years can differ from event years (e.g. ASIACRYPT 2024/2025),
    so conference membership is determined by proceedings keys or verified
    issue sets. PVLDB is explicitly collected by publication year instead.
    """
    stream, pattern = _scope(venue_key, year)
    if venue_key == "pets":
        return _collect_popets(year, fetch)
    if venue_key == "sigmod" and year >= 2023:
        return _collect_sigmod_journal(year, fetch)
    if venue_key == "vldb":
        return _collect_pvldb(year, fetch)
    urls = []
    records = {}
    seen, excluded = set(), 0
    result = {"records": [], "urls": urls, "status": "incomplete", "note": ""}
    try:
        url = _query_url(_books_query(stream, pattern))
        urls.append(url)
        books = {}
        for row in _bindings(fetch(url)):
            book = _value(row, "book")
            if not re.fullmatch(pattern, book):
                raise ValueError("DBLP returned a proceedings key outside the requested scope")
            count = int(_value(row, "count"))
            if count < 0 or book in books:
                raise ValueError("DBLP returned invalid or duplicate proceedings counts")
            books[book] = count
            toc = _value(row, "toc")
            if toc:
                urls.append(toc + ("" if toc.endswith(".html") else ".html"))
        if not books:
            result.update(status="unavailable", note=(
                "No main proceedings are indexed in DBLP for this event year; "
                "this is unknown coverage, not a verified zero-paper conference."
            ))
            return result

        observed = {book: set() for book in books}
        expected_max = sum(books.values())
        offset = 0
        while offset < expected_max:
            url = _query_url(_papers_query(books, offset))
            urls.append(url)
            rows = _bindings(fetch(url))
            if not rows:
                break
            for row in rows:
                uri = _value(row, "paper")
                title = _value(row, "title").strip()
                prefixes = [_REC + "conf/" + stream + "/"]
                if venue_key == "asiaccs":
                    prefixes.append(_REC + "conf/ccs/")
                if not any(uri.startswith(prefix) for prefix in prefixes) or not title:
                    raise ValueError("DBLP returned a paper without a valid key/title")
                if uri in seen:
                    raise ValueError("DBLP pagination repeated a paper; roster may have changed")
                parent_books = _value(row, "books").split(_SEPARATOR)
                if not parent_books or any(book not in books for book in parent_books):
                    raise ValueError("DBLP paper is missing its main proceedings membership")
                seen.add(uri)
                for book in parent_books:
                    observed[book].add(uri)
                if _proceedings_front_matter(title):
                    excluded += 1
                    continue
                key = uri[len(_REC):]
                publication_year = _value(row, "publicationYear")
                paper = {
                    "external_id": key, "dblp_key": key, "title": title,
                    "year": year, "abstract": None,
                    "doi": normalize_doi(_value(row, "doi")),
                    "url": _value(row, "url") or uri,
                    "authors": _ordered_authors(_value(row, "authors")),
                    "proceedings": parent_books,
                }
                if publication_year.isdigit():
                    paper["publication_year"] = int(publication_year)
                records[uri] = paper
            offset += len(rows)
            if len(rows) < PAGE_SIZE:
                break
        mismatches = [book for book in books if not books[book] or len(observed[book]) != books[book]]
        if mismatches:
            result["note"] = "Proceedings counts are incomplete: " + "; ".join(
                ("%s: volume has no indexed papers; cannot verify its coverage" % book.rsplit("/", 1)[-1])
                if not books[book] else
                ("%s: read %d of %d" % (book.rsplit("/", 1)[-1], len(observed[book]), books[book]))
                for book in mismatches
            )
        else:
            result.update(status="complete", note=(
                "Read all %d DBLP-indexed main-proceedings papers across %d volume(s); "
                "counts checked against the API. DBLP indexing itself may still be incomplete."
            ) % (len(records), len(books)))
            if excluded:
                result["note"] += " %d front-matter records excluded after validating raw API counts." % excluded
            if not records:
                result["status"] = "incomplete"
                result["note"] += " No paper records remain; this is not a verified zero-paper conference."
    except RequestLimit:
        raise
    except Exception as exc:
        result["note"] = "DBLP collection incomplete: " + str(exc)
    result["records"] = list(records.values())
    return result


def _html_soup(html):
    """Reject saved challenges, error pages and visibly truncated documents."""
    if not isinstance(html, str) or not re.search(r"</html\s*>\s*$", html, re.I):
        raise ValueError("DBLP HTML is empty or truncated (missing closing html tag)")
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True).casefold() if soup.title else ""
    if soup.find(id="anubis_challenge") or any(term in title for term in (
        "not a bot", "access denied", "oh noes", "captcha", "just a moment",
    )):
        raise ValueError("DBLP returned an access challenge, not conference records")
    return soup


def parse_dblp_index(venue_key, year, html, index_url=None):
    """Find all main annual TOC links in a user-saved DBLP series index."""
    stream, pattern = _scope(venue_key, year)
    soup = _html_soup(html)
    if venue_key == "pets":
        base = index_url or "https://dblp.org/db/journals/popets/index.html"
        urls = set()
        for link in soup.select("a[href]"):
            url = urljoin(base, link["href"])
            if urlsplit(url).path == "/db/journals/popets/popets%d.html" % year:
                urls.add(url.split("#", 1)[0])
        if not soup.select("a[href*='popets']"):
            raise ValueError("Saved DBLP PoPETs index contains no volume links")
        return sorted(urls)
    if pattern is None:
        raise ValueError("Use the DBLP API journal collector: saved volume indexes cannot establish the requested SIGMOD event or PVLDB publication-year scope")
    base = index_url or "https://dblp.org/db/conf/" + stream + "/index.html"
    urls = set()
    for entry in soup.select("li.entry"):
        key = entry.get("id", "")
        if not re.fullmatch(pattern, _REC + key):
            continue
        for link in entry.select("a[href]"):
            url = urljoin(base, link["href"])
            path = urlsplit(url).path
            directory = "(?:asiaccs|ccs)" if venue_key == "asiaccs" else stream
            if re.fullmatch(rf"/db/conf/{directory}/{stream}{year}(?:-[0-9]+)?[.]html", path):
                urls.add(url.split("#", 1)[0])
    if not soup.select("li.entry"):
        raise ValueError("Saved DBLP index contains no proceedings entries")
    return sorted(urls)


def parse_dblp_html(html, year, source_url):
    """Parse every inproceedings item in a user-saved main-conference TOC.

    Posters/keynotes appearing as individual main-proceedings papers are kept;
    section headings are retained for callers that want a narrower scope.
    Raises on empty, malformed or truncated pages instead of reporting success.
    """
    soup = _html_soup(html)
    journal = bool(re.fullmatch(r"/db/journals/popets/popets[0-9]{4}[.]html", urlsplit(source_url).path))
    if journal and urlsplit(source_url).path != "/db/journals/popets/popets%d.html" % year:
        raise ValueError("Saved DBLP PoPETs page is for a different volume year")
    entries = soup.select("li.entry.article" if journal else "li.entry.inproceedings")
    if not entries:
        raise ValueError("Saved DBLP TOC contains no article records" if journal else "Saved DBLP TOC contains no inproceedings records")
    records = []
    seen = set()
    for entry in entries:
        key = entry.get("id", "")
        title = entry.select_one(".title")
        prefix = "journals/popets/" if journal else "conf/"
        if not key.startswith(prefix) or title is None or not title.get_text(" ", strip=True):
            raise ValueError("DBLP TOC contains an incomplete paper entry")
        if key in seen:
            raise ValueError("DBLP TOC contains a duplicate paper key")
        seen.add(key)
        if journal and _popets_editorial(title.get_text(" ", strip=True)):
            continue
        if not journal and _proceedings_front_matter(title.get_text(" ", strip=True)):
            continue
        links = [urljoin(source_url, a["href"]) for a in entry.select("a[href]")]
        doi = next((normalize_doi(link) for link in links if normalize_doi(link)), None)
        electronic = entry.select_one("nav.publ li.ee a[href], nav.publ .head a[href], .publ a.ee[href], a[itemprop='url'][href]")
        publisher_url = urljoin(source_url, electronic["href"]) if electronic else None
        if not publisher_url or urlsplit(publisher_url).hostname in ("dblp.org", "dblp.uni-trier.de"):
            publisher_url = "https://doi.org/" + doi if doi else _REC + key
        authors = []
        for author in entry.select('[itemprop="author"]'):
            name = author.select_one('[itemprop="name"]')
            authors.append((name or author).get_text(" ", strip=True))
        heading = entry.find_previous(["h2", "h3", "h4"])
        records.append({
            "external_id": key, "dblp_key": key, "title": title.get_text(" ", strip=True),
            "year": year, "doi": doi, "authors": authors, "url": publisher_url,
            "abstract": None, "section": heading.get_text(" ", strip=True) if heading else None,
        })
    return records
