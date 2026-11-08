"""Collect NDSS main papers from official accepted lists and HTML programs.

The accepted list is the roster; the program supplies most abstracts in one
request.  Papers missing from the program are retained and their own pages are
read.  This matters for multi-cycle years, whose program can cover fewer papers
than the accepted list.  The 2017 archive predates that layout and publishes its
research roster and abstracts together in the programme.
"""
from __future__ import annotations

import re
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from .api import APIError
from .storage import normalize_doi, normalize_title

BASE = "https://www.ndss-symposium.org"
_NON_PAPER = re.compile(r"\b(?:keynotes?|posters?|workshops?|invited talks?|opening remarks|welcome|test of time)\b", re.I)


def _text(node):
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _paragraphs(node):
    if not node:
        return None
    # Leaf paragraphs avoid duplicating malformed nested <p> markup on NDSS.
    paragraphs = [_text(p) for p in node.find_all("p") if not p.find("p")]
    return "\n\n".join(p for p in paragraphs if p) or _text(node) or None


def _url(href, base):
    parts = urlsplit(urljoin(base, href))
    if parts.scheme not in {"http", "https"} or parts.hostname not in {"www.ndss-symposium.org", "ndss-symposium.org"}:
        return None
    return urlunsplit(("https", "www.ndss-symposium.org", parts.path.rstrip("/") + "/" if not parts.path.lower().endswith(".pdf") else parts.path, "", ""))


def _paper_url(href, base, year):
    url = _url(href, base)
    if not url:
        return None
    path = urlsplit(url).path
    if re.fullmatch(r"/ndss-paper/[^/]+/", path):
        return url
    if year == 2017 and re.fullmatch(r"/ndss2017/ndss-2017-programme/[^/]+/", path):
        return url
    if path.startswith("/wp-content/uploads/") and path.lower().endswith(".pdf") and str(year) in path:
        return url
    return None


def _authors(node):
    if not node:
        return []
    metadata = [tag.get("content", "").strip() for tag in node.select('meta[name="citation_author"]')]
    if any(metadata):
        return list(dict.fromkeys(x for x in metadata if x))
    value = node.get_text(" ", strip=True)
    value = re.sub(r"^\s*Authors?\s*:\s*", "", value, flags=re.I)
    # Affiliations contain commas and 'and'; remove balanced parenthesized
    # groups before splitting names. Unbalanced strings are not guessed.
    cleaned, depth = [], 0
    for character in value:
        if character == "(":
            depth += 1
        elif character == ")":
            if not depth:
                return []
            depth -= 1
        elif not depth:
            cleaned.append(character)
    if depth:
        return []
    names = re.split(r"\s*[,;]\s*|\s+and\s+|\s*&\s*", "".join(cleaned))
    names = [re.sub(r"^and\s+", "", x.strip(" .;")) for x in names]
    return list(dict.fromkeys(x for x in names if x and x.lower() != "and"))


def _doi(node):
    tag = node.select_one('meta[name="citation_doi"]') if node else None
    return normalize_doi(tag.get("content", "")) if tag else None


def _paper_pdf(node, base, year):
    for link in node.find_all("a", href=True):
        if _text(link).lower() == "paper":
            url = _paper_url(link["href"], base, year)
            if url and urlsplit(url).path.lower().endswith(".pdf"):
                return url
    return None


def _record(title, url, year, authors=None, abstract=None, doi=None, pdf_url=None):
    return {"external_id": url, "title": title.strip(), "year": year,
            "url": url, "doi": doi, "authors": authors or [], "abstract": abstract,
            "section": "Main conference papers", "_pdf_url": pdf_url}


def _valid_heading(soup, year, accepted=False):
    heading = _text(soup.find("h1"))
    return bool(re.search(r"\bNDSS\b", heading, re.I) and str(year) in heading
                and (not accepted or "accepted papers" in heading.lower()))


def _accepted(html, url, year):
    soup = BeautifulSoup(html, "html.parser")
    if not _valid_heading(soup, year, accepted=True):
        return {}, 1
    main = soup.select_one("main") or soup
    nodes = main.select(".pt-cv-content-item, .accepted-papers-section .single-paper, .paper-list .rel-paper")
    records, uncertain = {}, 0
    for node in nodes:
        heading = node.find(["h2", "h3"])
        link = heading.find("a", href=True) if heading else None
        link = link or node.select_one("a.paper-link-abs[href], .paper-link-abs a[href]")
        canonical = _paper_url(link["href"], url, year) if link else None
        title = _text(heading)
        if not canonical or not title:
            uncertain += 1
            continue
        authors = node.select_one(".pt-cv-content") or node.find("p")
        record = _record(title, canonical, year, _authors(authors), doi=_doi(node))
        if canonical in records and normalize_title(title) != normalize_title(records[canonical]["title"]):
            uncertain += 1
        else:
            records[canonical] = record
    expected = re.search(r"\b([\d,]+)\s+papers\s+(?:were\s+)?accepted\b", _text(main), re.I)
    if expected and len(records) != int(expected[1].replace(",", "")):
        uncertain += 1
    if main.select_one(".pt-cv-pagination, a[rel='next']"):
        # Do not certify a first page as a whole-year roster.
        uncertain += 1
    return records, uncertain + (not nodes) + ("</html>" not in html.lower())


def _program(html, url, year):
    soup = BeautifulSoup(html, "html.parser")
    if not _valid_heading(soup, year):
        return {}, 1
    main = soup.select_one("main") or soup
    records, uncertain = {}, 0
    # 2019 onward: one list item per talk; keynotes share the same HTML.
    for node in main.select("ul.list-group-session > li"):
        link = node.select_one('strong a[href*="/ndss-paper/"]')
        if not link:
            continue
        session = node.find_parent("ul").find_previous_sibling("a")
        section = _text(session)
        if _NON_PAPER.search(section):
            continue
        canonical = _paper_url(link["href"], url, year)
        if not canonical:
            uncertain += 1
            continue
        button = node.select_one(".abstract-btn [data-target]")
        target = str(button.get("data-target", "")) if button else ""
        abstract = node.find(id=target[1:]) if target.startswith("#") else None
        record = _record(_text(link), canonical, year, _authors(node.find("i")), _paragraphs(abstract), _doi(node),
                         _paper_pdf(node, url, year))
        records[canonical] = record
    # 2018: research entries have an inline modal-body and a paper PDF link.
    for abstract in main.select(".modal-body"):
        node = abstract.find_parent("div", class_="row")
        if not node:
            continue
        title_node = node.select_one("p > b")
        paper = next((a for a in node.find_all("a", href=True) if _text(a).lower() == "paper"), None)
        canonical = _paper_url(paper["href"], url, year) if paper else None
        if not title_node or not canonical:
            uncertain += 1
            continue
        title = _text(title_node).rstrip(".")
        authors_node = BeautifulSoup(str(title_node.parent), "html.parser")
        authors_node.find("b").decompose()
        records[canonical] = _record(title, canonical, year, _authors(authors_node), _paragraphs(abstract), _doi(node))
    # 2017: each h3 title is followed by abstract paragraph(s), then Authors.
    if year == 2017:
        for heading in main.find_all("h3"):
            link = heading.find("a", href=True)
            canonical = _paper_url(link["href"], url, year) if link else None
            if not canonical:
                continue
            section = _text(heading.find_previous("h2"))
            if _NON_PAPER.search(section):
                continue
            paragraphs, authors = [], []
            for sibling in heading.next_siblings:
                if getattr(sibling, "name", None) in {"h2", "h3"}:
                    break
                if getattr(sibling, "name", None) != "p":
                    continue
                if re.match(r"Authors?\s*:", _text(sibling), re.I):
                    # Explicit <br> lines delimit older author records.
                    for line in sibling.get_text("\n", strip=True).splitlines():
                        if not re.fullmatch(r"Authors?\s*:", line, re.I):
                            authors.extend(_authors(BeautifulSoup("<p>" + line + "</p>", "html.parser")))
                elif _text(sibling):
                    paragraphs.append(_text(sibling))
            records[canonical] = _record(_text(link), canonical, year, authors, "\n\n".join(paragraphs) or None)
    return records, uncertain + ("</html>" not in html.lower())


def _detail(html, record, year):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.ndss-paper")
    data = article.select_one(".paper-data") if article else None
    title = _text(soup.select_one("h1.entry-title"))
    if not data or not title or "</html>" not in html.lower():
        return None
    expected_title = normalize_title(record["title"])
    actual_title = normalize_title(title)
    if record["title"].endswith(("...", "…")):
        if not actual_title.startswith(expected_title):
            return None
    elif actual_title != expected_title:
        return None
    categories = [value for value in article.get("class", []) if value.startswith("category-")]
    if categories and not any(re.fullmatch(r"category-ndss-?" + str(year), value) for value in categories):
        return None
    authors_node = data.find("strong")
    authors = _authors(authors_node)
    if authors_node:
        authors_node.decompose()
    return _record(title, record["url"], year, authors, _paragraphs(data), _doi(soup),
                   _paper_pdf(article, record["url"], year))


def collect_ndss(year: int, fetch: Callable[[str], str]) -> dict:
    """Return the published main-paper roster, never workshops or posters.

    A completed run enumerates this source's currently published list, not a
    guarantee that an evolving event has published its final proceedings.
    Detail failures retain known titles and leave the run incomplete for retry.
    """
    if isinstance(year, bool) or not isinstance(year, int) or not 2017 <= year <= 2099:
        raise ValueError("NDSS year must be an integer between 2017 and 2099")
    urls, failures, uncertain = [], [], 0
    program = {}
    if year == 2017:
        url = BASE + "/ndss2017/ndss-2017-programme/"
        urls.append(url)
        records, uncertain = _program(fetch(url), url, year)
    else:
        accepted_url = BASE + "/ndss%d/accepted-papers/" % year
        urls.append(accepted_url)
        records, uncertain = _accepted(fetch(accepted_url), accepted_url, year)
        if records:
            wrapper_url = BASE + "/ndss%d/program/" % year
            try:
                urls.append(wrapper_url)
                wrapper_html = fetch(wrapper_url)
                wrapper = BeautifulSoup(wrapper_html, "html.parser")
                main = wrapper.select_one("main") or wrapper
                candidates = []
                for a in main.find_all("a", href=True):
                    candidate = _url(a["href"], wrapper_url)
                    if candidate and str(year) in candidate and re.search(r"\bsymposium\s+program(?:me)?\b", _text(a), re.I):
                        candidates.append(candidate)
                program_html, program_url = wrapper_html, wrapper_url
                if candidates:
                    program_url = list(dict.fromkeys(candidates))[0]
                    urls.append(program_url)
                    program_html = fetch(program_url)
                program, malformed = _program(program_html, program_url, year)
                uncertain += malformed
                for key, record in records.items():
                    if key in program:
                        match = program[key]
                        record.update({name: match[name] for name in ("title", "authors", "abstract", "doi", "_pdf_url") if match[name]})
            except (APIError, OSError) as exc:
                failures.append("Program retrieval failed: " + str(exc))
        for key, record in list(records.items()):
            truncated_title = record["title"].endswith(("...", "…"))
            if record["abstract"] and not truncated_title:
                continue
            # PDF-only links are valid bibliography records; never parse a PDF
            # as HTML. These can be enriched later through DOI/title APIs.
            if urlsplit(key).path.lower().endswith(".pdf"):
                continue
            try:
                urls.append(key)
                detail = _detail(fetch(key), record, year)
                if detail:
                    record.update({name: detail[name] for name in ("title", "authors", "abstract", "doi", "_pdf_url") if detail[name]})
                else:
                    uncertain += 1
            except (APIError, OSError) as exc:
                failures.append(key + ": " + str(exc))
            if record["title"].endswith(("...", "…")):
                # Truncated card text must not overwrite a complete API title.
                del records[key]
                uncertain += 1
    # NDSS2022 lists two papers twice under distinct landing-page URLs. The
    # publication PDF establishes identity even when a title has a spacing
    # typo. Prefer the program's URL, also used by the proceedings bibliography.
    duplicates, publications = 0, {}
    for key in sorted(records, key=lambda key: key not in program):
        record = records[key]
        identity = record.pop("_pdf_url", None)
        if identity and identity in publications:
            existing = publications[identity]
            for name in ("abstract", "authors", "doi"):
                if not existing[name] and record[name]:
                    existing[name] = record[name]
            del records[key]
            duplicates += 1
        elif identity:
            publications[identity] = record
    status = "complete" if records and not uncertain and not failures else "incomplete"
    note = ("Enumerated %d main research papers from the official NDSS roster; "
            "complete refers to the currently published source, not an independent final proceedings count." % len(records))
    if duplicates:
        note += " %d duplicate landing pages sharing the same publication PDF were merged." % duplicates
    if uncertain:
        note += " %d malformed, truncated, mismatched, or unverified source entries/pages require review." % uncertain
    if failures:
        note += " %d retrieval failures; resume to retry. %s" % (len(failures), "; ".join(failures[:3]))
    return {"records": list(records.values()), "urls": list(dict.fromkeys(urls)), "status": status, "note": note}
