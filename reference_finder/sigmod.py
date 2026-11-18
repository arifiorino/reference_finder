"""Retrieve SIGMOD research records and abstracts from official ACM OpenTOCs.

Research papers moved to PACMMOD in 2023. Its calendar volumes also contain
PODS papers and papers for the following SIGMOD, so membership is determined by
reviewed volume/issue pairs, not a journal name or publication year alone.
DBLP supplies the full conference inventory; these official OpenTOCs supplement
it with DOI-linked abstracts. Only the public 2024/2025 layouts below are known.
"""
from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

from .api import APIError
from .storage import normalize_doi, normalize_title


# The event's issue membership is confirmed by the official conference pages
# and https://dblp.org/db/conf/sigmod/index.html. Publication year = volume+2022.
CONFERENCE_ISSUES = {
    2024: {(1, 3), (1, 4), (2, 1), (2, 3)},
    2025: {(2, 4), (2, 6), (3, 1), (3, 3)},
}
OPEN_TOCS = {
    2024: ("sigmod_proceedings.shtml", ("toc.html",)),
    2025: ("sigmod_papers.shtml", ("toc-2-6.html", "toc-3-1.html", "toc-3-3.html")),
}
_HEADING = re.compile(
    r"^Proceedings of the ACM on Management of Data:\s*(?:PACMMOD|SIGMOD):\s*"
    r"Vol\.\s*(\d+),\s*No\.\s*(\d+)\.\s*(\d{4})$", re.I)
_EDITORIAL = re.compile(r"^PACMMOD\b.*\beditorial\b", re.I)


def _text(node):
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _doi_link(href):
    """Read ACM DOI targets, including the literal links in mailed OpenTOCs.

    Some official 2025 exports retain Proofpoint's URL encoding. Decode that
    string locally and validate the target; never fetch the redirect service.
    """
    parts = urlsplit(href)
    if parts.hostname == "urldefense.proofpoint.com" and parts.path == "/v2/url":
        values = parse_qs(parts.query).get("u", [])
        if len(values) != 1:
            return None
        decoded = values[0].replace("_", "/")
        decoded = re.sub(r"-([0-9a-fA-F]{2})", r"%\1", decoded)
        parts = urlsplit(unquote(decoded))
    if parts.scheme not in {"https", "http"}:
        return None
    if parts.hostname == "dl.acm.org":
        match = re.fullmatch(r"/doi/(10\.1145/\d+)", parts.path)
    elif parts.hostname == "doi.org":
        match = re.fullmatch(r"/(10\.1145/\d+)", parts.path)
    else:
        return None
    return normalize_doi(match[1]) if match and not (parts.query or parts.fragment) else None


def _abstract(node):
    fragment = BeautifulSoup(str(node), "html.parser")
    for image in fragment.select("img[alt]"):
        image.replace_with(image["alt"])
    for script in fragment.select('script[type^="math/tex"]'):
        script.replace_with(str(script.string or ""))
    for unwanted in fragment.select("script, style, button"):
        unwanted.decompose()
    paragraphs = [_text(p) for p in fragment.find_all("p")]
    return "\n\n".join(p for p in paragraphs if p) if paragraphs else _text(fragment)


def parse_sigmod_toc(html, year):
    """Parse DOI-bearing research articles from one official OpenTOC export.

    The 2024 file combines four issue blocks. Paper title, authors, and
    abstract must be consecutive siblings inside each DLcontent block, so an
    absent abstract cannot accidentally borrow the following paper's text.
    """
    if year not in CONFERENCE_ISSUES:
        raise ValueError("No reviewed SIGMOD OpenTOC issue mapping for this year")
    soup = BeautifulSoup(html, "html.parser")
    contents = soup.select("div#DLcontent")
    if not contents or not re.search(r"</html>\s*$", html, re.I):
        raise ValueError("Official SIGMOD OpenTOC is missing or truncated")
    records, issues, excluded, errors = [], set(), 0, []
    for content in contents:
        heading = _HEADING.fullmatch(_text(content.find_previous("h1")))
        if not heading:
            raise ValueError("Unrecognized PACMMOD volume/issue heading")
        volume, issue, publication_year = map(int, heading.groups())
        pair = (volume, issue)
        if pair not in CONFERENCE_ISSUES[year] or publication_year != volume + 2022:
            raise ValueError("PACMMOD issue is outside the requested SIGMOD conference")
        if pair in issues:
            raise ValueError("The OpenTOC repeats a PACMMOD issue")
        issues.add(pair)
        headings = content.find_all("h3", recursive=False)
        if not headings:
            raise ValueError("PACMMOD issue contains no recognizable article headings")
        for node in headings:
            link = node.select_one("a.DLtitleLink[href]")
            title = _text(link)
            doi = _doi_link(link["href"]) if link else None
            if not title or not doi:
                errors.append("A paper title or exact ACM DOI could not be read")
                continue
            if _EDITORIAL.search(title):
                excluded += 1
                continue
            authors_node = node.find_next_sibling()
            abstract_node = authors_node.find_next_sibling() if authors_node else None
            authors_ok = (authors_node is not None and authors_node.name == "ul"
                          and "DLauthors" in authors_node.get("class", []))
            abstract_ok = (abstract_node is not None and abstract_node.name == "div"
                           and "DLabstract" in abstract_node.get("class", []))
            authors = [_text(item) for item in authors_node.find_all("li", recursive=False)] if authors_ok else []
            abstract = _abstract(abstract_node) if abstract_ok else None
            if not authors or any(not name for name in authors) or not abstract:
                errors.append("Missing or malformed author/abstract block for " + doi)
            records.append({"external_id": doi, "title": title, "doi": doi,
                            "year": year, "publication_year": publication_year,
                            "authors": [name for name in authors if name],
                            "url": "https://doi.org/" + doi, "abstract": abstract or None,
                            "section": "SIGMOD research; PACMMOD %d(%d)" % pair})
    return {"records": records, "issues": issues, "excluded": excluded, "errors": errors}


def collect_sigmod(year, fetch):
    """Supplement an event roster from linked official research OpenTOCs.

    Completed source coverage requires every reviewed conference issue. In
    2025 the official site links only three of four issues; that scan remains
    incomplete, while retaining its useful records and abstracts. Request
    limits propagate, and ordinary source failures stay retryable.
    """
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError("SIGMOD year must be an integer")
    result = {"records": [], "urls": [], "status": "unavailable", "note": ""}
    if year not in OPEN_TOCS:
        result["note"] = "No official OpenTOC parser is configured for SIGMOD %d; use its DBLP inventory and API abstracts." % year
        return result
    base = "https://%d.sigmod.org/" % year
    landing_name, toc_names = OPEN_TOCS[year]
    landing = urljoin(base, landing_name)
    result["urls"].append(landing)
    html = fetch(landing)
    soup = BeautifulSoup(html, "html.parser")
    result["status"] = "incomplete"
    if (not re.search(r"\b%d\s+ACM SIGMOD/PODS\b" % year, _text(soup.title), re.I)
            or "</html>" not in html.lower()):
        result["note"] = "The official conference page identity was not verified."
        return result
    linked = {urljoin(landing, a["href"]) for a in soup.select("a[href]")}
    records, issues, excluded, errors = {}, set(), 0, []
    for name in toc_names:
        url = urljoin(base, name)
        if url not in linked:
            errors.append("Expected OpenTOC is not linked by the conference: " + name)
            continue
        result["urls"].append(url)
        try:
            parsed = parse_sigmod_toc(fetch(url), year)
        except (APIError, OSError, ValueError) as exc:
            errors.append(name + ": " + str(exc))
            continue
        issues.update(parsed["issues"])
        excluded += parsed["excluded"]
        errors.extend(parsed["errors"])
        for record in parsed["records"]:
            old = records.get(record["doi"])
            if old is not None:
                if normalize_title(old["title"]) != normalize_title(record["title"]) or old["section"] != record["section"]:
                    errors.append("Conflicting duplicate DOI " + record["doi"])
                continue
            records[record["doi"]] = record
    result["records"] = list(records.values())
    missing = sorted(CONFERENCE_ISSUES[year] - issues)
    if records and not (missing or errors):
        result["status"] = "complete"
    result["note"] = ("Read %d DOI-bearing SIGMOD research papers from %d/%d conference issues in official OpenTOCs; "
                      "%d editorials excluded. DBLP supplies the full issue-scoped inventory. "
                      "Publication years can precede the conference year; PODS issues and companion proceedings are excluded.") % (
                          len(records), len(issues), len(CONFERENCE_ISSUES[year]), excluded)
    if missing:
        result["note"] += " Missing official OpenTOC issues: " + ", ".join("%d(%d)" % pair for pair in missing) + "."
    if errors:
        result["note"] += " Problems: " + "; ".join(errors)
    return result
