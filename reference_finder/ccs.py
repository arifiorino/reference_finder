"""Enumerate the official CCS 2026 accepted-paper list, including both cycles.

This list fills the interval between acceptance and publication/indexing of the
proceedings.  It contains titles and authors, but no abstracts or paper links at
the time this parser was written.  Membership remains provisional.
"""
import hashlib
import re
import unicodedata
from urllib.parse import quote, urljoin, urlsplit

from bs4 import BeautifulSoup


ACCEPTED_URLS = {
    2026: "https://www.sigsac.org/ccs/CCS2026/program/accepted-papers.html",
}
_CYCLES = ("First Cycle", "Second Cycle")


def _text(node):
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _identity(year, title):
    normalized = " ".join(unicodedata.normalize("NFKC", title).casefold().split())
    return "{}:{}".format(year, hashlib.sha256(normalized.encode("utf-8")).hexdigest())


def _remove_affiliation(value):
    """Remove the final parenthesized affiliation, preserving nested brackets."""
    value = " ".join(value.split())
    if value.endswith(")"):
        depth = 0
        for index in range(len(value) - 1, -1, -1):
            if value[index] == ")":
                depth += 1
            elif value[index] == "(":
                depth -= 1
                if depth == 0:
                    # Names and affiliations are separated by whitespace in
                    # the official table.  Do not strip unrelated name text.
                    if index and value[index - 1].isspace():
                        return value[:index].strip()
                    break
    return value


def _authors(cell):
    fragment = BeautifulSoup(str(cell), "html.parser")
    for br in fragment.find_all("br"):
        br.replace_with("\n")
    # Preserve inline formatting within a name; only the source's line breaks
    # separate authors.  Commas within affiliations do not delimit names.
    names = [_remove_affiliation(line) for line in fragment.get_text().splitlines() if line.strip()]
    return list(dict.fromkeys(name for name in names if name))


def _paper_url(row, title_cell, source_url):
    for anchor in title_cell.select("a[href]"):
        url = urljoin(source_url, anchor["href"])
        parts = urlsplit(url)
        if parts.scheme in ("http", "https") and parts.hostname:
            return url
    identifier = row.get("id") or title_cell.get("id")
    return source_url + "#" + quote(identifier, safe="-_.~") if identifier else source_url


def collect_ccs(year, fetch):
    """Return the official accepted-paper roster for a supported event year.

    ``fetch(url)`` owns HTTP errors, caching, and pacing.  Only the observed 2026
    URL is supported; other years are explicitly unavailable, with no guessed
    URLs. ``complete`` means every row of both published acceptance cycles was
    parsed, not that a final proceedings roster has been independently verified.
    """
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError("CCS year must be an integer")
    url = ACCEPTED_URLS.get(year)
    result = {"records": [], "urls": [url] if url else [], "status": "unavailable", "note": ""}
    if not url:
        result["note"] = "No official accepted-paper parser is configured for CCS {}; use its indexed proceedings.".format(year)
        return result
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    result["status"] = "incomplete"
    if not re.search(r"\bCCS\s*" + str(year) + r"\b", _text(soup.title), re.I):
        result["note"] = "The accepted-paper page title does not identify CCS {}.".format(year)
        return result
    heading = soup.find(["h1", "h2"], string=re.compile(r"^\s*accepted papers\s*$", re.I))
    if heading is None:
        result["note"] = "The official accepted-papers heading was not found."
        return result

    records = {}
    counts = {}
    malformed = 0
    for table in soup.select("table.accepted-papers-table"):
        cycle_heading = table.find_previous(["h1", "h2", "h3"])
        cycle = _text(cycle_heading)
        if cycle not in _CYCLES or cycle in counts:
            malformed += 1
            continue
        counts[cycle] = 0
        rows = table.find_all("tr")
        header_seen = False
        for row_index, row in enumerate(rows):
            cells = row.find_all(["td", "th"], recursive=False)
            if cells and all(cell.name == "th" for cell in cells):
                valid_header = (row_index == 0 and len(cells) == 2 and
                                [_text(cell).casefold() for cell in cells] in
                                (["title", "author"], ["title", "authors"]))
                if not valid_header:
                    malformed += 1
                header_seen = valid_header
                continue
            if len(cells) != 2 or any(cell.name != "td" for cell in cells):
                malformed += 1
                continue
            title = _text(cells[0])
            authors = _authors(cells[1])
            if not title or not authors:
                malformed += 1
                continue
            external_id = _identity(year, title)
            if external_id in records:
                # A duplicate row across acceptance cycles may signal a page
                # editing error. Keep one record and surface the disagreement.
                malformed += 1
                continue
            records[external_id] = {
                "external_id": external_id, "title": title, "year": year,
                "doi": None, "authors": authors,
                "url": _paper_url(row, cells[0], url), "abstract": None,
                "section": cycle,
            }
            counts[cycle] += 1
        if not header_seen:
            malformed += 1

    result["records"] = list(records.values())
    missing_cycles = [cycle for cycle in _CYCLES if not counts.get(cycle)]
    truncated = "</html>" not in html.lower()
    if not malformed and not missing_cycles and not truncated:
        result["status"] = "complete"
    result["note"] = (
        "Parsed {} official accepted papers ({}). Provisional acceptance list; "
        "complete refers to the source's two published cycles, not final proceedings. "
        "This source lists no abstracts."
    ).format(len(records), "; ".join("{}: {}".format(cycle, counts.get(cycle, 0)) for cycle in _CYCLES))
    if malformed:
        result["note"] += " {} malformed, duplicate, or unrecognized rows/tables require review.".format(malformed)
    if missing_cycles:
        result["note"] += " Missing or empty cycles: " + ", ".join(missing_cycles) + "."
    if truncated:
        result["note"] += " HTML document appears truncated."
    return result
