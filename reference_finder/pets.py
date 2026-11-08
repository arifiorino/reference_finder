"""Collect PoPETs research papers, presented at PETS, from the official archive.

The annual volume lists all four issues. Each research paper's landing page
provides its abstract and citation metadata; editorial introductions are not
research papers. A completed run means all four published issues and their
paper pages were parsed, not that another source independently verified them.
"""
from __future__ import annotations

import re
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from .api import APIError
from .storage import normalize_doi, normalize_title


_EDITORS = re.compile(r"^editors?['’]?\s+introduction[.:]?$", re.I)


def _text(node):
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _paper_url(href, base, year):
    parts = urlsplit(urljoin(base, href))
    if (parts.scheme not in ("http", "https")
            or parts.hostname not in ("petsymposium.org", "www.petsymposium.org")
            or not re.fullmatch(r"/popets/{0}/popets-{0}-\d+\.php".format(year), parts.path)):
        return None
    return urlunsplit(("https", "petsymposium.org", parts.path, "", ""))


def _meta(soup, name):
    node = soup.find("meta", attrs={"name": name})
    return str(node.get("content", "")).strip() if node else ""


def _label(content, name):
    for node in content.find_all(["b", "strong"]):
        if _text(node).casefold() == name.casefold() + ":":
            return node
    return None


def _field_value(content, name):
    label = _label(content, name)
    if not label:
        return ""
    pieces = []
    for sibling in label.next_siblings:
        if getattr(sibling, "name", None) in ("br", "b", "strong"):
            break
        pieces.append(_text(sibling) if hasattr(sibling, "get_text") else str(sibling))
    return " ".join(" ".join(pieces).split())


def _detail(html, record, issue):
    """Return trusted enrichment only when the page matches its roster entry."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.content")
    if content is None or "</html>" not in html.lower():
        return None
    title = _meta(soup, "citation_title") or _text(content.find("h2"))
    year = _meta(soup, "citation_publication_volume") or _field_value(content, "Volume")
    paper_issue = _meta(soup, "citation_publication_issue") or _field_value(content, "Issue")
    abstract_label = _label(content, "Abstract")
    if (normalize_title(title) != normalize_title(record["title"])
            or year != str(record["year"]) or paper_issue != str(issue)
            or abstract_label is None):
        return None
    paragraph = abstract_label.find_parent("p")
    if paragraph is None:
        return None
    abstract = re.sub(r"^Abstract:\s*", "", _text(paragraph), flags=re.I) or None
    doi = normalize_doi(_meta(soup, "citation_doi"))
    if not doi:
        # Artifact/dataset DOI links elsewhere on the page are not paper IDs.
        doi = normalize_doi(_field_value(content, "DOI"))
    authors = [str(node.get("content", "")).strip()
               for node in soup.select('meta[name="citation_author"]')]
    return dict(record, abstract=abstract, doi=doi,
                authors=list(dict.fromkeys(name for name in authors if name)))


def collect_pets(year: int, fetch: Callable[[str], str]) -> dict:
    """Collect one PoPETs volume without double-counting PETS presentations.

    The caller owns caching, pacing, request budgets, and retries. Failure to
    fetch the volume propagates. Failed or unrecognized paper pages retain
    their roster title and leave the run incomplete so it can be retried.
    RequestLimit deliberately propagates; cached requests make a resumed scan
    cheap. Partial annual volumes remain incomplete until all four issues exist.
    """
    if isinstance(year, bool) or not isinstance(year, int) or not 2015 <= year <= 2099:
        raise ValueError("PoPETs year must be an integer between 2015 and 2099")
    url = "https://petsymposium.org/popets/{}/".format(year)
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    result = {"records": [], "urls": [url], "status": "incomplete", "note": ""}
    content = soup.select_one("div.content.accepted-list")
    heading = content.find("h2") if content else None
    if not heading or _text(heading) != "Volume {}".format(year):
        result["note"] = "The official PoPETs volume heading or paper list was not found for {}.".format(year)
        return result

    records = {}
    issues = {}
    issue = None
    malformed = 0
    excluded = 0
    for node in content.find_all(["h3", "li"]):
        if node.name == "h3":
            match = re.fullmatch(r"Issue\s+([1-4])", _text(node))
            issue = int(match[1]) if match else None
            if issue is None or issue in issues:
                malformed += 1
            elif issue is not None:
                issues[issue] = 0
            continue
        link = node.select_one("b a[href], strong a[href]")
        title = _text(link)
        if _EDITORS.fullmatch(title):
            excluded += 1
            continue
        canonical = _paper_url(link["href"], url, year) if link else None
        if not title or not canonical or issue is None:
            malformed += 1
            continue
        if canonical in records:
            old = records[canonical]
            if normalize_title(old["title"]) != normalize_title(title) or old["section"] != "Issue {}".format(issue):
                malformed += 1
            continue
        issues[issue] += 1
        records[canonical] = {"external_id": canonical, "title": title, "year": year,
                              "doi": None, "authors": [], "url": canonical,
                              "abstract": None, "section": "Issue {}".format(issue)}

    failures = 0
    by_doi = {}
    for canonical, record in records.items():
        result["urls"].append(canonical)
        try:
            enriched = _detail(fetch(canonical), record, int(record["section"].split()[-1]))
        except (APIError, OSError):
            enriched = None
        if enriched is None:
            failures += 1
            enriched = record
        doi = enriched["doi"]
        previous = by_doi.get(doi) if doi else None
        if previous is not None:
            if normalize_title(previous["title"]) == normalize_title(enriched["title"]):
                if not previous["abstract"] and enriched["abstract"]:
                    previous["abstract"] = enriched["abstract"]
                continue
            # Do not silently merge conflicting titles using a bad DOI.
            malformed += 1
            previous["doi"] = enriched["doi"] = None
        elif doi:
            by_doi[doi] = enriched
        result["records"].append(enriched)

    missing = sorted(set(range(1, 5)) - {number for number, count in issues.items() if count})
    truncated = "</html>" not in html.lower()
    if records and not (malformed or failures or missing or truncated):
        result["status"] = "complete"
    result["note"] = ("Enumerated {} research papers from the official PoPETs volume; "
                      "excluded {} editors' introductions. Complete refers to this source's "
                      "four issues and paper pages, not an independent proceedings count.").format(
                          len(result["records"]), excluded)
    if missing:
        result["note"] += " Issues not yet present or without recognizable research papers: {}.".format(
            ", ".join(map(str, missing)))
    if malformed:
        result["note"] += " {} malformed or conflicting entries require review.".format(malformed)
    if failures:
        result["note"] += " {} paper pages failed retrieval or validation; their roster titles were retained.".format(failures)
    if truncated:
        result["note"] += " Volume HTML appears truncated."
    return result
