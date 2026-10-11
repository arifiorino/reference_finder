"""Enumerate USENIX Security research papers from its official schedule.

USENIX puts paper titles, author affiliations, and complete abstracts in the
technical-sessions HTML, including the collapsed schedule entries.  One page per
year is sufficient; a presentation's video or slides are not evidence of a paper.
"""

import re
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from .storage import normalize_doi


_AUTHOR_FIELD = ".field-name-field-paper-people-text"
_ABSTRACT_FIELD = ".field-name-field-paper-description-long"
_NON_PAPER_SECTION = re.compile(
    r"\b(?:enigma|keynote|invited|poster|posters|panel|test of time|lightning talks)\b", re.I)


def _text(element):
    return " ".join(element.get_text(" ", strip=True).split()) if element else ""


def _canonical_url(href, base, year):
    parts = urlsplit(urljoin(base, href))
    prefix = "/conference/usenixsecurity{:02d}/".format(year % 100)
    if parts.hostname not in ("usenix.org", "www.usenix.org"):
        return None
    # Most years use /presentation/<slug>, while at least one 2026 research
    # paper has a valid /usenixsecurity26/<slug> URL.  The node's publication
    # fields establish paper status; requiring a presentation path loses it.
    if not parts.path.startswith(prefix) or not parts.path[len(prefix):].strip("/"):
        return None
    return urlunsplit(("https", "www.usenix.org", parts.path.rstrip("/"), "", ""))


def _authors(node):
    """Use explicit names, or remove italic affiliations before splitting names."""
    metadata = [tag.get("content", "").strip()
                for tag in node.select('meta[name="citation_author"]')]
    if any(metadata):
        return list(dict.fromkeys(name for name in metadata if name))
    field = node.select_one(_AUTHOR_FIELD)
    if not field:
        return []
    # The real schedule uses <em> for affiliations, with semicolons often
    # *inside* those elements.  Replace them with delimiters rather than simply
    # deleting them, which would concatenate adjacent groups of authors.
    fragment = BeautifulSoup(str(field), "html.parser")
    affiliations = fragment.select("em, i")
    if not affiliations:
        # Without explicit affiliation markup the string is ambiguous.  Do not
        # turn universities or departments into invented author names.
        return []
    for affiliation in affiliations:
        affiliation.replace_with(";")
    names = re.split(r"\s*[,;]\s*|\s+and\s+|\s*&\s*", _text(fragment))
    names = [re.sub(r"^and\s+", "", name.strip()) for name in names]
    return list(dict.fromkeys(name for name in names if name and name.lower() != "and"))


def _abstract(node):
    field = node.select_one(_ABSTRACT_FIELD)
    if not field:
        return None
    # Preserve paragraph boundaries while keeping inline emphasis/math within
    # its paragraph.  This excludes biographies elsewhere in the same node.
    paragraphs = [_text(p) for p in field.find_all("p", recursive=True)]
    value = "\n\n".join(p for p in paragraphs if p) if paragraphs else _text(field)
    return value or None


def _doi(node):
    # Abstracts can link to a dataset, software artifact, or another paper.
    # Only explicit citation metadata identifies the DOI of this publication.
    for tag in node.select('meta[name="citation_doi"]'):
        value = normalize_doi(tag.get("content", ""))
        if value:
            return value
    return None


def collect_usenix(year: int, fetch: Callable[[str], str]) -> dict:
    """Return one year's official research-paper roster and available abstracts.

    ``fetch`` owns network retries, caching, politeness, and HTTP errors.  Errors
    propagate so a failed download cannot be confused with an empty roster.
    ``complete`` means all recognizable research entries on this particular
    source page were read; it is not an independent guarantee about publication
    completeness or about an evolving current-year conference.
    """
    if not isinstance(year, int) or not 2000 <= year <= 2099:
        raise ValueError("USENIX year must be an integer between 2000 and 2099")
    url = "https://www.usenix.org/conference/usenixsecurity{:02d}/technical-sessions".format(year % 100)
    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select(".node-paper")
    if not nodes:
        heading = _text(soup.select_one("h1"))
        unpublished = bool(re.search(r"(?:not found|access denied|coming soon|not yet available)",
                                     heading + " " + _text(soup.select_one("#content")), re.I))
        return {"records": [], "urls": [url],
                "status": "unavailable" if unpublished else "incomplete",
                "note": "No recognizable research-paper entries on the official schedule; "
                        "the roster may be unpublished, inaccessible, or the HTML layout may have changed."}

    records = {}
    excluded = 0
    uncertain = 0
    for node in nodes:
        session = node.find_parent(class_="node-session")
        section = _text(session.find("h2", recursive=False)) if session else ""
        authors_field = node.select_one(_AUTHOR_FIELD)
        has_citation = bool(node.select_one('meta[name="citation_title"]'))
        paper_media = bool(node.select_one(".usenix-schedule-media.pdf, .field-name-field-paper-pdf"))
        # Talks share node-paper with research, but use presented-by, not the
        # paper author field.  The PDF icon is corroborating publication evidence.
        if _NON_PAPER_SECTION.search(section) or (node.select_one(".field-name-field-presented-by")
                                                and not authors_field and not has_citation):
            excluded += 1
            continue
        if not (authors_field or has_citation or paper_media):
            uncertain += 1
            continue
        link = node.select_one("h2 a[href]")
        title = _text(link)
        canonical = _canonical_url(link["href"], url, year) if link else None
        if not title or not canonical:
            uncertain += 1
            continue
        record = {"external_id": canonical, "title": title, "year": year,
                  "doi": _doi(node), "authors": _authors(node), "url": canonical,
                  "abstract": _abstract(node), "section": section}
        old = records.get(canonical)
        if old:
            if not old["abstract"] and record["abstract"]:
                old["abstract"] = record["abstract"]
            if old["title"] != record["title"]:
                uncertain += 1
        else:
            records[canonical] = record

    # A missing document tail indicates a truncated download.  Do not certify
    # only the prefix as a completed roster even when some records parsed.
    truncated = "</html>" not in html.lower()
    status = "complete" if records and not uncertain and not truncated else "incomplete"
    note = ("Enumerated {} research papers from the official technical-sessions page; "
            "excluded {} invited/Enigma/panel/poster/test-of-time/lightning-talk entries. "
            "Complete refers to this source page, not an independently verified final proceedings count.").format(
                len(records), excluded)
    if uncertain:
        note += " {} entries could not be classified or parsed and require review.".format(uncertain)
    if truncated:
        note += " HTML document appears truncated."
    return {"records": list(records.values()), "urls": [url], "status": status, "note": note}
