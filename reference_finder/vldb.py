"""Validate PVLDB articles and retrieve abstracts from the official volumes.

PVLDB volumes span two calendar years, and presentation rollovers mean a volume
is not the roster of one VLDB conference. This collector deliberately uses the
DBLP article publication year, then joins those articles to the publisher's
embedded public volume data. It never assigns an event year from a volume.
"""
from __future__ import annotations

from collections import defaultdict
import json
import re
import unicodedata
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

from .api import APIError, RequestLimit
from .dblp import collect_dblp
from .storage import normalize_title


_NONPAPER = re.compile(r"^(?:front\s*matter|back\s*matter|editorial|preface|table of contents|letter from\b.*)[.!:]?$", re.I)
_HOSTS = {"vldb.org", "www.vldb.org"}


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _pdf_key(value):
    """Normalize only the publisher's documented current/legacy paper paths."""
    if not isinstance(value, str):
        return None
    parts = urlsplit(value.strip())
    path = unquote(parts.path)
    if parts.scheme or parts.netloc:
        if parts.scheme not in {"http", "https"} or parts.hostname not in _HOSTS:
            return None
        path = re.sub(r"^/(?:pvldb/|pvldb_wp/archive/)", "", path)
    else:
        path = path.lstrip("/")
    match = re.fullmatch(r"vol([1-9][0-9]*)/(p[0-9]+-[^/]+?)(?:\.pdf)?", path)
    return (int(match[1]), match[2]) if match else None


def _authors(value):
    if isinstance(value, list):
        if any(not _text(author) for author in value):
            raise ValueError("Invalid official author list")
        return [_text(author) for author in value]
    if not isinstance(value, str) or not value.strip():
        return []
    # Publisher exports sometimes attach affiliations containing commas. Strip
    # balanced parentheses before splitting, rather than inventing extra authors.
    while re.search(r"\([^()]*\)", value):
        value = re.sub(r"\([^()]*\)", "", value)
    return [name.strip().rstrip("*").strip() for name in re.split(r",|;|\s+and\s+", value)
            if name.strip().rstrip("*").strip()]


def _author_key(value):
    # DBLP appends numeric homonym identifiers. They are not part of a name.
    value = re.sub(r"\s+[0-9]{4}$", "", value)
    value = unicodedata.normalize("NFKD", value).casefold()
    return "".join(ch for ch in value if ch.isalnum() and not unicodedata.combining(ch))


def _same_authors(left, right):
    return bool(left and right and [_author_key(a) for a in left] == [_author_key(a) for a in right])


def parse_volume(html, volume):
    """Read the public Next.js data embedded in a complete official volume page.

    The issue list establishes paper membership. Summary rows are enrichment
    only and cannot create papers. Both PDF identity and title/author agreement
    are required before transferring an abstract from a summary row.
    """
    if not isinstance(html, str) or not re.search(r"</html\s*>\s*$", html, re.I):
        raise ValueError("PVLDB volume HTML is empty or truncated")
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.select('script#__NEXT_DATA__[type="application/json"]')
    if len(scripts) != 1:
        raise ValueError("PVLDB volume has no unambiguous embedded public data")
    payload = json.loads(scripts[0].string or "")
    page = payload.get("props", {}).get("pageProps", {})
    if (payload.get("page") != "/volumes/[volume]" or payload.get("query", {}).get("volume") != str(volume)
            or page.get("volume") != str(volume)):
        raise ValueError("PVLDB volume identity does not match the requested volume")
    issues = page.get("groupedIssues")
    summaries = page.get("volumeSummaries")
    if not isinstance(issues, dict) or not issues or not isinstance(summaries, list):
        raise ValueError("PVLDB issue list or abstract summaries are missing")
    papers, excluded = {}, 0
    for issue, entries in issues.items():
        if not re.fullmatch(r"[1-9][0-9]*", str(issue)) or not isinstance(entries, list) or not entries:
            raise ValueError("PVLDB contains an invalid or empty issue")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("PVLDB issue contains a malformed entry")
            title = _text(entry.get("title"))
            if _NONPAPER.fullmatch(title):
                excluded += 1
                continue
            key = _pdf_key(entry.get("pdf"))
            if not title or not key or key[0] != volume or str(entry.get("issue")) != str(issue):
                raise ValueError("PVLDB paper lacks a valid title, PDF, or matching issue")
            if key in papers:
                raise ValueError("PVLDB repeats a paper PDF in its issue list")
            papers[key] = {"title": title, "authors": _authors(entry.get("authors")),
                           "issue": str(issue), "pdf": "https://www.vldb.org/pvldb/vol%d/%s.pdf" % key,
                           "abstract": None}
    if not papers:
        raise ValueError("PVLDB volume has no recognizable article entries")
    titles = defaultdict(list)
    for key, paper in papers.items():
        titles[normalize_title(paper["title"])].append((key, paper))
    seen = set()
    for summary in summaries:
        if not isinstance(summary, dict):
            raise ValueError("PVLDB contains a malformed abstract summary")
        if _NONPAPER.fullmatch(_text(summary.get("Paper Title"))):
            continue
        summary_id = summary.get("Paper ID")
        key = _pdf_key(summary_id)
        if key is None and isinstance(summary_id, str) and re.fullmatch(r"p[0-9]+-[^/]+", summary_id):
            key = _pdf_key("vol%d/%s" % (volume, summary_id))
        # Volumes 14–15 used internal numeric submission IDs. Their summaries
        # require a unique title AND full author-list match to an issue entry.
        if key is None and (summary_id is None or re.fullmatch(r"[0-9]+", str(summary_id))):
            candidates = [(k, p) for k, p in titles.get(normalize_title(_text(summary.get("Paper Title"))), [])
                          if _same_authors(_authors(summary.get("Author Names")), p["authors"])]
            if len(candidates) == 1:
                key = candidates[0][0]
        # Summary catalogs can include stale entries; they do not define the
        # authoritative issue roster and must never create extra membership.
        if key not in papers:
            continue
        if key in seen:
            raise ValueError("PVLDB repeats an abstract summary for one PDF")
        seen.add(key)
        abstract = summary.get("Abstract")
        if abstract is None or abstract == "":
            continue
        if not isinstance(abstract, str):
            raise ValueError("PVLDB abstract is not text")
        paper = papers[key]
        if (normalize_title(_text(summary.get("Paper Title"))) != normalize_title(paper["title"])
                and not _same_authors(_authors(summary.get("Author Names")), paper["authors"])):
            raise ValueError("PVLDB abstract summary does not match its paper identity")
        paper["abstract"] = abstract.strip() or None
    return {"papers": papers, "excluded": excluded}


def collect_vldb(year, fetch):
    """Collect publisher-verified PVLDB articles using DBLP publication years.

    This supplements the DBLP roster rather than pretending the publisher's
    volume end-year is a conference year. Missing or conflicting official
    entries remain DBLP-only and make this source scan incomplete/retryable.
    """
    if isinstance(year, bool) or not isinstance(year, int) or not 2017 <= year <= 2099:
        raise ValueError("PVLDB publication year must be an integer from 2017 to 2099")
    indexed = collect_dblp("vldb", year, fetch)
    result = {"records": [], "urls": list(indexed.get("urls") or []),
              "status": "incomplete", "note": "", "year_basis": "publication"}
    if not indexed.get("records"):
        result["status"] = indexed.get("status", "incomplete")
        result["note"] = "PVLDB publication-year inventory unavailable: " + indexed.get("note", "")
        return result
    volumes = defaultdict(list)
    errors, matched = [], set()
    for record in indexed["records"]:
        volume = str(record.get("volume") or "")
        if not volume.isdigit() or int(volume) < 1 or record.get("publication_year") != year:
            errors.append("DBLP article has missing volume or mismatched publication year: " + record["external_id"])
            continue
        volumes[int(volume)].append(record)
    for volume, records in sorted(volumes.items()):
        url = "https://www.vldb.org/pvldb/volumes/%d" % volume
        result["urls"].append(url)
        try:
            official = parse_volume(fetch(url), volume)
        except RequestLimit:
            raise
        except (APIError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append("volume %d: %s" % (volume, exc))
            continue
        title_map = defaultdict(list)
        for key, paper in official["papers"].items():
            title_map[normalize_title(paper["title"])].append((key, paper))
        for record in records:
            key = _pdf_key(record.get("url"))
            paper = official["papers"].get(key)
            # Exact PDF plus either title or full ordered author agreement is
            # enough to handle a publisher spelling correction. If DBLP only
            # provides a DOI URL, require both unique title and all authors.
            if paper is not None:
                if (normalize_title(record["title"]) != normalize_title(paper["title"])
                        and not _same_authors(record.get("authors", []), paper["authors"])):
                    paper = None
            elif key is None:
                candidates = [(k, p) for k, p in title_map.get(normalize_title(record["title"]), [])
                              if _same_authors(record.get("authors", []), p["authors"])]
                if len(candidates) == 1:
                    key, paper = candidates[0]
            if paper is None or key in matched:
                errors.append("unmatched or repeated official article: " + record["external_id"])
                continue
            matched.add(key)
            result["records"].append(dict(record, external_id=paper["pdf"], dblp_key=record["external_id"],
                url=paper["pdf"], abstract=paper["abstract"],
                section="PVLDB publication year %d; volume %d, issue %s" % (year, volume, paper["issue"])))
    abstracts = sum(bool(record["abstract"]) for record in result["records"])
    result["note"] = (
        "Verified %d/%d DBLP-indexed PVLDB articles for publication year %d against official volume lists; "
        "%d have official HTML abstracts. Years are publication years, not VLDB presentation years: "
        "volumes span calendar years and papers may roll over to a later conference. Front matter and "
        "editorials are excluded. This scan depends on DBLP's year metadata and does not establish "
        "that every newly published article has been indexed."
    ) % (len(result["records"]), len(indexed["records"]), year, abstracts)
    if not errors and indexed.get("status") == "complete" and result["records"]:
        result["status"] = "complete"
    if errors:
        result["note"] += " Unresolved source entries: " + "; ".join(errors)
    return result
