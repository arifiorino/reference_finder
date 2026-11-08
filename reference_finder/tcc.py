"""Read TCC 2026's official accepted-paper data, including its abstracts.

The conference's acceptedpapers.php page explicitly loads json/papers.json.
Only that acceptedPapers array is a roster; program sessions and invited talks
are not papers. This is a narrow fallback for the 2026 conference, whose DBLP
proceedings may not yet be available, not a claim about other years' layouts.
"""
import json
import re
from typing import Callable

from bs4 import BeautifulSoup


ACCEPTED_URL = "https://tcc.iacr.org/2026/acceptedpapers.php"
PAPERS_URL = "https://tcc.iacr.org/2026/json/papers.json"


def _text(value):
    return " ".join(value.split()) if isinstance(value, str) else ""


def _positive_id(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"[1-9]\d*", value):
        return value
    return None


def collect_tcc(year: int, fetch: Callable[[str], str]) -> dict:
    """Collect the published 2026 accepted roster using the caller's fetcher.

    Fetch errors and request-budget limits propagate for safe retry. Invalid
    JSON/layout and malformed entries never establish a completed checkpoint.
    The published public CryptoDB key provides a stable URL that can also match
    future IACR records; a year-scoped submission ID is used if it is absent.
    """
    if isinstance(year, bool) or not isinstance(year, int) or year != 2026:
        raise ValueError("The official TCC accepted-paper collector supports only 2026")
    result = {"records": [], "urls": [ACCEPTED_URL], "status": "incomplete", "note": ""}
    html = fetch(ACCEPTED_URL)
    soup = BeautifulSoup(html, "html.parser")
    headings = {_text(node.get_text(" ", strip=True)) for node in soup.find_all("h1")}
    accepted = any(_text(node.get_text(" ", strip=True)) == "Accepted Papers"
                   for node in soup.find_all("h2"))
    # Validate the conference/year and explicit data link before consuming JSON.
    linked = any(re.search(r"\bfetch\s*\(\s*(['\"])json/papers\.json\1", node.get_text())
                 for node in soup.find_all("script") if not node.get("src"))
    if "TCC 2026" not in headings or not accepted or not linked or "</html>" not in html.lower():
        result["note"] = "Official TCC 2026 accepted-paper page identity or linked data source was not verified."
        return result

    result["urls"].append(PAPERS_URL)
    try:
        payload = json.loads(fetch(PAPERS_URL))
    except (ValueError, TypeError):
        result["note"] = "Official TCC accepted-paper data is not valid JSON; retry after the source is repaired."
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("acceptedPapers"), list):
        result["note"] = "The official JSON has no recognizable acceptedPapers array."
        return result

    entries = payload["acceptedPapers"]
    if not entries:
        result["status"] = "unavailable"
        result["note"] = "The official TCC 2026 accepted-paper array is empty; no roster is available yet."
        return result
    paper_ids, pubkeys = set(), set()
    malformed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        identifier = _positive_id(entry.get("paperId"))
        title = _text(entry.get("title"))
        pubkey = _positive_id(entry.get("pubkey"))
        if (not identifier or not title or identifier in paper_ids
                or (pubkey and pubkey in pubkeys)):
            malformed += 1
            continue
        paper_ids.add(identifier)
        if pubkey:
            pubkeys.add(pubkey)
        elif entry.get("pubkey") not in (None, ""):
            malformed += 1

        authors = entry.get("authors", [])
        if not isinstance(authors, list) or any(not _text(author) for author in authors):
            malformed += 1
            authors = []
        authors = list(dict.fromkeys(_text(author) for author in authors))
        abstract = entry.get("abstract")
        if abstract is not None and not isinstance(abstract, str):
            malformed += 1
            abstract = None
        # JSON strings are plain text, including mathematical < and > signs.
        # Parsing them as HTML would discard legitimate scientific notation.
        abstract = _text(abstract) or None
        url = ("https://www.iacr.org/cryptodb/data/paper.php?pubkey=" + pubkey
               if pubkey else ACCEPTED_URL)
        result["records"].append({
            "external_id": pubkey or "tcc:2026:paper:" + identifier,
            "title": title, "year": year, "doi": None, "authors": authors,
            "url": url, "abstract": abstract, "section": "Accepted main conference papers",
        })

    count = len(result["records"])
    abstracts = sum(bool(record["abstract"]) for record in result["records"])
    if count and not malformed:
        result["status"] = "complete"
    result["note"] = (
        "Parsed {}/{} accepted research papers and {} abstracts from the official TCC 2026 "
        "accepted-paper JSON. Complete refers to this published accepted roster; the program's "
        "invited talks and sessions are not included, and future proceedings may differ. "
        "No DOIs are supplied by this source."
    ).format(count, len(entries), abstracts)
    if malformed:
        result["note"] += " {} malformed or duplicate entries/fields require review.".format(malformed)
    return result
