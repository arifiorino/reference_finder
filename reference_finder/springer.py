"""Fill missing abstracts from public Springer chapter pages identified by DOI.

This is enrichment of an existing proceedings inventory, never paper discovery.
Only an explicit, identity-checked HTML abstract is stored. The caller supplies
the fetcher so request budgets, pacing, retries, and caching remain shared.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup

from .api import APIError, RequestLimit
from .storage import normalize_doi, normalize_title

LOG = logging.getLogger(__name__)


class _AccessBlocked(ValueError):
    """A publisher access page requires stopping, not trying more paper URLs."""


def _provider_block(exc):
    return isinstance(exc, _AccessBlocked) or (isinstance(exc, APIError) and bool(re.search(
        r"HTTP\s+(?:401|403|429)\b|network request failed|bot challenge|captcha|access denied",
        str(exc), re.I)))


def _text(node):
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _metadata(soup, name):
    return list(dict.fromkeys(
        str(tag.get("content", "")).strip()
        for tag in soup.find_all("meta", attrs={"name": re.compile("^" + name + "$", re.I)})
        if str(tag.get("content", "")).strip()
    ))


def _title_key(value):
    # Invisible typography is not an identity difference. Ordinary punctuation
    # and capitalization are already ignored by normalize_title.
    return normalize_title(value.replace("\u00ad", "").replace("\u200b", ""))


def _titles_match(existing, published):
    if _title_key(existing) == _title_key(published):
        return True
    # Some DBLP titles append a plaintext rendering to the same TeX expression,
    # e.g. ``\mathsf AC^0 AC 0`` versus Springer's ``$$\mathsf {AC}^0$$``.
    # Recognize only a final, simple formatted identifier, optionally raised to
    # an integer power. Preserve the exact identifier, exponent, and font
    # command; never fuzzy-match titles or remove mathematical content.
    rendered = re.fullmatch(r"(.*?)(?:\$\$(.*?)\$\$|\\\((.*?)\\\))\s*\.?", published.strip(), re.S)
    if not rendered:
        return False
    expression = (rendered.group(2) or rendered.group(3) or "").strip()
    power = re.fullmatch(r"\\(mathsf|mathrm|mathbf|mathit)\s*\{([A-Za-z][A-Za-z0-9]*)\}(?:\s*\^\s*(?:\{([0-9]+)\}|([0-9]+)))?", expression)
    if not power:
        return False
    font, name = power.group(1), power.group(2)
    exponent = power.group(3) or power.group(4)
    name_pattern = re.escape(name)
    pattern = (r"(.*?)\\" + re.escape(font)
               + r"\s*(?:\{" + name_pattern + r"\}|" + name_pattern + r")")
    if exponent is not None:
        exponent_pattern = re.escape(exponent)
        pattern += (r"\s*\^\s*(?:\{" + exponent_pattern + r"\}|" + exponent_pattern + r")"
                    + r"(?:\s+" + name_pattern + r"\s+" + exponent_pattern + r")?")
    pattern += r"\s*\.?"
    legacy = re.fullmatch(pattern, existing.strip(), re.S)
    return bool(legacy and _title_key(legacy.group(1)) == _title_key(rendered.group(1)))


def _subtitle_matches(soup, existing, published):
    # Some chapters omit their subtitle from citation_title. Springer places it
    # in a dedicated paragraph immediately after the chapter heading. Require
    # that heading to agree with the citation metadata, then compare the entire
    # reconstructed title exactly; arbitrary page paragraphs are not evidence.
    headings = soup.select('h1.c-article-title[data-test="chapter-title"]')
    if len(headings) != 1 or _title_key(_text(headings[0])) != _title_key(published):
        return False
    subtitle = headings[0].find_next_sibling()
    if (subtitle is None or subtitle.name != "p"
            or "c-chapter-category" not in subtitle.get("class", []) or not _text(subtitle)):
        return False
    return _title_key(published + " " + _text(subtitle)) == _title_key(existing)


@lru_cache(maxsize=1)
def _reviewed_title_aliases():
    path = Path(__file__).with_name("springer_title_aliases.json")
    aliases = json.loads(path.read_text(encoding="utf-8"))
    return {(entry["doi"], entry["inventory_title"], entry["publisher_title"]) for entry in aliases}


def parse_springer_abstract(html, doi, title):
    """Validate chapter identity and return its abstract and citation authors.

    An absent abstract on an otherwise valid complete chapter is represented by
    None. Malformed, truncated, or mismatched pages raise ValueError and remain
    eligible for a future retry; description metadata is never an abstract.
    """
    requested = normalize_doi(doi)
    if not requested or not requested.startswith("10.1007/"):
        raise ValueError("Unsupported Springer chapter DOI")
    if not isinstance(html, str) or not re.search(r"</html\s*>\s*$", html, re.I):
        raise ValueError("Springer HTML is empty or truncated")
    soup = BeautifulSoup(html, "html.parser")
    page_title = _text(soup.title).casefold()
    if any(marker in page_title for marker in ("access denied", "captcha", "just a moment", "not a bot")):
        raise _AccessBlocked("Springer returned an access page, not a chapter")
    dois = {normalize_doi(value) for value in _metadata(soup, "citation_doi")}
    if dois != {requested}:
        raise ValueError("Springer citation DOI is absent, ambiguous, or does not match the requested paper")
    titles = _metadata(soup, "citation_title")
    if (len(titles) != 1 or not _title_key(title)
            or not (_titles_match(title, titles[0]) or _subtitle_matches(soup, title, titles[0])
                    or (requested, title, titles[0]) in _reviewed_title_aliases())):
        raise ValueError("Springer citation title does not match the existing paper")
    authors = _metadata(soup, "citation_author")

    field = soup.select_one("#Abs1-content")
    if field is None:
        field = soup.select_one('section[data-title="Abstract"] .c-article-section__content')
    if field is None:
        if any(_text(heading).casefold().rstrip(":") == "abstract"
               for heading in soup.find_all(["h2", "h3"])):
            raise ValueError("Springer Abstract heading has an unrecognized content layout")
        return {"abstract": None, "authors": authors}
    section = field.find_parent(class_="c-article-section") or field.find_parent("section")
    heading = section.find(["h2", "h3"]) if section else None
    if _text(heading).casefold().rstrip(":") != "abstract":
        raise ValueError("Springer abstract container lacks an explicit Abstract heading")

    fragment = BeautifulSoup(str(field), "html.parser")
    # Preserve textual math alternatives rather than inserting image URLs.
    for image in fragment.select("img[alt]"):
        image.replace_with(image.get("alt", ""))
    for script in fragment.select('script[type^="math/tex"]'):
        script.replace_with(str(script.string or ""))
    for unwanted in fragment.select("script, style, button"):
        unwanted.decompose()
    paragraphs = [_text(p) for p in fragment.find_all("p")]
    abstract = "\n\n".join(p for p in paragraphs if p) if paragraphs else _text(fragment)
    if abstract.casefold().rstrip(".") in {"no abstract available", "abstract not available", "no abstract provided"}:
        abstract = ""
    return {"abstract": abstract or None, "authors": authors}


def enrich_springer(db, venues, start, end, fetch):
    """Enrich pending Springer DOIs once, committing each verified result.

    A single snapshot prevents unsupported DOIs and retryable errors from
    repeatedly occupying the first pending batch. Unsuccessful downloads are
    never marked finished. HTTP access/rate blocks stop this provider promptly.
    """
    summary = {"candidates": 0, "checked": 0, "filled": 0, "no_abstract": 0,
               "skipped_unsupported": 0, "errors": [], "status": "complete"}
    if not venues:
        return summary
    pending = db.missing_for("springer", venues, start, end, limit=-1)
    candidates = []
    for row in pending:
        doi = normalize_doi(row["doi"])
        if doi and doi.startswith("10.1007/"):
            candidates.append((row["id"], doi))
        else:
            summary["skipped_unsupported"] += 1
    summary["candidates"] = len(candidates)
    LOG.info("Springer: %d missing chapter abstracts to check", len(candidates))
    for paper_id, doi in candidates:
        current = db.conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        if current is None or current["abstract"] is not None:
            continue
        summary["checked"] += 1
        url = "https://link.springer.com/chapter/" + quote(doi, safe="/_-.()")
        try:
            parsed = parse_springer_abstract(fetch(url), doi, current["title"])
            with db.conn:
                if parsed["abstract"]:
                    # Keep the inventory's title and existing author metadata.
                    # No venue/roster membership is invented by this lookup.
                    record = {
                        "external_id": doi, "doi": doi, "title": current["title"],
                        "year": current["publication_year"], "url": url,
                        "authors": json.loads(current["authors_json"]) or parsed["authors"],
                        "abstract": parsed["abstract"],
                    }
                    db.upsert("springer", record, known_paper_id=paper_id)
                    summary["filled"] += 1
                else:
                    db.mark_enrichment("springer", doi, "no_abstract")
                    summary["no_abstract"] += 1
        except RequestLimit:
            raise
        except (APIError, ValueError, OSError) as exc:
            message = "%s: %s" % (doi, exc)
            summary["errors"].append(message)
            summary["status"] = "incomplete"
            LOG.warning("Springer enrichment incomplete: %s", message)
            if _provider_block(exc):
                break
        if summary["checked"] % 25 == 0:
            LOG.info("Springer: %d/%d checked, %d abstracts filled, %d errors",
                     summary["checked"], summary["candidates"], summary["filled"], len(summary["errors"]))
    LOG.info("Springer: %d/%d checked, %d abstracts filled, %d without abstracts, %d errors",
             summary["checked"], summary["candidates"], summary["filled"], summary["no_abstract"], len(summary["errors"]))
    return summary
