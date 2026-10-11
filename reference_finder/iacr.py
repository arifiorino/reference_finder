"""Read IACR's published conference rosters and their inline abstracts.

CryptoDB's venue index supplies both event links and expected entry counts.  Its
conference pages contain the abstracts inline, when available, so collecting a
roster never requires a request per paper.  ``complete`` means the parsed roster
agrees with that index, not that every paper has an abstract or that a current
year's proceedings can no longer change.
"""
import re
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from bs4 import BeautifulSoup


INDEX_URL = "https://www.iacr.org/cryptodb/data/byconf.php"
IACR_VENUES = frozenset(("crypto", "eurocrypt", "asiacrypt"))


def _text(node) -> str:
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _iacr_link(href: str, page: str) -> Optional[str]:
    """Accept only links to the expected page on CryptoDB itself."""
    url = urljoin(INDEX_URL, href)
    parsed = urlparse(url)
    if (parsed.scheme in ("http", "https")
            and parsed.hostname in ("iacr.org", "www.iacr.org")
            and parsed.path == "/cryptodb/data/" + page):
        return url
    return None


def _doi(node) -> Optional[str]:
    for link in node.select("a[href]"):
        parsed = urlparse(link.get("href", ""))
        if parsed.hostname in ("doi.org", "dx.doi.org"):
            value = unquote(parsed.path.lstrip("/")).strip()
            if re.match(r"^10\.\d{4,9}/\S+$", value, flags=re.I):
                return value.lower()
    return None


def _invited(node) -> bool:
    # Abstracts may mention invited talks.  Only use CryptoDB's explicit award
    # label, not a substring search through the title/abstract or navigation.
    for label in node.select(".award, [title]"):
        if re.search(r"\binvited\s+(?:talk|paper)\b",
                     str(label.get("title", "")) + " " + _text(label), re.I):
            return True
    return False


def collect_iacr(venue_key: str, year: int, fetch: Callable[[str], str]) -> dict:
    """Collect one main conference roster using the caller's cached fetcher.

    Network errors propagate so the caller can record a failed/incomplete job
    and resume.  Missing event links return ``unavailable``, while layout errors
    and count disagreements return ``incomplete`` with all recoverable records.
    """
    if venue_key not in IACR_VENUES:
        raise ValueError("Unsupported IACR venue: " + venue_key)
    result = {"records": [], "urls": [INDEX_URL], "status": "incomplete", "note": ""}
    index = BeautifulSoup(fetch(INDEX_URL), "html.parser")
    table = index.select_one("table#data-table")
    if table is None:
        result["note"] = "CryptoDB venue index table not found; roster not verified."
        return result

    event_url = None
    expected = None
    valid_index_rows = 0
    for row in table.select("tbody tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 3:
            continue
        link = cells[0].select_one("a[href]")
        url = _iacr_link(link["href"], "conf.php") if link else None
        if not url:
            continue
        query = parse_qs(urlparse(url).query)
        row_venue = query.get("venue", [""])[0].lower()
        row_year = query.get("year", [""])[0]
        count_text = _text(cells[2]).replace(",", "")
        if not row_year.isdigit() or not count_text.isdigit():
            continue
        valid_index_rows += 1
        if row_venue == venue_key and int(row_year) == year:
            if _text(cells[0]).lower() != venue_key or _text(cells[1]) != str(year):
                result["note"] = "CryptoDB index label disagrees with its event link."
                return result
            event_url, expected = url, int(count_text)
            break

    if event_url is None:
        if not valid_index_rows:
            result["note"] = "CryptoDB venue index has no recognizable event rows."
        else:
            result["status"] = "unavailable"
            result["note"] = "No published CryptoDB roster is listed for %s %d." % (venue_key.upper(), year)
        return result

    result["urls"].append(event_url)
    soup = BeautifulSoup(fetch(event_url), "html.parser")
    heading = soup.find(["h1", "h2", "h3"], string=re.compile(
        r"^\s*Papers from\s+" + re.escape(venue_key) + r"\s+" + str(year) + r"\s*$", re.I))
    if heading is None:
        result["note"] = "CryptoDB event heading not found or names a different conference/year."
        return result

    entries = soup.select(".pub-title a[href]")
    identifiers = set()
    malformed = 0
    invited = 0
    volumes = 0
    for link in entries:
        url = _iacr_link(link["href"], "paper.php")
        identifier = parse_qs(urlparse(url).query).get("pubkey", [""])[0] if url else ""
        title = _text(link)
        row = link.find_parent("div", class_="row")
        if not url or not identifier.isdigit() or not title or row is None:
            malformed += 1
            continue
        if identifier in identifiers:
            malformed += 1
            continue
        identifiers.add(identifier)
        if re.match(r"^advances in cryptology\b.*\bproceedings\b", title, re.I):
            volumes += 1
            continue
        if _invited(row):
            invited += 1
            continue
        abstract = _text(row.select_one(".abstract")) or None
        result["records"].append({
            "external_id": identifier,
            "title": title,
            "year": year,
            "doi": _doi(row),
            "authors": [_text(author) for author in row.select(".author") if _text(author)],
            "url": url,
            "abstract": abstract,
            "section": "Main conference papers",
        })

    raw_count = len(identifiers)
    details = "%d main papers; %d invited entries and %d proceedings volumes excluded; %d/%d CryptoDB entries parsed" % (
        len(result["records"]), invited, volumes, raw_count, expected)
    if malformed or raw_count != expected:
        result["note"] = details + "; roster not verified (%d malformed/duplicate entries)." % malformed
    elif not result["records"]:
        result["status"] = "unavailable"
        result["note"] = details + "; no published main-paper roster."
    else:
        result["status"] = "complete"
        result["note"] = details + "; complete relative to the published CryptoDB roster."
    return result
