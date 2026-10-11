"""Conference catalog and conservative matching of provider venue labels.

Provider venue names are metadata, not paper titles. Never use ``matches`` to
classify a paper from a keyword in its title or abstract. In particular, the
generic Springer LNCS series is not evidence of a cryptography conference.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold().replace("&", " and ")
    value = "".join(c for c in value if not unicodedata.combining(c))
    # API records include both "CCS '24" and "USENIXSecurity24" spellings.
    value = re.sub(r"(?<=[a-z])(?=\d)", " ", value)
    value = re.sub(r"\b\d+(?:st|nd|rd|th)\b", " ", value)
    value = re.sub(r"\b\d{2,4}\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


_EXCLUDED = re.compile(
    r"\b(?:workshops?|companions?|posters?|doctoral|tutorials?|"
    r"spw|eurospw)\b|\bextended abstracts\b|\bsummer school\b|"
    r"\b(?:euro ?)?s and pw\b"
)


def is_excluded(name: str) -> bool:
    """Whether a venue label identifies an excluded associated event."""
    return isinstance(name, str) and bool(_EXCLUDED.search(_normalize(name)))

# Full conference names may appear inside a proceedings title followed by a
# place, dates, or a volume number. Short acronyms are matched only as labels.
_FULL_NAMES = {
    "ccs": (
        r"\bacm(?: sigsac)? conference on computer and communications security\b",
    ),
    "usenix-security": (r"\busenix security symposium\b",),
    "sp": (r"\bieee symposium on security and privacy\b",),
    "eurosp": (r"\b(?:ieee )?european symposium on security and privacy\b",),
    "asiaccs": (
        r"\bacm asia conference on computer and communications security\b",
        r"\bacm(?: asia)? symposium on information computer and communications security\b",
    ),
    "crypto": (
        r"\binternational cryptology conference\b",
        r"\badvances in cryptology crypto\b",
    ),
    "eurocrypt": (
        r"\binternational conference on the theory and applications? of cryptographic techniques\b",
        r"\badvances in cryptology eurocrypt\b",
    ),
    "asiacrypt": (
        r"\binternational conference on the theory and applications? of cryptology and information security\b",
        r"\badvances in cryptology asiacrypt\b",
    ),
}

_SHORT_NAMES = {
    "ccs": {"ccs", "acm ccs", "acm sigsac ccs", "computer and communications security", "conference on computer and communications security"},
    "usenix-security": {"usenix security", "usenixsecurity"},
    "sp": {"sp", "s and p", "ieee sp", "ieee s and p", "ieee ssp", "symposium on security and privacy"},
    "eurosp": {"eurosp", "euro sp", "euro s and p", "euros and p", "ieee eurosp", "ieee euro sp", "ieee euro s and p", "ieee euros and p"},
    "asiaccs": {"asiaccs", "asia ccs", "acm asiaccs", "acm asia ccs"},
    "crypto": {"crypto", "iacr crypto"},
    "eurocrypt": {"eurocrypt", "iacr eurocrypt"},
    "asiacrypt": {"asiacrypt", "iacr asiacrypt"},
}


@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    openalex_queries: tuple[str, ...]
    semantic_scholar_names: tuple[str, ...]

    def matches(self, name: str) -> bool:
        """Recognize a main-conference venue label, rejecting adjacent events."""
        if not isinstance(name, str) or not name.strip():
            return False
        normalized = _normalize(name)
        if _EXCLUDED.search(normalized):
            return False
        # Protect the main events when providers put both acronyms in a label.
        if self.key == "ccs" and re.search(r"\b(?:asia|asiaccs)\b", normalized):
            return False
        if self.key == "sp" and re.search(r"\b(?:european|euro\w*|asia\w*)\b", normalized):
            return False
        if self.key == "crypto" and re.search(r"\b(?:eurocrypt|asiacrypt)\b", normalized):
            return False
        if any(re.search(pattern, normalized) for pattern in _FULL_NAMES[self.key]):
            return True
        label = re.sub(r"^(?:proceedings(?: of)?(?: the)? |the )", "", normalized)
        label = re.sub(r" (?:proceedings|part [ivx]+)$", "", label)
        return label in _SHORT_NAMES[self.key]


VENUES = {
    "ccs": Venue(
        "ccs", "ACM Conference on Computer and Communications Security (CCS)",
        ("ACM Conference on Computer and Communications Security", "ACM SIGSAC Conference on Computer and Communications Security"),
        ("Conference on Computer and Communications Security", "ACM Conference on Computer and Communications Security", "ACM SIGSAC Conference on Computer and Communications Security", "CCS"),
    ),
    "usenix-security": Venue(
        "usenix-security", "USENIX Security Symposium",
        ("USENIX Security Symposium", "USENIX Security"),
        ("USENIX Security Symposium", "USENIX Security"),
    ),
    "sp": Venue(
        "sp", "IEEE Symposium on Security and Privacy (S&P)",
        ("IEEE Symposium on Security and Privacy",),
        ("IEEE Symposium on Security and Privacy", "Symposium on Security and Privacy", "SP"),
    ),
    "crypto": Venue(
        "crypto", "CRYPTO (International Cryptology Conference)",
        ("International Cryptology Conference", "Advances in Cryptology CRYPTO"),
        ("Annual International Cryptology Conference", "International Cryptology Conference", "CRYPTO", "Advances in Cryptology - CRYPTO"),
    ),
    "eurosp": Venue(
        "eurosp", "IEEE European Symposium on Security and Privacy (EuroS&P)",
        ("IEEE European Symposium on Security and Privacy",),
        ("IEEE European Symposium on Security and Privacy", "European Symposium on Security and Privacy", "EuroSP"),
    ),
    "asiaccs": Venue(
        "asiaccs", "ACM Asia Conference on Computer and Communications Security (AsiaCCS)",
        ("ACM Asia Conference on Computer and Communications Security", "ACM Symposium on Information Computer and Communications Security"),
        ("ACM Asia Conference on Computer and Communications Security", "ACM Symposium on Information Computer and Communications Security", "AsiaCCS"),
    ),
    "eurocrypt": Venue(
        "eurocrypt", "EUROCRYPT",
        ("Theory and Applications of Cryptographic Techniques", "Advances in Cryptology EUROCRYPT"),
        ("International Conference on the Theory and Application of Cryptographic Techniques", "Annual International Conference on the Theory and Applications of Cryptographic Techniques", "EUROCRYPT", "Advances in Cryptology - EUROCRYPT"),
    ),
    "asiacrypt": Venue(
        "asiacrypt", "ASIACRYPT",
        ("Theory and Application of Cryptology and Information Security", "Advances in Cryptology ASIACRYPT"),
        ("International Conference on the Theory and Application of Cryptology and Information Security", "ASIACRYPT", "Advances in Cryptology - ASIACRYPT"),
    ),
}


def source_matches(venue: Venue, source: dict) -> bool:
    """Check an OpenAlex source's label and alternate labels, never its ID."""
    if is_excluded(source.get("display_name")):
        return False
    names = [source.get("display_name"), *(source.get("alternate_titles") or [])]
    return any(venue.matches(name) for name in names if isinstance(name, str))
