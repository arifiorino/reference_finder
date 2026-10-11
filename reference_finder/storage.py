"""SQLite storage, provenance, and crash-safe import checkpoints."""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ABSTRACT_PRIORITY = {"openalex": 1, "semantic_scholar": 2, "usenix": 3, "iacr": 3}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_title(title):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", title or "").casefold()))


def normalize_doi(doi):
    if not doi:
        return None
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi.strip(), flags=re.I)
    value = unquote(value).lower()
    return value if re.match(r"^10\.\d{4,9}/\S+$", value) else None


def reconstruct_abstract(index):
    """OpenAlex provides token -> positions, not a list in reading order."""
    if not isinstance(index, dict) or not index:
        return None
    positions = {}
    for word, offsets in index.items():
        if isinstance(offsets, list):
            for offset in offsets:
                if isinstance(offset, int) and offset >= 0:
                    positions[offset] = word
    return " ".join(positions[p] for p in sorted(positions)).strip() or None


def paper_from_record(provider, record):
    if provider in {"dblp", "usenix", "iacr", "ccs"}:
        return {
            "external_id": record["external_id"], "doi": normalize_doi(record.get("doi")),
            "title": record.get("title") or "", "year": record.get("publication_year", record.get("year")), "date": None,
            "abstract": (record.get("abstract") or "").strip() or None,
            "authors": record.get("authors") or [], "url": record.get("url"),
        }
    if provider == "openalex":
        return {
            "external_id": record["id"], "doi": normalize_doi(record.get("doi")),
            "title": record.get("display_name") or record.get("title") or "",
            "year": record.get("publication_year"), "date": record.get("publication_date"),
            "abstract": reconstruct_abstract(record.get("abstract_inverted_index")),
            "authors": [a.get("author", {}).get("display_name", "") for a in record.get("authorships") or []],
            "url": (record.get("primary_location") or {}).get("landing_page_url") or record["id"],
        }
    abstract = record.get("abstract")
    return {
        "external_id": record["paperId"], "doi": normalize_doi((record.get("externalIds") or {}).get("DOI")),
        "title": record.get("title") or "", "year": record.get("year"), "date": record.get("publicationDate"),
        "abstract": abstract.strip() if isinstance(abstract, str) and abstract.strip() else None,
        "authors": [a.get("name", "") for a in record.get("authors") or []],
        "url": record.get("url") or "https://www.semanticscholar.org/paper/" + record["paperId"],
    }


class Database:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(Path(path).expanduser()) if str(path) != ":memory:" else ":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError(f"Unsupported database schema version {version}")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY, doi TEXT UNIQUE, title TEXT NOT NULL,
                normalized_title TEXT NOT NULL, publication_year INTEGER, publication_date TEXT,
                authors_json TEXT NOT NULL DEFAULT '[]', abstract TEXT, abstract_source TEXT,
                url TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS papers_title_year ON papers(normalized_title, publication_year);
            CREATE TABLE IF NOT EXISTS source_records (
                provider TEXT NOT NULL, external_id TEXT NOT NULL,
                paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                raw_json TEXT NOT NULL, fetched_at TEXT NOT NULL,
                PRIMARY KEY(provider, external_id)
            );
            CREATE INDEX IF NOT EXISTS source_records_paper ON source_records(paper_id);
            CREATE TABLE IF NOT EXISTS paper_venues (
                paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                venue TEXT NOT NULL, year INTEGER NOT NULL, provider TEXT NOT NULL,
                PRIMARY KEY(paper_id, venue, year, provider)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                job TEXT PRIMARY KEY, cursor TEXT, done INTEGER NOT NULL DEFAULT 0,
                fetched INTEGER NOT NULL DEFAULT 0, accepted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS openalex_sources (
                venue TEXT NOT NULL, source_id TEXT NOT NULL, name TEXT NOT NULL,
                PRIMARY KEY(venue, source_id)
            );
            CREATE TABLE IF NOT EXISTS enrichment_attempts (
                provider TEXT NOT NULL, doi TEXT NOT NULL, outcome TEXT NOT NULL, checked_at TEXT NOT NULL,
                PRIMARY KEY(provider, doi)
            );
            CREATE TABLE IF NOT EXISTS roster_runs (
                provider TEXT NOT NULL, venue TEXT NOT NULL, year INTEGER NOT NULL,
                status TEXT NOT NULL, note TEXT NOT NULL, urls_json TEXT NOT NULL,
                record_count INTEGER NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(provider,venue,year)
            );
            CREATE TABLE IF NOT EXISTS roster_entries (
                provider TEXT NOT NULL, external_id TEXT NOT NULL, venue TEXT NOT NULL,
                year INTEGER NOT NULL, paper_id INTEGER NOT NULL REFERENCES papers(id),
                title TEXT NOT NULL, doi TEXT, url TEXT, authors_json TEXT NOT NULL,
                section TEXT, raw_json TEXT NOT NULL,
                PRIMARY KEY(provider,external_id,venue,year)
            );
            CREATE INDEX IF NOT EXISTS roster_entries_paper ON roster_entries(paper_id);
            PRAGMA user_version=1;
        """)

    def checkpoint(self, job):
        return self.conn.execute("SELECT * FROM checkpoints WHERE job=?", (job,)).fetchone()

    def save_checkpoint(self, job, cursor, done, fetched, accepted):
        self.conn.execute("""INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job) DO UPDATE SET cursor=excluded.cursor, done=excluded.done,
            fetched=checkpoints.fetched+excluded.fetched, accepted=checkpoints.accepted+excluded.accepted,
            updated_at=excluded.updated_at""", (job, cursor, int(done), fetched, accepted, now()))

    def _merge(self, keep, remove):
        """Join records that a later DOI lookup proves are the same paper."""
        target = self.conn.execute("SELECT * FROM papers WHERE id=?", (keep,)).fetchone()
        old = self.conn.execute("SELECT * FROM papers WHERE id=?", (remove,)).fetchone()
        if old["abstract"] and (not target["abstract"] or ABSTRACT_PRIORITY.get(old["abstract_source"], 0) > ABSTRACT_PRIORITY.get(target["abstract_source"], 0)):
            self.conn.execute("UPDATE papers SET abstract=?, abstract_source=? WHERE id=?",
                              (old["abstract"], old["abstract_source"], keep))
        self.conn.execute("UPDATE source_records SET paper_id=? WHERE paper_id=?", (keep, remove))
        self.conn.execute("UPDATE roster_entries SET paper_id=? WHERE paper_id=?", (keep, remove))
        self.conn.execute("""INSERT OR IGNORE INTO paper_venues
            SELECT ?, venue, year, provider FROM paper_venues WHERE paper_id=?""", (keep, remove))
        self.conn.execute("DELETE FROM papers WHERE id=?", (remove,))
        if not target["doi"] and old["doi"]:
            self.conn.execute("UPDATE papers SET doi=? WHERE id=?", (old["doi"], keep))

    def upsert(self, provider, record, venue=None, year=None, known_paper_id=None):
        p = paper_from_record(provider, record)
        title_key = normalize_title(p["title"])
        matches = set()
        if known_paper_id is not None:
            matches.add(known_paper_id)
        identity = self.conn.execute("SELECT paper_id FROM source_records WHERE provider=? AND external_id=?",
                                     (provider, p["external_id"])).fetchone()
        if identity:
            matches.add(identity[0])
        if p["doi"]:
            match = self.conn.execute("SELECT id FROM papers WHERE doi=?", (p["doi"],)).fetchone()
            if match:
                matches.add(match[0])
        if not matches and title_key and venue:
            # Conservative fallback for papers without DOIs (common at USENIX).
            rows = self.conn.execute("""SELECT DISTINCT p.id, p.doi, p.authors_json FROM papers p
                JOIN paper_venues v ON v.paper_id=p.id WHERE normalized_title=? AND v.year=? AND v.venue=?""",
                (title_key, year, venue)).fetchall()
            candidates = []
            for row in rows:
                old_authors = json.loads(row["authors_json"])
                same_author = bool(p["authors"] and old_authors and normalize_title(p["authors"][0]) and
                                   normalize_title(p["authors"][0]) == normalize_title(old_authors[0]))
                if (not p["doi"] or not row["doi"]) and same_author:
                    candidates.append(row["id"])
            if len(candidates) == 1:
                matches.add(candidates[0])
        timestamp = now()
        if matches:
            paper_id = min(matches)
            for other in sorted(matches - {paper_id}):
                self._merge(paper_id, other)
            old = self.conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
            use_abstract = bool(p["abstract"] and (not old["abstract"] or
                                ABSTRACT_PRIORITY.get(provider, 0) >= ABSTRACT_PRIORITY.get(old["abstract_source"], 0)))
            self.conn.execute("""UPDATE papers SET doi=COALESCE(doi,?), title=?, normalized_title=?,
                publication_year=COALESCE(publication_year,?), publication_date=COALESCE(publication_date,?),
                authors_json=?, abstract=?, abstract_source=?, url=COALESCE(url,?), updated_at=? WHERE id=?""",
                (p["doi"], p["title"] or old["title"], title_key or old["normalized_title"], p["year"], p["date"],
                 json.dumps(p["authors"], ensure_ascii=False) if p["authors"] else old["authors_json"],
                 p["abstract"] if use_abstract else old["abstract"], provider if use_abstract else old["abstract_source"],
                 p["url"], timestamp, paper_id))
        else:
            result = self.conn.execute("""INSERT INTO papers
                (doi,title,normalized_title,publication_year,publication_date,authors_json,abstract,abstract_source,url,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (p["doi"], p["title"], title_key, p["year"], p["date"], json.dumps(p["authors"], ensure_ascii=False),
                 p["abstract"], provider if p["abstract"] else None, p["url"], timestamp, timestamp))
            paper_id = result.lastrowid
        self.conn.execute("""INSERT INTO source_records VALUES (?,?,?,?,?) ON CONFLICT(provider,external_id)
            DO UPDATE SET paper_id=excluded.paper_id,raw_json=excluded.raw_json,fetched_at=excluded.fetched_at""",
            (provider, p["external_id"], paper_id, json.dumps(record, ensure_ascii=False), timestamp))
        if venue is not None:
            self.conn.execute("INSERT OR IGNORE INTO paper_venues VALUES (?,?,?,?)", (paper_id, venue, year, provider))
        if p["doi"]:
            self.mark_enrichment(provider, p["doi"], "abstract" if p["abstract"] else "no_abstract")
        return paper_id

    def mark_enrichment(self, provider, doi, outcome):
        self.conn.execute("INSERT OR REPLACE INTO enrichment_attempts VALUES (?,?,?,?)", (provider, doi, outcome, now()))

    def missing_for(self, provider, venues, start, end, limit=100):
        placeholders = ",".join("?" for _ in venues)
        return self.conn.execute(f"""SELECT DISTINCT p.id,p.doi FROM papers p JOIN paper_venues v ON p.id=v.paper_id
            WHERE p.abstract IS NULL AND p.doi IS NOT NULL AND v.venue IN ({placeholders}) AND v.year BETWEEN ? AND ?
            AND NOT EXISTS (SELECT 1 FROM enrichment_attempts a WHERE a.provider=? AND a.doi=p.doi)
            ORDER BY p.id LIMIT ?""", (*venues, start, end, provider, limit)).fetchall()

    def refresh(self, venues, start, end):
        with self.conn:
            # Job keys include venue/year; source discovery is refreshed for each selected venue too.
            for row in self.conn.execute("SELECT job FROM checkpoints").fetchall():
                job = json.loads(row[0])
                if job.get("venue") in venues and ("year" not in job or start <= job["year"] <= end):
                    self.conn.execute("DELETE FROM checkpoints WHERE job=?", (row[0],))
            placeholders = ",".join("?" for _ in venues)
            self.conn.execute(f"""DELETE FROM enrichment_attempts WHERE doi IN (
                SELECT p.doi FROM papers p JOIN paper_venues v ON p.id=v.paper_id
                WHERE v.venue IN ({placeholders}) AND v.year BETWEEN ? AND ?)""", (*venues, start, end))

    def coverage(self, venue, year):
        return self.conn.execute("""SELECT COUNT(DISTINCT p.id) AS papers,
            COUNT(DISTINCT CASE WHEN abstract IS NOT NULL THEN p.id END) AS abstracts,
            COUNT(DISTINCT CASE WHEN abstract IS NULL AND doi IS NULL THEN p.id END) AS missing_without_doi
            FROM papers p JOIN paper_venues v ON p.id=v.paper_id WHERE v.venue=? AND v.year=?""", (venue, year)).fetchone()
