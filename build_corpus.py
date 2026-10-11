#!/usr/bin/env python3
"""Build an explicit proceedings roster, retrieve official abstracts, and export it."""
from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
import sqlite3
import sys

from reference_finder.api import APIError, Budget, RequestLimit
from reference_finder.corpus_report import export_corpus
from reference_finder.roster import RosterStore
from reference_finder.storage import Database
from reference_finder.venues import VENUES
from reference_finder.webfetch import WebFetcher


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", default="data/references.sqlite3")
    p.add_argument("--start-year", type=int, default=date.today().year - 9)
    p.add_argument("--end-year", type=int, default=date.today().year)
    p.add_argument("--venues", nargs="+", choices=list(VENUES), default=list(VENUES))
    p.add_argument("--sources", nargs="+", choices=["dblp", "usenix", "iacr", "ccs"], default=["dblp", "usenix", "iacr", "ccs"])
    p.add_argument("--output", default="data/corpus")
    p.add_argument("--cache-dir", default="data/page_cache")
    p.add_argument("--refresh", action="store_true", help="Refetch source lists and rebuild their roster membership")
    p.add_argument("--report-only", action="store_true", help="Regenerate CSV/HTML from SQLite, with no network")
    p.add_argument("--max-requests", type=int, help="Bound HTTP attempts; rerun without --refresh to resume")
    p.add_argument("--interval", type=float, default=1.0, help="Minimum seconds between requests to the same host")
    args = p.parse_args(argv)
    if not 2017 <= args.start_year <= args.end_year <= date.today().year:
        p.error("use 2017 <= start-year <= end-year <= current year")
    if args.interval < 0 or (args.max_requests is not None and args.max_requests <= 0):
        p.error("interval must be nonnegative and max-requests positive")
    if args.report_only and args.refresh:
        p.error("--report-only cannot be combined with --refresh")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not args.report_only:
        try:
            from reference_finder.dblp import collect_dblp
            from reference_finder.usenix import collect_usenix
            from reference_finder.iacr import collect_iacr
            from reference_finder.ccs import collect_ccs
        except ImportError as exc:
            if exc.name == "bs4":
                p.error("BeautifulSoup is required: python3 -m pip install -r requirements.txt (or use .venv/bin/python)")
            raise
    db_path = Path(args.db).expanduser()
    backup = db_path.with_name(db_path.stem + ".before-roster.sqlite3")
    if db_path.exists() and not backup.exists() and not args.report_only:
        with sqlite3.connect(str(db_path)) as source, sqlite3.connect(str(backup)) as target:
            source.backup(target)
        logging.info("Saved pre-roster backup to %s", backup)
    db = Database(str(db_path))
    store = RosterStore(db)
    venues = [VENUES[v] for v in dict.fromkeys(args.venues)]
    budget = Budget(args.max_requests)
    fetch = WebFetcher(args.cache_dir, args.refresh, args.interval, budget)
    status = 0
    try:
        if not args.report_only:
            for provider in dict.fromkeys(args.sources):
                for venue in venues:
                    if provider == "usenix" and venue.key != "usenix-security":
                        continue
                    if provider == "iacr" and venue.key not in {"crypto", "eurocrypt", "asiacrypt"}:
                        continue
                    if provider == "ccs" and venue.key != "ccs":
                        continue
                    for year in range(args.start_year, args.end_year + 1):
                        if provider == "ccs" and year != 2026:
                            continue
                        if not args.refresh and store.finished(provider, venue.key, year):
                            continue
                        try:
                            if provider == "dblp":
                                result = collect_dblp(venue.key, year, fetch)
                            elif provider == "usenix":
                                result = collect_usenix(year, fetch)
                            elif provider == "ccs":
                                result = collect_ccs(year, fetch)
                            else:
                                result = collect_iacr(venue.key, year, fetch)
                        except APIError as exc:
                            result = {"records": [], "urls": [], "status": "incomplete", "note": str(exc)}
                        result = store.save(provider, venue.key, year, result)
                        abstracts = sum(bool(r.get("abstract")) for r in result["records"])
                        logging.info("%s %-15s %s: %s papers, %s source abstracts [%s] %s", provider, venue.key, year,
                                     len(result["records"]), abstracts, result["status"], result.get("note", ""))
                        if result["status"] == "incomplete":
                            status = 1
    except RequestLimit as exc:
        logging.info("%s", exc)
        status = 2
    except KeyboardInterrupt:
        logging.warning("Interrupted; saved source lists will resume on rerun.")
        status = 130
    finally:
        export_corpus(store, venues, args.start_year, args.end_year, args.output)
        logging.info("HTTP requests this run: %s", budget.used)
        db.conn.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
