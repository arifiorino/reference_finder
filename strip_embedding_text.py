#!/usr/bin/env python3
"""Remove duplicate paper text from a legacy embedding SQLite database."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from reference_finder.embedding_migration import strip_embedding_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/papers.sqlite3", help="separate paper SQLite database (default: data/papers.sqlite3)")
    parser.add_argument("--index", required=True, help="legacy embedding SQLite database")
    parser.add_argument("--output", required=True, help="new linked embedding SQLite file; existing files are never replaced")
    args = parser.parse_args(argv)
    try:
        result = strip_embedding_text(args.db, args.index, args.output)
    except (OSError, ValueError, sqlite3.Error) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
