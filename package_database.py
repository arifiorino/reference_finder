#!/usr/bin/env python3
"""Create compact, lossless SQLite archives or restore them without model dependencies."""
from __future__ import annotations

import argparse
import json
import lzma
import sqlite3
import sys

from reference_finder.database_package import PAGE_SIZES, pack_database, unpack_database


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack", help="snapshot, compact, and compress a SQLite database")
    pack.add_argument("--db", required=True)
    pack.add_argument("--output", required=True, help="new .sqlite3.xz file; existing files are never replaced")
    pack.add_argument("--page-size", type=int, choices=PAGE_SIZES, default=4096)
    unpack = commands.add_parser("unpack", help="verify and restore a SQLite XZ archive")
    unpack.add_argument("--archive", required=True)
    unpack.add_argument("--output", required=True, help="new .sqlite3 file; existing files are never replaced")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "pack":
            result = pack_database(args.db, args.output, args.page_size)
        else:
            result = unpack_database(args.archive, args.output)
    except (OSError, ValueError, sqlite3.Error, lzma.LZMAError, EOFError) as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
