from __future__ import annotations

import argparse
import time
from pathlib import Path

from pme.config import get_extraction_dir
from pme.database import get_session, init_db
from pme.importer import sync_extractions


parser = argparse.ArgumentParser(
    description="Continuously import new extraction JSON files."
)
parser.add_argument(
    "--directory",
    type=Path,
    default=None,
    help="Override the default data/processed/extractions directory.",
)
parser.add_argument(
    "--interval",
    type=float,
    default=5,
    help="Seconds between scans (default: 5).",
)
args = parser.parse_args()

directory = args.directory or get_extraction_dir()
init_db()
print(f"Watching extraction directory: {directory}")
print("Press Ctrl+C to stop.")

try:
    while True:
        with get_session() as session:
            imported, failed = sync_extractions(session, directory)
        if imported or failed:
            print(f"Sync complete: processed={imported}, failed={failed}")
        time.sleep(args.interval)
except KeyboardInterrupt:
    print("\nWatcher stopped.")
