"""Run database validation and import for the normalized database module.

Phase: database import.
Input: canonical ``record.json`` extraction files under the selected directory.
Output: validated extraction summaries and normalized database rows.
Commands:
  ``uv run python -m src.database.run_database --help``
  ``uv run python -m src.database.run_database --directory data/processed/extractions``
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.database.database import get_session, init_db
from src.database.importer import import_directory
from scripts.validate_extractions import canonical_records
from src.database.schemas import PMEExtraction
from src.database.selection import selected_documents


def main() -> int:
    """Validate canonical records, then import them into the database."""
    parser = argparse.ArgumentParser(description="Validate and import database records")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("data/processed/extractions"),
    )
    parser.add_argument("--selection", type=Path, help="Import only document IDs listed in an extraction last_selection.json")
    args = parser.parse_args()
    document_ids = selected_documents(args.selection) if args.selection else None
    if not args.directory.exists():
        parser.error(f"directory does not exist: {args.directory}")

    invalid = []
    for path in canonical_records(args.directory):
        if document_ids is not None and path.parent.name not in document_ids:
            continue
        try:
            PMEExtraction.from_file(path)
        except Exception as error:
            invalid.append((path, error))
    if invalid:
        for path, error in invalid:
            print(f"[ERROR] {path}: {error}")
        return 1

    init_db()
    session = get_session()
    try:
        successful, failed = import_directory(session, args.directory, document_ids=document_ids)
    finally:
        session.close()
    print(f"\nProcessed: {successful}\nFailed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
