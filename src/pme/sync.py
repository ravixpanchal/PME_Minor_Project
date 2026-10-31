"""Import complete extraction records into the dashboard database."""
from src.pme.database import get_session, init_db
from src.pme.importer import sync_extractions


def main():
    init_db()
    with get_session() as session:
        imported, failed = sync_extractions(session)
    print(f"Processed {imported} records; failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
