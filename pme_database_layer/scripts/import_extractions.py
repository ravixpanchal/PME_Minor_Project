from pathlib import Path
import argparse
from pme.database import init_db, get_session
from pme.importer import import_directory, import_one
from pme.schemas import PMEExtraction

p = argparse.ArgumentParser(description="Import PME extraction JSON.")
p.add_argument(
    "input",
    type=Path,
    nargs="?",
    default=Path("data/processed/extractions"),
    help="A JSON file or directory containing JSON files.",
)
a = p.parse_args()
if not a.input.exists():
    print(f"Input does not exist: {a.input}")
    raise SystemExit(1)

init_db()
s = get_session()
try:
    if a.input.is_file():
        try:
            with s.begin_nested():
                _, created = import_one(s, PMEExtraction.from_file(a.input))
            s.commit()
            print(f"[OK] {a.input.name} {'imported' if created else 'already imported'}")
            ok, bad = 1, 0
        except Exception as error:
            s.rollback()
            print(f"[ERROR] {a.input.name}: {error}")
            ok, bad = 0, 1
    else:
        ok, bad = import_directory(s, a.input)
finally:
    s.close()
print(f"\nProcessed: {ok}\nFailed: {bad}")
raise SystemExit(1 if bad else 0)
