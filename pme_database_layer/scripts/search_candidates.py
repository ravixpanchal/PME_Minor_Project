from __future__ import annotations

import argparse
from datetime import datetime

from pme.database import get_session, init_db
from pme.repository import search_candidates


parser = argparse.ArgumentParser(description="Search PME candidates.")
parser.add_argument("--name", required=True)
parser.add_argument("--father-name", required=True)
parser.add_argument("--dob", required=True, help="DD-MM-YYYY or YYYY-MM-DD")
args = parser.parse_args()

try:
    dob = datetime.strptime(args.dob, "%d-%m-%Y").date()
except ValueError:
    dob = datetime.strptime(args.dob, "%Y-%m-%d").date()

init_db()
session = get_session()
try:
    matches = search_candidates(session, args.name, args.father_name, dob)
    for candidate in matches:
        print(
            f"{candidate.id}\t{candidate.roll_number or '-'}\t"
            f"{candidate.candidate_name}\t{candidate.father_name}\t"
            f"{candidate.date_of_birth}"
        )
    print(f"Matches: {len(matches)}")
finally:
    session.close()
