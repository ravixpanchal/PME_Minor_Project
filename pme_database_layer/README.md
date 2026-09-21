# PME Database Layer

The PME database layer validates Qwen extraction JSON, stores the extracted
candidate and medical-examination data in SQLAlchemy tables, and provides a
Streamlit dashboard for searching and reviewing records.

All commands below should be run from:

```bash
cd "/home/ravi/Downloads/Minor Project/pme_database_layer"
```

## Install

```bash
uv sync
```

The project uses the `uv.lock` file to install the pinned dependencies,
including SQLAlchemy, Pydantic, and Streamlit.

## Validate extraction JSON

Validate all extraction files:

```bash
uv run --project . python scripts/validate_extractions.py \
  "../data/processed/extractions"
```

Validate one extraction file:

```bash
uv run --project . python scripts/validate_extractions.py \
  "../data/processed/extractions/<document>.json"
```

## Import extraction data

Import all extracted JSON files into the database:

```bash
uv run --project . python scripts/import_extractions.py \
  "../data/processed/extractions"
```

Import one extraction directly:

```bash
uv run --project . python scripts/import_extractions.py \
  "../data/processed/extractions/<document>.json"
```

The importer is idempotent. Importing the same JSON again reports
`already imported` instead of creating a duplicate examination.

The default extraction directory is the pipeline output directory:

```text
../data/processed/extractions
```

This can be overridden with `EXTRACTION_DIR`.

## Real-time extraction sync

To continuously import new or updated extraction JSON files while the
extraction pipeline is running, start the watcher in a separate terminal:

```bash
uv run --project . python scripts/watch_extractions.py
```

The watcher scans every five seconds, writes valid records to the same
database, and keeps duplicate documents safe through
`extraction_runs.document_id`.

## Command-line search

Search by the candidate identity tuple:

```bash
uv run --project . python scripts/search_candidates.py \
  --name "PRAKASH CHANDRA MURMU" \
  --father-name "LOBIN MURMU" \
  --dob "21-02-1996"
```

## Visual dashboard

Start the browser dashboard:

```bash
uv run streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

The dashboard uses the same `DATABASE_URL` and the same `data/pme.db` database
as the import and command-line search scripts. It does not create a separate
database.

The dashboard also synchronizes the extraction directory when it starts and
provides a **Sync extraction files** button. For continuous background syncing,
run `watch_extractions.py` alongside the dashboard.

### Dashboard search fields

Enter all three values together:

| Field | Example |
|---|---|
| Candidate name | `PRAKASH CHANDRA MURMU` |
| Father's name | `LOBIN MURMU` |
| Date of birth | `21-02-1996` |

The DOB field is a text field and must use `DD-MM-YYYY` format. After a
successful search, the dashboard displays candidate identity, examination,
PME, vision, fitness, physical, and declaration details.
It also displays the original high-resolution pages from
`../data/interim/high_res/<document_id>/` and the complete validated extraction
payload. The image directory is resolved using the same `document_id` stored in
`extraction_runs`; it never matches images by candidate name. Set `INTERIM_DIR`
to override the interim-data root when running the dashboard.

Use **Candidate directory** in the sidebar to browse stored candidates.

## Database

Default database:

```text
data/pme.db
```

For MySQL, set a valid SQLAlchemy URL:

```bash
export DATABASE_URL="mysql+pymysql://USER:PASSWORD@HOST:3306/pme"
```

Open the SQLite database in DB Browser for SQLite:

```bash
/usr/bin/sqlitebrowser "/home/ravi/Downloads/Minor Project/pme_database_layer/data/pme.db"
```

If the database has not been created or populated yet, import the extraction
data first:

```bash
uv run --project . python scripts/import_extractions.py \
  "../data/processed/extractions"
```

## Tables

The database contains:

```text
candidates
medical_examinations
pme_cases
pme_case_measurements
vision_examinations
physical_examinations
fitness_classifications
declarations
doctors
extraction_runs
```

`extraction_runs.document_id` is unique, so the same extraction document
cannot create duplicate examination data. The raw validated extraction is
retained in `extraction_runs.raw_json` for provenance and auditability.

## Quick end-to-end check

```bash
uv run --project . python scripts/validate_extractions.py \
  "../data/processed/extractions"
uv run --project . python scripts/import_extractions.py \
  "../data/processed/extractions"
uv run streamlit run dashboard/app.py
```

Then search the dashboard using:

```text
Candidate name: PRAKASH CHANDRA MURMU
Father's name: LOBIN MURMU
Date of birth: 21-02-1996
```

The supplied sample record should return:

```text
Roll number: 242243220965001
Recruitment CEN: 02/2024
```

## Verified sample data

The tagged extraction file has been checked against the stored database row:

```text
Source:
../data/processed/extractions/0f33019e5dd1a3d1784d17503063233d9a3bbedc29acb6807b3c38d70d0ac483/record.json

Candidate name:          PRAKASH CHANDRA MURMU
Father's name:           LOBIN MURMU
Date of birth:           21-02-1996
Roll number:             242243220965001
Recruitment CEN:         02/2024
Medical class:           A3 and below
DV date:                 05-08-2026
Medical examination:    06-08-2026
```

The identity fields, examination dates, medical class, recruitment CEN, roll
number, and identification marks match between the extraction JSON and the
database. Related PME, vision, fitness, and declaration records are also
stored for this candidate.

The verified database counts at the time of checking were:

```text
Candidates:              12
Medical examinations:    12
PME cases:               12
Vision examinations:     12
Fitness classifications: 12
Declarations:            132
Extraction runs:         12
```

The data flow is:

```text
data/processed/extractions/*.json
              ↓
      pme_database_layer importer
              ↓
          data/pme.db
              ↓
     Streamlit dashboard
```

The importer and dashboard use the same database. New extraction files can be
picked up continuously by running:

```bash
uv run --project . python scripts/watch_extractions.py
```

The watcher scans the extraction directory every five seconds. The dashboard
also synchronizes extraction files at startup and provides a **Sync extraction
files** button. Duplicate documents remain safe because
`extraction_runs.document_id` is unique.
