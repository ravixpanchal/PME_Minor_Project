# Doctor evidence and recognition

## Source integration

This implementation uses `3aryan8/pme_database` main at `948afcb` as the source
baseline. It retains the PDF merge stage, module orchestration, generic text
extraction, report-image storage, Page 2 vision rows, current dashboard metrics
and sidebar, and existing report generation. Doctor recognition is integrated
into `src/database` rather than introducing a second database package.

## GPU system commands

Use Python 3.11+ and a CUDA-enabled PyTorch build compatible with your NVIDIA
driver. Core dependencies and the VLM optional dependencies are in `pyproject.toml`
and `uv.lock`; the implementation uses Transformers 4.x. No model training,
Tesseract, or external OCR service is required. Qwen2.5-VL performs localization
and crop OCR; first use needs downloaded model weights or an existing HF cache.

```bash
uv sync --extra vlm --group dev
uv run --extra vlm python -m src.verify_gpu
# Configure configs/pdf_sources.yaml and verify configs/splits.yaml boundaries.
uv run --extra vlm python -m pipelines.run_all

# Existing manifest: add doctor evidence to older extraction records.
CUDA_VISIBLE_DEVICES=0 uv run --extra vlm python -m src.extraction.run_extraction --overwrite
# Optional: --ndocs 2, --doc DOCUMENT_ID, --source raw, or --mode per_page
uv run python -m src.database.run_database
uv run streamlit run frontend/app.py

# CPU-only regression suite, with synthetic images/fake model responses.
uv run --group dev pytest tests -q
```

The central `pipelines.run_all` runner includes PDF merge, assembly, preprocessing,
detection, extraction and database import. Its GPU child commands request the VLM
extra. The compatibility `src.run_pipeline` runner remains the upstream CPU-only
merge/assembly/preprocessing entry point. The generic text-only command under
`scripts/run_pipeline.py` cannot localize image evidence; use the VLM extraction
command above to produce signature/stamp bounding boxes.

`VLM_DEVICE=cuda:0` is the default. The shared lazy loader selects BF16 on supported
GPUs and FP16 otherwise. `VLM_DEVICE=cpu` is available for troubleshooting, but full
model inference is slow. No memory capacity or speed claims have been validated.

Extraction expects four ordered pages for the candidate-report format (zero-based
indices 0..3). Blank pages retain their position. Doctor evidence is inspected only
on indices 2 and 3. The database preserves upstream support for importing records
with other declared page counts and checks rendered page completeness.

## Extraction and recognition behavior

- Candidate fields use the existing nested schema/provenance contract. Additional
  explicit fields cover clinical findings, contact details, declaration date,
  chest X-ray and advice used by the dashboard.
- On pages 3 and 4, the existing loaded VLM locates the medical examiner's signature
  and stamp, excluding candidate/witness signatures. Invalid/out-of-bounds,
  degenerate and oversized boxes are rejected.
- Save valid crops and full-page annotations before OCR or recognition. An unclear
  signature, overlapping stamp, OCR failure, multiple doctors or identity conflict
  must not discard localized evidence. Crops use original, unannotated renders.
- A readable stamp name requires two agreeing crop-OCR reads and inclusion in the
  transcribed stamp text. Names are normalized only for comparisons, not guessed.
- Match signatures with SIFT descriptors, ratio filtering, RANSAC geometry and
  aligned ink overlap. Initial thresholds are 0.90 similarity, 0.10 runner-up
  margin and 18 spatially distributed inliers, configured in
  `configs/doctor_recognition.yaml`. Similarity is not an identity probability.
- Enroll a new doctor only when a usable signature and legible stamp name are
  unambiguously paired on the same page. Both pages are inspected before enrollment;
  only one selected reference is enrolled. Conflicting stamp names, multiple
  doctors, weak same-name matches or disagreeing page signatures require review.
- A reliable signature match retrieves the saved name when no current stamp is
  readable. A contradictory readable name blocks identification. Signature-only
  matches do not add gallery references, preventing progressive gallery drift.
- Overlapping/low-quality signatures are retained for review but excluded from
  automatic matching/enrollment. Same-name weak matches are never auto-merged.

Statuses: `registered`, `matched`, `unmatched`, `missing`, `missing_signature`,
`ambiguous`, `identity_conflict`, `needs_review`, `error`. A clear stamp without a
usable signature may display its OCR name but cannot enroll a signature.

Both document and per-page extraction modes run doctor recognition. Field extraction
failure still attempts doctor evidence, preserves a null-shaped candidate record
and returns a failure status. Existing `record.json` files are skipped unless
`--overwrite` is passed. Review errors and matching notes after running.

## Outputs and dashboard

Within `data/processed/extractions/DOCUMENT_ID/`:

- `record.json` contains candidate fields and typed `doctor_recognition` metadata.
- `doctor_evidence/page_0002/` and `doctor_evidence/page_0003/` hold `signature.png`,
  `stamp.png`, `combined.png` and `annotated.png` when applicable.
- Blue boxes enclose signatures, orange boxes enclose stamps, and a green box
  encloses their union. If only one region is detected, the combined box encloses
  that region. Both normalized and pixel coordinates, image dimensions, source
  page, paths, OCR text and quality notes are saved per page.
- The dashboard shows selected crops and expandable annotations/combined crops
  for each inspected page. Unavailable crops are never substituted with a different
  report's evidence. Current record JSON can be reviewed before database sync;
  resync to refresh the candidate/clinical panels.

`data/processed/doctors/registry.sqlite3` stores unique normalized doctor identities,
original PNG signature/stamp blobs and source documents. Enrollment is serialized
transactionally. Keep this file across runs; back up/copy `data/` when moving machines.
Generated galleries/crops stay ignored. The Makefile clean command preserves the
configured default gallery directory; custom gallery locations need their own backup.
No manual identity-review UI is included; uncertain cases remain unenrolled.

## Database schema and upgrades

`src.database.run_database` and dashboard startup initialize the schema. New tables:

- `doctor_registry_links`: stable gallery ID linked to a distinct dashboard doctor.
  Different gallery IDs are not merged just because display names match.
- `doctor_recognition_results`: one result per extraction run, with status,
  constrained 0..1 similarity, algorithm, observed name and optional identity link.
- `doctor_evidence`: per-report-page boxes, dimensions, paths, OCR text, quality flags
  and notes; unique per extraction run/page and restricted to report pages 3 and 4.

Existing tables are retained. Older SQLite databases with NOT NULL roll numbers
are upgraded by a transactional candidates-table rebuild preserving rows, indexes,
triggers and referencing rows; foreign-key checks must pass before commit. The
upgrade is idempotent. PostgreSQL/MySQL use an explicit ALTER for nullability.
Back up your deployed database before upgrading; MySQL/PostgreSQL upgrades were
not exercised in this environment.

Raw extraction JSON remains available for audit. Import recognizes only canonical
`record.json` files, skips diagnostics, fills previously missing roll numbers,
prevents conflicting changes to established identities and keeps examination/run
IDs stable when clinical data changes. Updated evidence/report-image rows are
replaced atomically. Existing raw metadata can backfill the new recognition tables.
Missing rendered pages still fail atomically, as required by the upstream importer.

## Changed areas and verification

- `src/extraction/{run_extraction,vlm_extractor,doctor_recognition,doctor_evidence}.py`:
  candidate extraction, doctor crop/OCR/matching, annotations and persistence.
- `src/models/schemas.py`, `configs/{schema,doctor_recognition}.yaml`: typed output,
  page evidence contract, missing clinical fields and quality thresholds.
- `src/database/{models,schemas,importer,database,migrations}.py`: normalized evidence,
  stable identities/IDs, schema upgrades and SQLite foreign-key enforcement.
- `frontend/app.py`: evidence UI added to the current upstream dashboard.
- `src/utils/vlm_runtime.py`, `src/detection/vlm_detector.py`, `src/verify_gpu.py`:
  shared lazy model loading and failing status when CUDA is unavailable.
- `pipelines/run_all.py`, `pyproject.toml`, `uv.lock`: orchestration/extras and
  packaged pipeline entry point. Existing upstream Docker files are retained.
- `.gitignore`, `Makefile`: track synthetic fixtures and protect persistent galleries.
- `tests/`: feature regressions, schema upgrade tests, and repaired upstream tests
  that previously depended on an ignored local manifest or missing JSON fixture.

CPU-only regression tests use synthetic images, temporary SQLite databases and
fake VLM responses. No real reports were extracted and no models were loaded,
trained or run on a GPU. Python compilation and whitespace checks were also used.
Real-report localization/OCR accuracy, signature variation, false identification
rates, calibration, GPU memory/runtime, live Streamlit behavior and non-SQLite
schema upgrades remain unverified. The matcher is a conservative untrained baseline;
validate a representative labeled holdout and review initial gallery enrollments
before relying on its automated doctor names.

## Recovering a four-candidate trial

Date parsing accepts day/month/two-digit-year report dates such as `29/7/25` and
`12-08-26` as well as four-digit-year formats. Two-digit years follow Python's
standard pivot (00–68 => 2000–2068, 69–99 => 1969–1999). Invalid calendar dates
still fail validation; raw date strings remain in extraction JSON for audit.

To refresh only doctor evidence for the four existing extracted candidates:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --extra vlm python -m src.extraction.run_extraction --ndocs 4 --doctor-only
uv run python -m src.database.run_database --selection data/processed/extractions/last_selection.json
uv run streamlit run frontend/app.py
```

Use `--ndocs 4 --overwrite` instead of `--doctor-only` for a first extraction or
when candidate fields also need refreshing. `last_selection.json` records the
selected document IDs; the database `--selection` option excludes unrelated old
records. Streamlit no longer reimports every saved record on every refresh; its
Sync button imports the current selection, and the view defaults to that selection.
The normal database command without `--selection` still imports all saved records.

Doctor localization now requests absolute pixels with an explicit coordinate
space, then maps boxes to original-image coordinates. Normalized 0..1 and explicitly
declared normalized 0..1000 responses are also supported. Unexpected coordinate
systems are rejected rather than guessed. Detection uses a larger 2016-pixel long
edge. If no valid boxes are found, it retries a closer view of the lower 65% of the
page and maps coordinates back to the full page; candidate signatures remain
excluded by the prompt. This retry does not guarantee detection on real reports.

Every doctor VLM response is saved with input dimensions/prompt in
`doctor_evidence/page_XXXX/response_XX.json`, even when parsing fails or the model
returns empty boxes. Logs explicitly report saved crops, matching status, name,
notes and diagnostic location. These files contain medical evidence and should
be handled with the same care as the reports. No quality or matching thresholds
were lowered to force an identity. If a report still produces no crops/name,
inspect its `record.json` doctor metadata and saved responses to distinguish an
empty detection, coordinate problem, OCR disagreement or quality rejection.
