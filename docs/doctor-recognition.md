# Doctor recognition and GPU execution

## Setup on the GPU system

Use Python 3.11+ and an NVIDIA driver compatible with your CUDA-enabled PyTorch
build. From the repository root on branch `sign-codex`:

```bash
git switch sign-codex
uv sync --extra vlm --group dev
uv run --extra vlm python -m src.verify_gpu
```

If the check fails, install a PyTorch build appropriate to your driver using the
[official PyTorch installer](https://pytorch.org/get-started/locally/) into `.venv`.
Do not proceed until verification succeeds. The VLM extra includes PyTorch,
torchvision, Transformers 4.x and Accelerate; core dependencies include OpenCV,
Pillow, PyMuPDF, pandas/pyarrow, Pydantic, SQLAlchemy and Streamlit. No Tesseract,
external OCR service, pretrained signature model or training job is required.
The existing Qwen2.5-VL model performs page localization and stamp OCR. Its first
use downloads model weights and needs internet access or a populated HF cache.

The loader uses `VLM_DEVICE=cuda:0` by default and selects BF16 when supported,
otherwise FP16. `CUDA_VISIBLE_DEVICES` can choose the physical GPU. `VLM_DEVICE=cpu`
is available for deployment troubleshooting, but full extraction is slow there.
The implementation follows the [Qwen2.5-VL Transformers interface](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/qwen2_5_vl).
No GPU memory capacity or throughput claim has been validated for this project.

## Commands

Configure the source PDF, its SHA256 and confirmed four-page candidate boundaries
in `configs/splits.yaml` before assembly. Inspect the boundaries before setting
`confirmed: true`. Assembly deliberately refuses an unconfirmed map.

```bash
# New PDF: assembly, preprocessing and extraction (now includes doctor recognition).
CUDA_VISIBLE_DEVICES=0 uv run --extra vlm python -m src.run_pipeline --phase all

# Existing manifest: extraction only, including signature/stamp processing.
CUDA_VISIBLE_DEVICES=0 uv run --extra vlm python -m src.extraction.run_extraction --source raw --overwrite

# Smaller trial on the GPU system; still uses all four pages per candidate.
CUDA_VISIBLE_DEVICES=0 uv run --extra vlm python -m src.extraction.run_extraction --ndocs 2 --overwrite

# Optional page-by-page extraction; doctor recognition is the same in either mode.
uv run --extra vlm python -m src.extraction.run_extraction --mode per_page --overwrite

# Refresh the dashboard database, then start the UI.
uv run python -m src.pme.sync
uv run streamlit run frontend/app.py

# CPU-only unit tests; fake VLM responses and synthetic images, no model loading.
uv run --group dev pytest tests -q
```

`--doc DOCUMENT_ID` selects one candidate. `--source vlm` (default) uses smaller
field-extraction images; `--source raw` uses original renders and requires more
VRAM. Doctor evidence always comes from original renders, regardless of this
option. Default Qwen is `Qwen/Qwen2.5-VL-7B-Instruct`, configured in `configs/models.yaml`.
`--phase extract` is also available on the pipeline entry point. `--limit` is only
for preprocessing; combined `all --limit` is rejected to prevent an incomplete
manifest being passed into extraction. Blank pages retain their page positions.
Incomplete documents or duplicate page indices are rejected, rather than shifting
page numbers. Review logs and nonzero exit codes for failures.

Existing `record.json` files are skipped unless `--overwrite` is supplied. Use
that flag once to add recognition to older records. Recognition errors remain
visible in the record and cause extraction to return a failure exit code while
preserving extracted fields. A missing or uncertain signature is a normal result,
not an execution error. The dashboard Sync button is equivalent to `src.pme.sync`.

## Recognition workflow

1. Extract all configured candidate fields from the four pages, preserving the
   original field tree, field provenance and per-page conflict behavior.
2. Inspect only zero-based page indices 2 and 3 (report pages 3 and 4). Reuse the
   loaded VLM to locate the medical examiner signature and stamp. Candidate and
   witness signatures are explicitly excluded. Coordinates are normalized to the
   raw image; malformed, tiny, oversized or out-of-bounds boxes are rejected.
3. Save localized crops and page annotations before OCR or recognition. Blank,
   low-ink, blurred or overlapping signature crops are retained for review but
   excluded from matching, so stamp text cannot drive a signature match.
4. OCR each usable stamp crop twice with different prompts. A personal name is
   eligible for enrollment only if both reads agree after case/title/punctuation
   normalization, it occurs in the transcribed stamp text, and the VLM identifies
   that stamp as belonging to that signature on the same page. No candidate name
   or full-page doctor guess is used to enroll a signature.
5. Inspect both pages before enrollment. Conflicting names/multiple doctors or
   signatures that cannot be reliably compared require review. Select one best
   evidence set for matching and retain both pages as review evidence; do not
   enroll duplicates from both pages or join an unpaired
   stamp on one page to a signature on the other.
6. Compare the signature to stored signatures using SIFT features, a strict
   nearest-neighbor ratio, RANSAC geometry and aligned ink overlap. Require at
   least 18 inliers spread across the signature, plausible scale/rotation, a
   score of at least 0.90, and a 0.10 margin over the next doctor. These initial
   values are configurable in `configs/doctor_recognition.yaml`.
7. A reliable match retrieves the persisted doctor's name even when the current
   report has no readable stamp. A conflicting current stamp name blocks that
   match. Signature-only matches never modify the gallery, avoiding feedback
   that progressively enrolls mistaken identities.
8. If there is no reliable match, enroll only a usable signature with its paired,
   readable stamp name. A same-name weak match or moderate resemblance (>=0.60)
   goes to review instead of creating another doctor. Namesakes are never merged
   automatically. Identical normalized names have a database uniqueness constraint.

A new enrollment has similarity score 0 when the gallery was empty. That score
is its similarity to an existing doctor, not confidence in OCR or enrollment.
Two agreeing reads from one model are consistency checks, not independent proof.

## Persistent outputs and dashboard

- `data/processed/extractions/DOCUMENT_ID/record.json`: candidate fields plus a
  typed `doctor_recognition` object (status, name/ID, similarity, selected page,
  boxes, crop paths, OCR text, notes and algorithm version).
- `doctor_evidence/page_0002/` and `doctor_evidence/page_0003/` within each
  document directory contain `signature.png`, `stamp.png`, `combined.png` and
  `annotated.png` whenever valid localized boxes are available. The full-page
  annotation draws blue signature, orange stamp and green combined-region boxes.
  The combined region is the union of valid signature/stamp boxes; if only one
  exists it encloses that one region. Crops always use the unannotated source.
  Coordinates are saved in normalized (0..1) and original-image pixel units with
  source image dimensions and zero-based page index. Invalid boxes are neither
  drawn nor saved. Paths are repository-relative; unavailable crops are never
  substituted with another report's evidence. `doctor_recognition.evidence` lists
  each inspected page; top-level paths point to the selected evidence set.
  Evidence remains available on OCR failure, identity conflict or ambiguity.
- `data/processed/doctors/registry.sqlite3`: canonical doctor name, normalized
  unique name key, original PNG signature and stamp blobs, source document and
  creation time. Lookup/enrollment is transactionally serialized. No paths to the
  source machine are required for matching on the next execution.
- `data/pme.db`: dashboard candidate database by default. `DATABASE_URL` can select
  another SQLAlchemy-compatible database; `EXTRACTION_DIR` and `INTERIM_DIR` can
  select existing output/image directories.

Back up/copy the entire `data/` directory between machines; especially retain the
doctor registry. Generated data remains gitignored. A fresh clone does not include
medical reports or an existing doctor gallery. Do not delete the gallery between
runs. Registry entries can be inspected with a SQLite viewer; `needs_review` does
not auto-enroll, and there is no review/approval UI in this change. Corrected or
additional reference signatures must be reviewed before changing gallery rows.
Changing the matching algorithm requires reevaluation of thresholds and gallery.

The dashboard displays doctor name, signature/stamp crops, status and similarity,
as well as expandable per-page full-image annotations and combined crops,
plus all raw extracted fields. Existing identity, examination, vision and fitness
panels remain. Import skips document/per-page diagnostic sidecars, preserves
recognition metadata, and atomically refreshes a document's examination when its
record changes, preserving examination and extraction-run IDs. Foreign-key
checks are enabled on SQLite connections. Newly available roll numbers are saved;
conflicting changes to established roll numbers fail with a review message and
roll back the refresh instead of changing the candidate identity. Candidates with missing roll numbers are not merged through a
NULL roll number. Resync after overwriting extractions to update all candidate
panels; the evidence panel can already read the current record.

Statuses: `registered`, `matched`, `unmatched`, `missing`, `missing_signature`,
`ambiguous`, `identity_conflict`, `needs_review`, `error`. An uncertain match never
returns a registered doctor's name. A readable stamp without a usable signature
may display that OCR name, but does not enroll a signature.

## Changed files

- `configs/schema.yaml`: explicit page-3 clinical fields and page-2 X-ray/advice;
  page-4 contact details and declaration date previously absent from extraction.
- `configs/doctor_recognition.yaml`: storage path and conservative matching/quality
  thresholds, validated before model loading.
- `src/models/schemas.py`: backward-compatible typed doctor metadata contract.
- `src/extraction/doctor_evidence.py`: deterministic crop/annotation generation.
- `src/extraction/doctor_recognition.py`: localization, crop validation, OCR checks,
  image matching, persistence and safe abstention.
- `src/extraction/run_extraction.py`: recognition in both extraction modes, strict
  page ordering, missing-image fallback, correct array field page attribution,
  parse-failure reporting and atomic record replacement.
- `src/utils/vlm_runtime.py`, `src/extraction/vlm_extractor.py`,
  `src/detection/vlm_detector.py`: shared lazy GPU-compatible model loader.
- `src/run_pipeline.py`: extraction included in the combined pipeline.
- `src/verify_gpu.py`: CUDA failures return a failing status instead of printing PASS.
- `src/pme/{__init__,config,database,models,repository,schemas,source_images}.py`:
  bring the currently used database code out of ignored `data/database/` into
  versionable source. `schemas.py` now accepts doctor metadata/page-4 contacts.
- `src/pme/importer.py`: complete-record import, refresh handling, recognized names,
  contact/X-ray/advice mappings and missing-roll candidate handling.
- `src/pme/sync.py`: reproducible dashboard synchronization command.
- `frontend/app.py`: integrates the existing local dashboard with tracked source,
  recognition display and a complete field viewer.
- `pyproject.toml`, `uv.lock`: SQLAlchemy dependency and Transformers 4.x constraint
  for the implemented model interface; refreshed lock metadata.
- `Makefile`: replace references to absent Docker files with available uv commands;
  remove a broad cleanup target that would erase persistent recognition data.
- `tests/test_doctor_recognition.py`, `tests/test_dashboard_integration.py`:
  CPU-only matching, persistence, ambiguity/conflict, OCR consistency and database
  integration regression tests.
- `README.md`, this document: setup and operational instructions.

## Verification and limits

Implemented on `sign-codex` after fetching/pulling `main` (already up to date).
The pre-existing deleted `pme_database_layer/` files have been left untouched;
its deletions are not part of the feature commit. The existing untracked frontend
and database implementation were integrated into the feature's source files.

Twenty CPU-only tests passed using synthetic images, temporary SQLite databases
and fake VLM responses. Tests also cover annotated-box coordinates, OCR failure,
overlap retention, additive schema upgrades, stable IDs and rollback on identity
conflicts. A synthetic annotation was visually inspected. Python compilation and
whitespace checks passed.
Dependency metadata was resolved with `uv lock`; packages/models were not installed
or loaded for extraction. No extraction, training, real-report processing or GPU
operations were executed in this environment.

Unverified: actual doctor/candidate localization accuracy, stamp OCR accuracy,
handwritten signature variation across real reports, false-positive/false-negative
rates, threshold calibration, runtime, GPU memory needs, CUDA driver compatibility,
and a live Streamlit browser session. The visual matcher is an untrained conservative
baseline, not a validated biometric identification model. Different pen strokes,
short signatures, scan skew/noise and stamps overlapping handwriting can produce
review/unmatched results even for a known doctor. Initial automated enrollment can
still inherit OCR/localization errors despite consistency checks. Validate gallery
entries and measure matching thresholds on representative labeled held-out reports
on the GPU system before treating automated names as authoritative.


## Database upgrade for bounding-box evidence

Run the same sync command after updating the code:

```bash
uv run python -m src.pme.sync
```

Startup/sync creates three additional tables if absent; existing tables and
candidate rows are not dropped or altered:

- `doctor_registry_links`: unique persistent gallery ID linked to a unique
  dashboard doctor row. Display-name equality does not merge different gallery IDs.
- `doctor_recognition_results`: one result per extraction run, with status,
  similarity constrained to 0..1, algorithm, observed name, notes and optional
  registered-doctor link.
- `doctor_evidence`: per-page boxes (normalized and pixels), image dimensions,
  crop/annotation paths, quality flags, stamp text and notes. A unique constraint
  permits one evidence row per extraction run/page; page indices must be 2 or 3.

This is an additive schema upgrade implemented by SQLAlchemy metadata creation,
not an alteration of old columns. Sync backfills recognition results from existing
record JSON, even when the candidate record itself has not changed. Older records
without the new `evidence` list remain readable. Run extraction with `--overwrite`
on your GPU machine to generate annotations/crops for those reports, then sync.
The pipeline also attempts doctor evidence extraction when full-document field
extraction fails, preserving a null-shaped field record and a failure exit code.

New evidence is saved independently of identity recognition. Conflicting doctor
names, multiple doctors, unusable signatures and OCR errors therefore retain the
localized page images and bounding boxes for review instead of discarding them.
