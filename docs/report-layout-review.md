# Visual review of the supplied RRB scan

Source: `RRB SCAN 16092026.pdf`, 200 pages, 50 consecutive four-page reports.
SHA256: `4d31bfadfa1fe73641a465ad7b22998e3d2aaaed57a1b76f26ce2701386cf6e5`.

Review scope: rendered doctor sections for all 100 report-page-3/page-4 images;
full-page visual inspection of PDF pages 3, 4, 39, 40, 107 and 108. PDF rendering
was used for inspection only. No VLM extraction, model loading or training was run.
The local visual reference montage uses manually chosen boxes, not automatic
model predictions. Source medical images are not added to the repository.

## Layout rules added to the localizer

- Report page 3: examiner block near lower centre/right, typically left of the
  candidate photograph. Relevant anchors are `Initials of Railway Medical Examiner`
  and `Office Seal`. The candidate signature/thumbprint sits separately at lower left.
- Report page 4: examiner block lower right, below the candidate signature/thumbprint
  section. The anchor is `Signature of the Medical Examiner`.
- The enclosing doctor box must include the complete handwritten signature, both
  language name lines and designation/location line. Long pen strokes may cross
  text and reach above the examiner label. Stamp text may extend past the form's
  printed right border; the form border must not truncate the doctor's surname.
- Keep overlap in original evidence. Isolate handwriting only in the separate
  matching asset; do not treat stamped letters as signature features.
- A faint/absent signature on one page is not a reason to reject a readable pair
  on the other. Do not invent an absent handwritten signature from the stamp name.

The detector now returns an explicit `doctor_region` enclosing signature and stamp,
plus separate `signature` and `stamp` boxes. A valid region must contain the validated
parts. Otherwise the saved combined crop uses their union with a small configurable
margin (`combined_padding: 0.008`). Invalid oversized enclosing boxes are rejected;
independent crops are preserved even when no safe combined region can be formed.
A combined region can also be retained for review if separate parts were not found.
No fixed screenshot coordinates or doctor-name-to-signature shortcuts are used.

## A useful three-doctor trial

When the manifest contains **only this PDF in its original page order**:

| Candidate position | Report pages | Doctor name visible in stamp |
| --- | --- | --- |
| 1 | PDF pages 3 and 4 | Dr. Darshansinh Solanki |
| 10 | PDF pages 39 and 40 | Dr. Parag Adhyapak |
| 27 | PDF pages 107 and 108 | Dr. A. Robin |

The first four candidates in this PDF all show Dr. Darshansinh Solanki's stamp,
so the first-four trial tests reuse of one doctor rather than three-doctor discrimination.
These are visual observations, not names hardcoded into OCR or gallery enrollment.

To use this trial, set `configs/pdf_sources.yaml` to only the supplied file:

```yaml
pdfs:
  - RRB SCAN 16092026.pdf
```

Ensure that file exists under `data/raw/`, then on the GPU machine:

```bash
uv run python -m src.run_pipeline --phase all
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --extra vlm python -m src.extraction.run_extraction --candidates 1 10 27 --overwrite
uv run python -m src.database.run_database --selection data/processed/extractions/last_selection.json
uv run streamlit run frontend/app.py
```

This prepares the PDF and extracts only the three selected candidates. Once those
records exist, `--candidates 1 10 27 --doctor-only` refreshes their doctor evidence
without repeating candidate field extraction. `--candidates` uses one-based manifest
positions; it cannot be combined with `--ndocs` or `--doc`. Positions will differ if
other PDFs precede this file in the merged source. Existing registry files should
be retained across trials. Manually inspected reference boxes do not establish
model localization or recognition accuracy; verify actual saved crops on the GPU.
