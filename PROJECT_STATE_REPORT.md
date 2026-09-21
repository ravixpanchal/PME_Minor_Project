# PROJECT_STATE_REPORT.md
**Project:** `pme_analysis` — Medical Examination Report Digitization & Query System
**Date:** 2026-09-07 · **Branch:** `nextStage` · **Auditor:** Bionic (senior-engineer audit + stress test + surgical repair)
**Scope:** Full codebase audit, stress testing of every pipeline component, minimal bug fixes, onboarding documentation.

---

## 0. Read-This-First (TL;DR)

| Question | Answer |
|---|---|
| What is this repo? | A 15-phase pipeline that converts multi-page scanned medical exam PDFs (typed + handwritten + Hindi, photos/stamps/signatures to discard) into structured, identity-stable, queryable records. Local-only (2× RTX 4500 Ada), PII never committed. |
| Where are we? | **Phases 1–4 complete & re-verified. Phase 5 code-complete on branch `nextStage` (uncommitted) — first real 7B run DONE (6 docs, v1 schema); extraction contract then upgraded to hierarchical schema v2 (page-structured, nested objects/arrays). Phase 6 (detection) prototype ready. Phases 7–15 not started.** |
| Does it work? | Yes — full `--phase all` run: 48/48 pages in ~20 s, IDs byte-stable, contract gate green, 18/18 unit tests pass. |
| What was broken and fixed? | **3 real bugs found & fixed** (a `SyntaxError` that made the entire Phase 5 extraction runner unimportable; a holdout-leak self-detection false positive; **a missing `torchvision` dependency that blocked the first real extraction run**) + 1 misleading error-message typo. Details in §6. |
| What is the next action? | (1) Commit `nextStage` work. (2) Hand-type ground truth (`scripts/make_gt_template.py`). (3) First real VLM run once ~16 GB of VRAM is free on one GPU (weights download already in progress). Details in §8. |

---

## 1. Progress Summary

### 1.1 What the system does

Railway Recruitment Board periodic medical examination reports arrive as multi-page scanned PDFs, often several people per file. The system:

1. **Splits** each source PDF into per-person documents using a human-confirmed page map (never a guess).
2. **Renders** every page at *adaptive DPI* — `min(source effective DPI, 300 cap)` — so scans are never upscaled and never blown up.
3. **Cleans** each page (blank check → deskew → dark-border trim → CLAHE → bilateral denoise) into a separate immutable-vs-cleaned image pair.
4. **Generates** a 1280 px-long-edge VLM copy per page (the "two-resolution rule" — VRAM governance).
5. **Extracts** (Phase 5, code-complete) schema-driven full-page fields via Qwen2.5-VL, merging per-page results into one validated per-document record with provenance and conflict flags.
6. **Scores** (Gate 1 harness, code-complete) extractions against hand-typed ground truth with a 5-class error taxonomy (correct / wrong / missed / hallucinated / correctly_nulled).

Core invariants (enforced in code, not documentation):

- **Content-hash identity:** `document_id = sha256(f"{source_sha}:person:{i}")`. Re-runs, machine moves, and preprocessing changes can never change an ID. *Re-verified this audit: byte-identical across a full re-run.*
- **Fail-fast validation:** every config and every manifest row is Pydantic-validated (`extra="forbid"`) before anything is written.
- **Raw is immutable:** `data/interim/high_res/` is never modified; cleaned output goes to `high_res_clean/`, enabling before/after diffing.
- **Never crash the batch:** per-page failures are logged, skipped, and exit codes propagate; downstream gates flag the gap.
- **Leakage guard:** held-out ground-truth documents are blacklisted in `configs/holdout_ids.txt`; `scripts/check_holdout_leakage.py` greps all tracked code/config for those IDs.

### 1.2 Phase status (15-phase master roadmap, `literature/medical_report_extraction_roadmap.md`)

| Phase | Name | Status | Evidence |
|---|---|---|---|
| 1 | Scaffolding, Docker, GPU | ✅ Complete | repo, `uv`, Docker, configs, `verify_gpu.py` PASS |
| 2 | Schema & region classes | ✅ Complete | `schema.yaml` (4 top-level + 2 table fields, now with `extraction_hint`), `regions.yaml` (4 extract + 4 discard), Pydantic-validated |
| 3 | Document assembly | ✅ Complete | 48 pages → 12 person-docs, adaptive-DPI renders, validated `manifest.parquet`, `splits_applied.json` audit trail |
| 4 | Preprocessing & dual-res | ✅ Complete | 48/48 cleaned + VLM copies, manifest upgraded, benchmark 0.30 s/page vs 2.0 s spec (MEETS) |
| 5 | Baseline VLM extraction (GATE 1) | 🟡 **Code-complete, first real run pending** | prompt/parser/runner/merge/provenance all tested deterministically; real 7B inference blocked by VRAM guardrail + missing cached weights (see §6.4 T13) |
| 6 | Region detection (prototype) | 🟡 Code-complete, needs GPU run | detector + batch runner + visualization written; coordinate-space handling (28 px patch alignment) unit-verified |
| 7–15 | Region extraction → entity resolution → GT gate → DB → review UI → Text-to-SQL → RAG → fine-tuning | ⬜ Not started | — |

### 1.3 Work completed in each prior stage (from `data/stages/`)

- **Stage 1 (`stage1(5sept).md`, `stage1(5sept_2).md`)** — Phases 1–2: repository, Docker/GPU setup, `uv` lockfile, config-driven architecture (singleton loader, Pydantic fail-fast), schema + region definitions, contract-test skeleton.
- **Stage 2 (`stage2(5sept).md`)** — Phase 3: person splitting with hard-refusal of unconfirmed maps, SHA-256 identity, adaptive-DPI renderer with source-quality estimation, manifest builder, 5 pytest tests, detection-prototype scaffold.
- **Stage 3 (`stage3(6sept).md`)** — Phase 4: `cleaner.py` (deskew window 0.3°–5.0°, sign self-check, dark-band trim with 8 px safety margin, CLAHE, bilateral), `run_preprocessing.py` (per-row `ManifestRow` validation before write, per-page failure isolation, spec benchmark), `vlm_resizer.py`, `run_pipeline.py` chaining, detection batch runner, 14/14 tests, 3 documented bug fixes (corrupt cv2 install, import name, OpenCV 5 `HoughLinesP` shape).
- **Stage 4 (current, uncommitted on `nextStage`)** — Phase 5 + Gate 1 harness:
  - `src/extraction/vlm_extractor.py` — schema-driven prompt (now includes per-field `extraction_hint` from `schema.yaml`), 28-multiple image pinning, tolerant JSON parser.
  - `src/extraction/run_extraction.py` — per-page → per-document merge with first-non-null-wins conflict flagging, provenance, idempotent outputs to `data/processed/extractions/`.
  - `src/ground_truth/normalize.py` + `scorer.py` — canonicalization (dates, integers, employee-ID case) + Gate 1 accuracy tables by source-type and by-field with the 5-class taxonomy and a PASS/FAIL verdict against the 90 % typed-field threshold.
  - `scripts/make_gt_template.py`, `scripts/check_holdout_leakage.py`, `configs/holdout_ids.txt` — Phase 9 prep + leakage guard.
  - `src/detection/run_detection.py` — full batch detection runner (was an empty stub).
  - `tests/test_extraction_baseline.py`, `tests/test_cleaner.py` — 8 new tests (all pass).

### 1.4 What THIS audit did (2026-09-07)

1. Read all documentation (`data/stages/*`, `literature/*`, configs) and inspected the uncommitted `nextStage` work via git diff.
2. Stress-tested **every** pipeline component and CLI tool (see §6 test log T1–T13), on CPU with ≤ 2-page subsets wherever an engine was involved, to keep both VRAM (reserved for local inference) and CPU time bounded.
3. Found and surgically fixed **2 bugs + 1 misleading error message** (§6.5). No refactors, no style changes.
4. Re-verified the full gate chain after fixes: 18/18 pytest, 30/30 module imports, contract gate green on 48 rows, all configs valid.
5. **Follow-up (2026-09-08):** the user's first real `run_extraction --ndocs 1` attempt crashed in `AutoProcessor.from_pretrained` — `Qwen2VLVideoProcessor` requires `torchvision`, which the `vlm` extra never declared. Installed `torchvision 0.29.0+cu130` (exact pair for the installed torch), added it to the `vlm` extra in `pyproject.toml`, and re-verified the previously failing processor-load line on CPU with zero VRAM (T17, §6.1).
6. **Change (2026-09-12):** extraction is now **single-pass per person** by default — all 4 pages attached in order in ONE `Qwen2.5-VL` call → ONE consolidated `record.json` per person (no more 4 near-empty `page_XXXX.json` files). Implemented as `--mode document` (default) vs `--mode per_page` (the old 4-calls + first-non-null-merge flow, kept as A/B fallback). Also fixed two latent extraction bugs found while tracing this path: literal `"null"/"None"/"N/A"` strings leaking into records as values (B5), and the `max_new_tokens=800` generation budget being smaller than a full v2 page-4 JSON → guaranteed truncation/parse-failure (B6). All verified CPU-side (23/23 pytest incl. 4 new, multi-image processor path exercised with person 1's real 4 images, zero VRAM) — T19–T21, §6.1; the first real single-pass run is pending free VRAM (~17 GB on one GPU).

---

## 2. Architecture at a Glance

```
data/raw/*.pdf (PII, gitignored)
   │
   ▼  PHASE 3 ─ assembly/ (splitter → render → manifest_builder)
data/interim/high_res/{doc_id}/page_NNNN.png      [IMMUTABLE raw renders, adaptive ≤300 DPI]
data/metadata/manifest.parquet                     [Phase-3 columns, Pydantic-gated]
   │
   ▼  PHASE 4 ─ preprocessing/ (cleaner → vlm_resizer → run_preprocessing)
data/interim/high_res_clean/{doc_id}/page_NNNN.png [cleaned; the review/detection input]
data/interim/vlm_res/{doc_id}/page_NNNN.png        [1280 px long edge; the VLM input]
data/metadata/manifest.parquet                     [upgraded: high_res_path, vlm_res_path, vlm_res_width, blank_page_flag]
   │
   ▼  PHASE 5 ─ extraction/ (vlm_extractor → run_extraction)        [code-complete]
  [default --mode document: ONE model call over all 4 pages in order → ONE JSON per person]
data/processed/extractions/{doc}/document.json     [sidecar: mode, person_index, model_input order, parse_ok, fields]
data/processed/extractions/{doc}/record.json       [ExtractionRecord: fields, provenance (section page), pages_covered]
  [--mode per_page A/B: page_NNNN.json per page + merged record.json (first-non-null-wins, conflicts)]
   │
   ▼  PHASE 6 ─ detection/ (vlm_detector → run_detection → visualize) [prototype, ready]
data/processed/regions/{doc}/page_NNNN.json        [normalized boxes]
data/processed/regions/{doc}/vis/page_NNNN.png     [human-review overlays on full-res raw]
   │
   ▼  GATE 1 ─ ground_truth/ (normalize + scorer)                   [code-complete]
data/ground_truth/gt/{doc}.json                    [hand-typed GT — dev set]
data/ground_truth/eval_holdout/{doc}.json          [held-out — score sparingly]
data/processed/eval/gate1_report[_tag].json        [per-value taxonomy report]
```

**Entrypoints**

| Command | What it does |
|---|---|
| `python src/run_pipeline.py --phase all` | Assembly → preprocessing (default phase) |
| `python src/run_pipeline.py --phase assembly` | Phase 3 only |
| `python src/run_pipeline.py --phase preprocess [--limit N]` | Phase 4 only (`--limit` = dry-run, manifest untouched) |
| `python -m src.extraction.run_extraction --ndocs N [--mode document\|per_page] [--source vlm\|clean\|raw] [--overwrite]` | Phase 5 (needs GPU + model; default = single pass, one JSON per person) |
| `python -m src.detection.run_detection --limit N [--source …] [--overwrite]` | Phase 6 (needs GPU + model) |
| `python -m src.ground_truth.scorer [--include-holdout] [--tag X]` | Gate 1 scoring |
| `python src/test_contracts.py` | Pixel-dimension gate over the manifest |
| `python src/validate_schema.py` | Config contract gate |
| `python scripts/check_holdout_leakage.py` | Evaluation-leakage pre-flight |

---

## 3. File & Module Directory

> Every non-generated file in the repository. "Inputs → Outputs" is the data contract. `PII` = handles patient data.

### 3.1 Repository root

| File | Role | Inputs → Outputs | Notes |
|---|---|---|---|
| `README.md` | Title placeholder (`# pme_analysis`) | — | **Debt:** needs a real README (overview, quickstart, architecture) |
| `pyproject.toml` | Package metadata, deps, entrypoints | — | 10 runtime deps; `vlm` extra (torch/**torchvision**/transformers/accelerate — torchvision added 2026-09-08, see B4); dev group (pytest, pytest-cov); `run-pipeline` script; hatchling build of `src/` |
| `uv.lock` | Deterministic lockfile | — | ~55+ packages, platform-pinned, SHA-256 hashed |
| `.python-version` | `3.11` pin for `uv` | — | |
| `Makefile` | Docker shortcuts: `build`, `shell`, `test`, `verify-gpu`, `gpu CMD=…`, `clean` | — | `gpu` uses the compose GPU override |
| `.gitignore` | **Data-leakage guard** | — | Blocks all PDF/PNG/parquet/JSON/CSV + all `data/*` subtrees (keeps `.gitkeep`); blocks `.env`, `.venv`, caches |
| `.dockerignore` | Build-context hygiene | — | Excludes `.git`, `.venv`, `.env`, `data/`, caches, `docs/` |
| `.env.example` | Env template (paths, GPU, log level, Label Studio) | — | Copy to `.env` |
| `PROJECT_STATE_REPORT.md` | **This document** | — | |

### 3.2 `configs/` — the "brain" (all validated on import)

| File | Role | Key contents |
|---|---|---|
| `pipeline.yaml` | Master pipeline config | 7 paths (raw/high_res/high_res_clean/vlm_res/metadata/manifest/processed); `render_dpi: 300` (CAP); `vlm_long_edge_px: 1280`; preprocessing chain (deskew, crop, CLAHE, bilateral d=9 σc=75 σs=75); `low_source_dpi_threshold: 150` |
| `schema.yaml` | **Extraction contract — v2 (2026-09-08): hierarchical, page-structured.** 4 top-level sections (`page_1_basic_info` … `page_4_declarations`), each `page:`-tagged so each page's prompt asks only for its own fields; nested `object` nodes for printed boxes (e.g. `accuracy_of_vision.right_eye.distance_uncorrected`) and `array` nodes for row-wise tables (e.g. `part_one[i].answer`, boolean); leaves stay `string/integer/float/date/boolean` with `printed_row_label` + `extraction_hint`. v1 flat baseline preserved at `schema.baseline.yaml` |
| `regions.yaml` | Detection class contract | extract: typed_text, handwritten_text, data_table_typed, data_table_handwritten · discard: photo, thumbprint, signature, stamp_seal |
| `models.yaml` | Model registry (Group 2 owns) | `default_vlm: Qwen/Qwen2.5-VL-7B-Instruct` (loaded as raw dict — no Pydantic validation by design) |
| `splits.yaml` | Person split map for `black_dataset.pdf` | 12 confirmed persons × 4 pages; **`confirmed: true`** — assembly hard-refuses unconfirmed maps; contains page indices only (no PII, safe to commit) |
| `holdout_ids.txt` | Evaluation-leakage blacklist | **Currently empty** — add 2 held-out `document_id`s once GT is split; scanned by `check_holdout_leakage.py` (self-excluded by the fix in §6.5-B) |

### 3.3 `src/` — root modules

| File | Role | Inputs → Outputs |
|---|---|---|
| `src/__init__.py` | Package marker | — |
| **`run_pipeline.py`** | **Main CLI entrypoint.** `--phase assembly\|preprocess\|all` (default `all`), `--dummy` smoke test, `--limit N` (preprocess dry-run). Chains assembly → preprocessing; assembly failure aborts before preprocessing; exit codes propagate via `SystemExit`. | manifest + raw PDFs → full Phase 3+4 state |
| **`test_contracts.py`** | **Pixel-dimension gate.** Per row: raw render width vs `(page_pts/72) × render_dpi_used` (±300 px, per-row adaptive DPI with legacy fallback), cleaned plausibility (≥60 % of expected), VLM long edge ±10 % of 1280 px. Skips cleanly pre-Phase-3. | manifest + PNGs → exit 0/1 |
| **`validate_schema.py`** | Config gate: pipeline/regions/schema vs Pydantic; reports models.yaml as raw. | YAMLs → exit 0/1 |
| **`verify_gpu.py`** | GPU passthrough check: `nvidia-smi` reachability; torch CUDA only if installed. | nvidia-smi → PASS/FAIL print |
| `models/schemas.py` | **All Pydantic contracts.** `RegionClassConfig`, `FieldDefinition` (+`extraction_hint`), `TableFieldDefinition`, `SchemaConfig`; manifest rows: `AssemblyRow` (12 Phase-3 fields, SHA-256 + ISO-8601 validators) → `ManifestRow` (+`high_res_path`, `vlm_res_path`, `vlm_res_width`, `blank_page_flag`); **new this stage:** `ProvenanceEntry`, `ExtractionRecord` (document-level merge contract), `GroundTruthDoc` (GT file contract, `?` = unreadable marker) | dicts → validated models |
| `utils/config_loader.py` | **Singleton** config loader; fail-fast Pydantic validation on first import of *any* `src` module; `get_path(key)` → absolute path; `get_preprocessing_param("a.b")`; `bootstrap_dirs()`; exposes `root_dir` | `configs/*.yaml` → `config` singleton |
| `utils/logger.py` | Named loggers, `LOG_LEVEL` env, single stdout handler | — |

### 3.4 `src/assembly/` — Phase 3 (document assembly)

| File | Role | Inputs → Outputs |
|---|---|---|
| `__init__.py` | Package marker | — |
| `id_generator.py` | `file_sha256()` (1 MB-chunk streaming) and `person_document_id(source_sha, i) = sha256(f"{sha}:person:{i}")` — content-only identity, stable across machines/re-runs | PDF bytes / (sha, index) → 64-char hex IDs |
| `splitter.py` | `SplitMap` validates the person map: bounds, **exact page coverage (no gaps/overlaps)**, ascending order; `split_status()` = `manually_cleared` vs `flagged`; `load_split_map()` reads `splits.yaml`, computes source SHA, checks page count. Refuses unconfirmed maps at the caller | `splits.yaml` + source PDF → validated map |
| `render.py` | Adaptive-DPI rasterizer. `effective_dpi()` = px/inch of the **dominant** embedded raster (largest displayed area; rasters covering <50 % of the page — e.g. logos — can't pull vector pages down). `render_dpi_for_page() = min(effective, cap)`. `render_person_pages()` writes `page_NNNN.png` + per-page quality records. **Scans are never upscaled; vector pages render at the cap.** | PDF + range → PNGs + `{effective_dpi_estimate, render_dpi_used, low_source_quality_flag}` |
| `manifest_builder.py` | `write_manifest(rows)` — every row Pydantic-validated (`AssemblyRow`) **before** parquet write | row dicts → `manifest.parquet` |
| `run_assembly.py` | **Phase 3 orchestrator:** bootstrap dirs → load+confirm split map → per person: ID → render → row build → write manifest → `splits_applied.json` audit trail. Returns 0/1. | raw PDF + `splits.yaml` → `high_res/` + manifest Phase-3 columns |

### 3.5 `src/preprocessing/` — Phase 4 (cleaning + dual resolution)

| File | Role | Inputs → Outputs |
|---|---|---|
| `__init__.py` | Package marker | — |
| `cleaner.py` | **Per-page cleaning chain; never raises.** (1) blank check (ink <0.5 % → returned untouched, flagged) → (2) deskew: median Hough-line angle, corrected only in 0.3°–5.0° window with sign self-check (flip if residual worsens); >5° → `skew_suspicious` (orientation problem, human review, **never auto-rotated on a guess**) → (3) dark-border trim: only fully-dark bands (row/col ≥60 % dark) with 8 px safety margin; deliberately does NOT tight-crop (margins hold signatures/stamps/thumbprints) → (4) CLAHE (LAB L-channel, clip 2.0, 8×8) → (5) bilateral denoise (params from `pipeline.yaml`). Flags: `blank, deskewed, skew_deg, borders_trimmed, skew_suspicious, ink_fraction`. | BGR ndarray → (cleaned BGR, flags) |
| `vlm_resizer.py` | `make_vlm_copy()` — long edge → `vlm_long_edge_px` (1280), aspect preserved, LANCZOS; returns final width for the manifest | cleaned PNG → `vlm_res` PNG + int width |
| `run_preprocessing.py` | **Phase 4 orchestrator.** Reads Phase-3 manifest; per page: read raw → `clean_page` → write `high_res_clean` → `make_vlm_copy` → **`ManifestRow` validates before manifest write** → accumulate. Per-page try/except (failures logged+skipped; row keeps Phase-3 state so the contract gate flags it — *visible, not silent*). `--limit N` = dry-run (**manifest not written** — re-verified this audit). Prints the spec benchmark (mean/max s/page vs 2.0 s) with MEETS/EXCEEDS verdict. Exit 0 only if zero failures. | manifest + `high_res/` → `high_res_clean/` + `vlm_res/` + upgraded manifest |

### 3.6 `src/extraction/` — Phase 5 (baseline VLM extraction, GATE 1) — *new this stage*

| File | Role | Inputs → Outputs |
|---|---|---|
| `__init__.py` | Package marker | — |
| `vlm_extractor.py` | **Zero-shot, schema-driven extraction.** `build_extraction_prompt()` — pure function, lists every schema field with its `extraction_hint`; rules: transcribe exactly, null-if-unsure, never guess, row-label anchoring, JSON-only output. **Three prompt modes (2026-09-12):** `page=N` (that page's sections only), `n_pages=N` (DOCUMENT mode — all sections, all 4 pages attached in order), legacy single-page. `_pin_size()` — resizes to long-edge target with dims as exact multiples of 28 (Qwen2.5-VL patch 14 × merge 2) so the processor's smart_resize is a no-op. `parse_extraction_output()` — tolerant (strips code fences, extracts first JSON object, filters to schema keys, coerces bare numbers + null-ish literals, trims whitespace). `VlmSchemaExtractor` — lazy torch/transformers import; bf16 on `cuda:0`; greedy; `_ask()` now takes an IMAGE LIST (1 page or all 4) with `min/max_pixels` bracketing the per-image areas so nothing is resampled; `max_new_tokens` 2048 (per-page) / 4096 (document). `extract_page(path, long_edge, page)` and **`extract_document([paths], long_edge)`** (single pass, one full-schema JSON). | image path(s) + long edge → `(shaped fields dict, parse_ok)` |
| `run_extraction.py` | **Phase 5 runner.** Args: `--ndocs N`, `--doc id`, `--mode document\|per_page` (**default `document`**, 2026-09-12), `--source vlm|clean|raw` (default `vlm`; `raw` = A/B "is preprocessing helping or hurting?"), `--long-edge`, `--overwrite`, skips blank pages. **document mode:** person→document mapping logged (`doc ← person N · 4 pages in order`), ONE `extract_document` call → `record.json` (ExtractionRecord: fields, provenance credited to each section's page, `pages_covered`, `conflicts=[]`) + `document.json` sidecar (mode, person_index, model_input order, parse_ok, fields) — **no per-page files**. **per_page mode (A/B fallback, original flow):** per page extract → `page_NNNN.json` → merge first-non-null-wins (later differing values → `conflicts`). try/except per doc (never crash the batch); exit 0 only if zero failures; idempotent (skips docs with existing `record.json` unless `--overwrite`). | manifest + `vlm_res/` (or clean/raw) → `data/processed/extractions/{doc}/` (document mode: `document.json` + `record.json`; per_page mode: `page_*.json` + `record.json`) |

### 3.7 `src/detection/` — Phase 6 (region detection prototype)

| File | Role | Inputs → Outputs |
|---|---|---|
| `__init__.py` | Package marker | — |
| `vlm_detector.py` | Qwen2.5-VL box detection with **exact coordinate mapping** (same 28-px pinning; `min_pixels=max_pixels=area` ⇒ scale-back exact). `Detection(region_class, box_norm[0..1], content_hint)`. Class prompt from `regions.yaml` (discard classes boxed to be excluded; tables boxed whole; no overlaps). `_parse_boxes()` static/tolerant. Lazy torch import. | image path → `[Detection]` (normalized boxes) |
| `run_detection.py` | **Batch runner** (was an empty stub; full implementation this stage). Args: `--limit`, `--doc`, `--source vlm|clean|raw`, `--overwrite`, `--include-blank`. Blank pages skipped by default; idempotent; writes `data/processed/regions/{doc}/page_NNNN.json` (normalized boxes + model + source) and **visualization always drawn on the full-res raw render** (normalized boxes are resolution-independent) to `vis/page_NNNN.png`. | manifest + images → region JSONs + review PNGs |
| `visualize.py` | `draw_boxes()` — colored rectangles per region class (blue typed / orange handwritten / purple tables / red discard), class+hint labels, review copy capped at 1600 px. | raw render + detections → review PNG |
| `pose_splits.py` | VLM-assisted **person-boundary proposal** (renders pages at 100 DPI, per-page "new report?" verdict, writes `configs/splits_draft.yaml` — **never auto-applied**; human must confirm). | source PDF → `splits_draft.yaml` |

### 3.8 `src/ground_truth/` — Gate 1 harness (Phase 9 prep) — *new this stage*

| File | Role | Inputs → Outputs |
|---|---|---|
| `__init__.py` | Package marker | — |
| `normalize.py` | Value canonicalization (scorer + future DB load): whitespace collapse; `?` = unreadable GT marker (kept, excluded from scoring); `employee_id` uppercased; dates → ISO (dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, dd/mm/yy, yyyy-mm-dd); integer/float → first numeric token. **The model transcribes; we canonicalize.** | raw value + data_type → canonical string |
| `scorer.py` | **Gate 1 harness.** Loads GT from `data/ground_truth/gt/` (+`eval_holdout/` with `--include-holdout`), matches against `extractions/{doc}/record.json`, scores per value into the 5-class taxonomy (correct/wrong/missed/hallucinated/correctly_nulled; GT `?` excluded), aggregates by source-type and by-field, prints accuracy tables + mismatch list (top 30) to iterate the prompt against, writes `data/processed/eval/gate1_report[_tag].json`. Verdict: typed ≥90 % → PASS (warns if sample <20 values). | GT JSONs + records → console tables + report JSON |

### 3.9 `tests/` — pytest suite (23 tests, all passing)

| File | Tests | Covers |
|---|---|---|
| `test_manifest_stability.py` | 5 | config contracts, path absoluteness, schema/region contents, ID determinism |
| `test_adaptive_rendering.py` | 7 | synthetic-PDF regressions: digital page at cap, logo-page not pulled down, 120-DPI scan not upscaled, 600-DPI scan capped, pixel math + flags, `ManifestRow` requires `render_dpi_used` |
| `test_cleaner.py` | 2 | text page survives cleaning unflagged-blank; blank page flagged + returned byte-identical |
| `test_extraction_baseline.py` | 4 | prompt is schema-driven (incl. hints + row labels + "NEVER guess"); parser tolerance + schema filtering; normalizer (dates, integers, ID case, `?`); scorer taxonomy (all 5 classes) |

### 3.10 `scripts/` — developer utilities

| File | Role |
|---|---|
| `contact_sheet.py` | Numbered 60-DPI thumbnail grids (8×6) → `data/metadata/contact_sheets/` for marking person boundaries by eye (PII — gitignored) |
| `make_splits.py` | Builds `configs/splits.yaml` from `--starts` pages; validates strictly-increasing starts; trims `--ignore`d edge pages; warns outside the 4–6 pages/person prior; writes **`confirmed: false`** (spot-check before enabling) |
| `make_gt_template.py` | *(new this stage)* Scaffolds `data/ground_truth/gt/{doc}.json` with all schema fields null for hand-typing; enforces the "type BEFORE looking at record.json — no anchoring" workflow |
| `check_holdout_leakage.py` | *(new this stage)* Pre-flight: no holdout `document_id` may appear in any tracked file under `configs/`, `src/`, `scripts/`, `tests/`. Exit 0 clean / 1 leak-or-not-set. **Fixed this audit:** now excludes its own ID file (see §6.5-B) |

### 3.11 `docker/` — containerization

| File | Role |
|---|---|
| `Dockerfile` | `nvidia/cuda:12.2.0-base-ubuntu22.04` + Python 3.11 + `uv` (from ghcr); venv at `/opt/venv` (**outside `/app`** so the dev volume mount never shadows it); layer-cached deps; entrypoint `uv run` |
| `docker-compose.yml` | `foundation` service (env, `..:/app` mount, 16 GB shm) + `label-studio` (port 8080, GT + read-only high-res volumes) |
| `docker-compose.gpu.yml` | GPU override (`driver: nvidia, count: all`) for Phase 5+; used by `make verify-gpu` / `make gpu` |

### 3.12 `data/` — data tree (PII — all gitignored except structure)

| Directory | Purpose | Current state |
|---|---|---|
| `data/raw/` | Input PDFs | `black_dataset.pdf` (48 pp, in `splits.yaml`), `colored_data.pdf` (54 pp, **not yet split**) |
| `data/interim/high_res/` | **Immutable** raw renders | 12 docs × 4 pages = 48 PNGs @ 200 DPI (source native) |
| `data/interim/high_res_clean/` | Cleaned images (consumption path) | 48 PNGs |
| `data/interim/vlm_res/` | VLM copies (1280 px long edge) | 48 PNGs (widths 886–961 px) |
| `data/metadata/` | Ledger + audit | `manifest.parquet` (48 rows, 16 cols, Phase-3+4), `splits_applied.json`, `contact_sheets/` |
| `data/processed/` | Phase 5+ outputs | `extractions/` (empty until first VLM run), `regions/` (empty), plus one hand-typed reference note `242243220965001_PRAKASH CHANDRA MURMU.md` (person 10 — useful as GT seed) |
| `data/ground_truth/` | GT | `eval_holdout/` (empty); `gt/` created on demand by `make_gt_template.py` |
| `data/stages/` | Stage reports | 4 documents (the project's own history — read them in order) |

### 3.13 `literature/` — design docs (read these before changing architecture)

| File | Role |
|---|---|
| `medical_report_extraction_roadmap.md` | Master 15-phase roadmap: problem, constraints (local-only, 2×24 GB VRAM, mixed Hindi handwriting), per-phase specs, Gate 1/2 criteria, leakage guard, entity-resolution policy |
| `group1_foundation_architecture.md` | Group 1 spec (Phases 1–4): work packages, v1.0→v1.1 changelog (7 design corrections), the Group-1↔Group-2 interface contract, Gate 1 exit checklist, non-goals |

> **Gap:** `docs/group1_contract.md` is referenced by stage docs and module docstrings but **does not exist** (empty `docs/` dir). The contract content currently lives in `literature/group1_foundation_architecture.md` §4 — promote it to `docs/` or fix the references (debt item, §8).

---

## 4. Data Contracts (what each artifact must contain)

**Manifest row (48 rows today, 16 columns)**

| Column | Phase | Contract |
|---|---|---|
| `document_id` | 3 | 64-char lowercase SHA-256, `sha256(f"{source_sha}:person:{i}")` |
| `page_index` / `source_page_number` / `person_index` | 3 | ≥0 / ≥1 / ≥0 |
| `source_file_sha256` | 3 | 64-char hex |
| `high_res_raw_path` | 3 | relative to project root |
| `file_mtime` | 3 | ISO-8601 (metadata only — **never** part of identity) |
| `raw_page_width_pts` | 3 | float |
| `effective_dpi_estimate` / `render_dpi_used` | 3 | ≥0; used = min(estimate, 300) |
| `low_source_quality_flag` | 3 | estimate < 150 |
| `split_check_status` | 3 | enum: `manually_cleared` \| `flagged` \| `deferred_to_phase7` \| `pending_inspection` |
| `high_res_path` | 4 | cleaned image (consumption path) |
| `vlm_res_path` / `vlm_res_width` | 4 | 1280 px long edge / actual width |
| `blank_page_flag` | 4 | near-blank (duplex backsides) — 0 of 48 today |

**`record.json` (Phase 5 output)** — `ExtractionRecord`: `document_id`, `model`, `fields: {name: str|null}`, `provenance: {name: {page_index, model, extracted_at, raw_value}}`, `conflicts: [names]` (first-non-null wins; conflicts flagged for review), `pages_covered: [indices]`.

**GT file** — `GroundTruthDoc`: `document_id`, `annotator`, `fields: {name: str|null|'? '}`, `notes`. `null` = genuinely absent; `'?'` = unreadable (excluded from scoring).

---

## 5. Environment & How to Run

**Host verified (2026-09-07, updated 2026-09-08):** Python 3.11.15 in `.venv`, torch 2.14.0+cu130, **torchvision 0.29.0+cu130 (added 2026-09-08 — B4)**, transformers 5.16.1, OpenCV 5.x, 48-core CPU, 125 GB RAM, driver 595.84, 2× RTX 4500 Ada 24 GB.

> 📌 **If you recreate this environment from scratch:** install the CUDA builds from PyTorch's index, not PyPI (the venv here holds `+cu130` builds that the lockfile does not track): `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` — then `uv sync --extra vlm` for the rest.

> ⚠️ **VRAM guardrail honored in this audit:** both GPUs were ~84 % occupied by the user's local inference (20.6/24.6 GB and 21.2/24.6 GB). All stress tests therefore ran on CPU with ≤2-page subsets; the one operation that *would* have consumed ~16 GB of VRAM (loading the 7B VLM) was deliberately **not** executed — and its code path was validated deterministically instead (T5/T9).

```bash
# Sync deps (add --extra vlm for Phase 5+)
uv sync

# Test suite (host) — 18 tests
.venv/bin/python -m pytest tests/ -v

# Full pipeline (assembly + preprocessing) — ~20 s for 48 pages
.venv/bin/python src/run_pipeline.py --phase all

# Gate: pixel contracts + config contracts
.venv/bin/python src/test_contracts.py
.venv/bin/python src/validate_schema.py

# Preprocess dry-run (first 2 pages, manifest untouched)
.venv/bin/python -m src.preprocessing.run_preprocessing --limit 2

# GT scaffolding (then hand-type BEFORE looking at model output)
.venv/bin/python scripts/make_gt_template.py --ndocs 6

# Leakage pre-flight (add holdout IDs to configs/holdout_ids.txt first)
.venv/bin/python scripts/check_holdout_leakage.py

# Phase 5 / Phase 6 (need: uv sync --extra vlm, model weights, free VRAM)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.extraction.run_extraction --ndocs 6
.venv/bin/python -m src.ground_truth.scorer --tag vlm1280
.venv/bin/python -m src.detection.run_detection --limit 6

# Container path
make build && make test && make verify-gpu
make gpu CMD="python -m src.detection.run_detection --limit 6"
```

---

## 6. Test & Repair Log (2026-09-07 audit)

### 6.1 Stress tests — results

| # | Component under test | Method | Result |
|---|---|---|---|
| T1 | Unit suite (18 tests: config, ID, adaptive render, cleaner, prompt, parser, normalizer, scorer) | `pytest tests/ -v` | ✅ 18/18 pass (7.4 s) |
| T2 | Config contracts | `src/validate_schema.py` | ✅ ALL CONFIGS VALID |
| T3 | Entrypoint smoke | `run_pipeline.py --dummy` | ✅ clean |
| T4 | **Full end-to-end** (assembly → preprocessing, all 48 pages) | `run_pipeline.py --phase all` | ✅ 48/48 ok, 0 failed, 0 blank, 19.7 s wall; benchmark mean 0.30 s/page (spec 2.0 s: MEETS) |
| T4b | **ID stability** (roadmap's core invariant) | manifest key-hash before/after re-run | ✅ `document_id`+`page_index` **byte-identical**; path columns identical |
| T4c | Contract gate post re-run | `src/test_contracts.py` | ✅ PASS, all 48 rows |
| T5 | Phase 5 runner: merge, conflicts, first-wins, failure isolation, provenance, idempotency | deterministic fake model injected; `--ndocs 2 --overwrite` then no-`--overwrite` re-run | ✅ exit 1 with 1 simulated page failure (batch survived, failed page excluded from `pages_covered`, no page-JSON); conflicts `employee_id` / `blood_pressure_systolic` flagged; provenance matches winners; idempotent skip rc 0 — **this test surfaced bug #1** (see below) |
| T6 | Gate 1 scorer end-to-end: GT load, all 5 taxonomy classes, both tables, gate verdict, report write, mismatch list | temp GT (2 docs, 12 values) + T5 records; `scorer.main()`; full cleanup after | ✅ 5 correct / 2 wrong / 3 hallucinated / 1 missed / 1 correctly_nulled; typed 60 % → FAIL verdict (expected, synthetic); report written; temp artifacts removed |
| T7 | Holdout leakage pre-flight: empty-file, clean, and real-leak scenarios | `check_holdout_leakage.py` with temp ID + probe file; restored after | ⚠️ **surfaced bug #2** (self-leak) — fixed, then re-verified: all 3 scenarios correct (empty→1, clean→0, leak→1 pointing at the real file) |
| T8 | Visualization | `draw_boxes()` on a real raw render with 4 synthetic detections | ✅ review PNG written (1190×1600) |
| T9 | Detector pure parts (no torch needed): `_parse_boxes` (valid/invalid/empty), `pin_size` 28-multiple + long-edge tolerance; extractor `_pin_size` | direct calls | ✅ 952×1288 (both 28-multiples, long edge within 0.6 % of 1280) |
| T10 | GPU gate | `src/verify_gpu.py` | ✅ PASS: nvidia-smi reachable, driver 595.84 |
| T11 | Contact-sheet generator | `scripts/contact_sheet.py` | ✅ 2 sheets, 48 pages |
| T12 | Split-map generator (with backup/restore of the live `splits.yaml`) | `scripts/make_splits.py --starts 1,5,…,45` | ✅ 12 persons generated, `confirmed: false` enforced; live file restored byte-identical |
| T13 | **Real 7B VLM inference** (1 page, smallest possible workload) | — | ⛔ **Not executed (by design, not by failure):** model weights not present locally (HF cache holds only a 12 KB ref stub; a real run needs a ~16 GB download) and both GPUs are ~84 % occupied by the user's local inference. Per the VRAM guardrail this was not forced. The entire model-call code path (prompt → pin → parse → merge → validate → score) is covered by T5+T9+T1 instead. First real run is the documented next step (§8). |
| T14 | Repo-wide import audit (all 30 modules, including lazy-torch ones) | `pkgutil.walk_packages` + import | ✅ 0 failures (after bug #1 fix) |
| T15 | `make_gt_template.py` (scaffold + schema-field completeness + cleanup) | run `--ndocs 1`, assert 6 null fields, delete | ✅ |
| T16 | `run_preprocessing --limit 2` dry-run semantics | file mtime + md5 before/after | ✅ manifest file **unchanged** (dry-run honored) |
| T17 | **2026-09-08 — Phase 5 pre-flight after first real-run attempt**: `torchvision` import, torch/torchvision version pairing, `Qwen2_5_VLForConditionalGeneration` import, and the exact line that crashed (`AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')` incl. `Qwen2VLVideoProcessor`) | `uv pip install torchvision --index-url …/cu130`, then direct Python calls, **CPU-only, zero VRAM** | ✅ torchvision 0.29.0+cu130 (exact pair for torch 2.14.0+cu130); model class imports; `AutoProcessor` → `Qwen2_5_VLProcessor` loads, `video_processor` OK — the ImportError from B4 is gone; first real `--ndocs 1` run is unblocked (pending free VRAM) |
| T18 | **2026-09-08 — schema v2 integration**: deterministic fake model driven through the **real** `run_extraction.main()` (per-page mode) | 4 pages × v2 schema | ✅ 46 leaf values merged across 4 pages, junk sections pruned, deliberate cross-page conflict flagged, first-non-null-wins, provenance at array leaf `part_one[0].answer` → page 3, `record.json` validated; scorer: `?` exclusion, boolean mapping, positional array rows | 
| T19 | **2026-09-12 — single-pass document mode, runner level**: fake extractor driven through the **real** `run_extraction.main()` with `--mode document`; outputs redirected to a temp dir | new test `test_runner_document_mode_one_json_per_person` | ✅ one `record.json` per person (no `page_*.json`); page order asserted (`page_0000..0003` in order); provenance credited to each section's own page (p1 leaf→0, p3 leaf→2); `pages_covered=[0,1,2,3]`, `conflicts=[]`; `document.json` sidecar has mode/person_index/parse_ok/model_input | 
| T20 | **2026-09-12 — per-page A/B mode still works after restructure** | new test `test_runner_per_page_mode_still_merges` (fake extractor, real `main()`) | ✅ `--mode per_page` still writes 4 `page_NNNN.json` + merged `record.json`; cross-page merge intact | 
| T21 | **2026-09-12 — multi-image processor path (the part only a real run would catch)**: `Qwen2_5_VLProcessor` on person 1's REAL 4 `vlm_res` images — `apply_chat_template` with 4 image slots + document prompt, `processor(images=[4 imgs], min/max_pixels bracket)` | direct Python call, **CPU-only, zero VRAM** (no model weights loaded) | ✅ 4 image grids in `image_grid_thw` (1×92×68, 1×92×68, 1×92×66, 1×92×66); no resampling (pinned sizes preserved); input ≈ 8.5 k tokens (≈6.2 k image + ≈2.3 k text) → fits ~17 GB total incl. 4096-token output budget; document prompt structure verified (all 4 sections, page-labelled, rules block) | 
| — | **2026-09-12 — unit suite after changes** | `pytest tests/ -v` | ✅ **23/23 pass** (was 19; +4: document prompt, null-literal coercion, runner document mode, runner per-page mode) |

### 6.2 Failures detected

| # | Severity | Where | Symptom | Root cause |
|---|---|---|---|---|
| B1 | **Blocking** | `src/extraction/run_extraction.py:108` | `SyntaxError: invalid syntax` — the **entire Phase 5 extraction runner could not be imported or run**; any `--ndocs` run crashed before doing anything. Not caught by the pytest suite (tests import `vlm_extractor`, not the runner). | Typo: `if name not conflicts:` — missing `in` |
| B2 | **High** (breaks the evaluation-leakage guard the moment it is set up) | `scripts/check_holdout_leakage.py` | The scanner includes `configs/` + `.txt` in its scan set, and the ID list itself lives in `configs/holdout_ids.txt` → the script **always reported its own ID file as a leak** (exit 1) even with a perfectly clean tree, once any ID was added. | The ID file was not excluded from the scanned file list |
| B3 | Low (misleading) | `src/assembly/splitter.py:55` | Error message (shown only when `splits.yaml` is missing) directs users to `uv run python -m src.detection.propose_splits` — a module that does not exist (actual: `pose_splits`). | Copy-paste name drift |
| B4 | **Blocking** (surfaced 2026-09-08 by the user's first real `run_extraction --ndocs 1`) | `pyproject.toml` `vlm` extra / `.venv` | `ImportError: Qwen2VLVideoProcessor requires the Torchvision library` — transformers' `AutoProcessor` for Qwen2.5-VL **always** loads the video processor, but `torchvision` was neither declared in the `vlm` extra nor installed, so every Phase 5 run died during processor load (before any GPU work). | Missing dependency declaration: `vlm = [torch, transformers, accelerate]` omitted torchvision |
| B5 | **High** (data-quality defect in every v2 record; confirmed by visual audit of real outputs 2026-09-09 — e.g. DINESH `821c…`: roll/vision/BP came back as the string `"null"` although real values exist on other pages) | `src/models/fields.py::_coerce_leaf` | The model emits literal `"null"`/`"None"`/`"N/A"` strings for some empty cells; the coercion pass kept them as **values**, so records carry `"null"` strings and the scorer counts them as *hallucinated*. | Leaf coercion only treated empty/whitespace as null; null-ish literals passed through as data |
| B6 | **High** (would have made the first v2 run fail on every doc) | `src/extraction/vlm_extractor.py::_ask` | `max_new_tokens=800` is smaller than the full v2 output JSON (a single page-4 JSON with both declaration tables already exceeds 800 tokens) → the generation is **truncated mid-JSON** → `_first_json_object` finds no balanced object → `parse_ok=False` → all-null record. Only ever ran with the tiny v1 6-field JSON, so it never surfaced. | Generation budget set for the v1 baseline, never raised after the v2 contract |

### 6.3 Fixes applied (minimal, surgical — no refactors, no style shifts)

| Fix | File | Change |
|---|---|---|
| B1 | `src/extraction/run_extraction.py` | 1 line: `if name not conflicts:` → `if name not in conflicts:` |
| B2 | `scripts/check_holdout_leakage.py` | 1 condition added to the scan list comprehension: `and p != IDS_FILE` |
| B3 | `src/assembly/splitter.py` | 1 word in the error message: `propose_splits` → `pose_splits` |
| B5 | `src/models/fields.py` | `_coerce_leaf`: null-ish literals `{'null','none','n/a','n.a.'}` (case-insensitive) now coerce to `None` (1 condition added); covers both parse-time shaping and the merge view; test `test_null_literal_strings_become_null` |
| B6 | `src/extraction/vlm_extractor.py` | `_ask()` gains a `max_new_tokens` parameter; `extract_page` passes **2048** (was 800), `extract_document` passes **4096** — budgets now exceed the largest v2 JSON in both modes |
| B4 | `pyproject.toml` + environment | 1 line added to the `vlm` extra (`"torchvision>=0.19.0"`) so `uv sync --extra vlm` can no longer reproduce the gap; installed the matching build: `uv pip install torchvision --index-url https://download.pytorch.org/whl/cu130` → **0.29.0+cu130** (exact pair for torch 2.14.0+cu130; torch itself untouched) |

**Verification after fixes:** 18/18 pytest · 30/30 module imports · contract gate PASS (48 rows) · all configs valid · leak-check 3-scenario matrix correct (empty→1 / clean→0 / real-leak→1 pointing at the leaking file only) · full `--phase all` re-run green with byte-stable IDs. All test artifacts from T5/T6/T7 were removed; `configs/holdout_ids.txt` restored to its original (empty) content; `configs/splits.yaml` restored byte-identical.

**B4 verification (2026-09-08, T17):** `import torch, torchvision` → 2.14.0+cu130 / 0.29.0+cu130 paired; `Qwen2_5_VLForConditionalGeneration` imports; `AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')` → `Qwen2_5_VLProcessor` with `video_processor` instantiated — i.e. the exact failing line of the user's run now succeeds, verified CPU-side with zero VRAM consumed. The only remaining prerequisite for the real run is free VRAM (~16 GB) + the weight download (config/tokenizer files already cached from the 2026-09-07 attempt).

**B5/B6 + single-pass verification (2026-09-12, T19–T21):** 23/23 pytest (incl. 4 new tests driving the **real** `run_extraction.main()` with fake extractors — document mode output layout, per-page A/B regression, document prompt shape, null-literal coercion); multi-image `Qwen2_5_VLProcessor` path exercised on person 1's real 4 images, CPU-only, zero VRAM (4 grids, no resampling, ≈8.5 k input tokens). Real-model single-pass run pending free VRAM (~17 GB on one GPU; weights already local).

### 6.4 Notes & observations (not bugs; no change made)

- **`docs/group1_contract.md` missing** — referenced by stage docs + module docstrings; content exists in `literature/group1_foundation_architecture.md` §4. Either promote it or fix references.
- `src/models/schemas.py` has a redundant duplicate `from typing import ...` line (harmless; left as-is per minimal-change policy).
- `data/processed/242243220965001_PRAKASH CHANDRA MURMU.md` is a hand-typed reference extraction for person 10 — an excellent seed for the first real GT file (keep it out of `eval_holdout/` unless you choose it as a holdout).
- `colored_data.pdf` (54 pp) still has no split map — needs contact-sheet → `make_splits.py` → spot-check → `confirmed: true`.
- `README.md` is still a one-line placeholder.
- GPU container path (Docker Desktop context vs native daemon) was flagged in stage 3 as pending user action; the host-side GPU itself verifies fine (T10).

---

## 7. Gate Status

| Gate | Criterion | Status |
|---|---|---|
| Group 1 (Phases 1–4) exit | schema stable, manifest complete, ID stability, resolution separation, contract 100 %, benchmark | ✅ **Met** (re-verified 2026-09-07: 48/48 rows, IDs stable, 0.30 s/page) |
| GATE 1 (Phase 5) | typed fields > 90 % on hand-typed GT | 🟡 **Harness ready; no real model data yet** — run T13's prerequisite, then `run_extraction` + `scorer` |
| GATE 2 (Phase 9) | typed ≥ 95 %, handwritten at written threshold | ⬜ pending Phase 5–9 |

---

## 8. Recommended Next Steps (priority order)

1. **Commit the `nextStage` work** (Phase 5 + harness + the 3 fixes). Suggested message: `feat: phase 5 baseline extraction + gate 1 harness; fix: extraction runner syntax error, leak-check self-detection`.
2. **Set up ground truth** (the real blocker for Gate 1):
   ```bash
   .venv/bin/python scripts/make_gt_template.py --ndocs 6   # type BEFORE reading model output
   # move 2 completed files to data/ground_truth/eval_holdout/,
   # list their document_ids in configs/holdout_ids.txt
   .venv/bin/python scripts/check_holdout_leakage.py        # must be clean
   ```
3. **First real VLM run** (needs: `uv sync --extra vlm` already present — **incl. torchvision since B4**; **download** `Qwen/Qwen2.5-VL-7B-Instruct` ~16 GB when bandwidth/VRAM allow — config + tokenizer files are already cached, the weight shards still need to come down; ~16 GB free VRAM on one GPU):
   ```bash
   CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.extraction.run_extraction --ndocs 6
   .venv/bin/python -m src.ground_truth.scorer --tag vlm1280
   # A/B: --source raw --overwrite  (is preprocessing helping or hurting?)
   ```
4. **Run the detection prototype** on 2–6 pages and review the `vis/` overlays (this is the Gate-1 decision input for scoping Phase 6).
5. **Split `colored_data.pdf`** (contact sheets → `make_splits.py` → confirm).
6. Write the real `README.md`; materialize `docs/group1_contract.md`; add CI (pytest + `validate_schema` + `test_contracts` on a fixture manifest).

---

## 9. Post-audit change log (2026-09-08, after this audit)

### 9.1 First real VLM run — completed by the user

After the B4 (torchvision) fix, the user ran the real model: **6 documents × 4 pages extracted with `Qwen/Qwen2.5-VL-7B-Instruct`** (old flat v1 schema, timestamps 2026-09-07 23:51–23:54 IST). Outputs in `data/processed/extractions/<doc>/{page_*.json, record.json}` — **real model data, kept untouched**. Note: these records are v1-shaped (6 flat fields) and predate schema v2; a v2 re-run (`--overwrite`) supersedes them. `data/processed/eval/gate1_report_run1.json` is a stale synthetic report from the audit's T6 — superseded.

### 9.2 Schema v2 — hierarchical, page-structured extraction contract

**Why:** the v1 6-field baseline missed the report's actual content (fitness verdict, vision table, PME case box, declaration answers) — verified by visual analysis of a full 4-page person document at `vlm_res` resolution. The user then specified a page-structured hierarchical schema; implemented as a contract upgrade (no behavior beyond the contract).

**Changes (all verified: 19/19 pytest · configs valid · contract gate 48 rows · E2E fake-model run of the real runner):**

| File | Change |
|---|---|
| `configs/schema.yaml` | **Rewritten (v2):** 4 page sections per user spec — p1 `general_details`/`narrative_details`/`physical_traits`; p2 `pme_case{urine,sugar,alb}` + `right_side_table[]` (K1/K2 × R/L) + `handwritten_remarks`; p3 `fit_in_class`/`unfit_in_class` + `accuracy_of_vision.{right,left}_eye.*` (7 cells each) + `handwritten_text_below_vision_table`; p4 `part_one[]`/`part_two[]` (`question_no` + boolean `answer`, `template_rows` prefilled for GT) |
| `configs/schema.baseline.yaml` | v1 schema preserved for A/B |
| `src/models/schemas.py` | `FieldDefinition` recursive: `data_type` += `object`/`array`, `children`, `item`, `page`, `template_rows`; shape validator; `SchemaConfig.fields: List[FieldDefinition]`; `ExtractionRecord.fields` / `GroundTruthDoc.fields` now nested (`Dict[str, Any]`) |
| `src/models/fields.py` | **New** pure tree helpers: `iter_leaves`/`leaf_specs`, `null_shape` (GT skeleton), `prompt_skeleton`, `prune_to_shape` (model JSON → schema shape), `flatten_fields`/`unflatten_fields` (merge view; array rows indexed `part_one[0].answer`) |
| `src/extraction/vlm_extractor.py` | `build_extraction_prompt(schema, page)` — page-filtered field list + explicit nested output skeleton + array/boolean rules; tolerant **nested** parser (balanced-brace JSON scan); per-page prompt in `extract_page(..., page)` |
| `src/extraction/run_extraction.py` | Passes doc-relative page number; recursive merge = first-non-null-wins at leaf paths (conflicts/provenance keyed by dotted paths, incl. array indices); `record.json` fields nested |
| `src/ground_truth/normalize.py` | += `boolean` mapping (Yes/No ↔ true/false); identity set `{employee_id, roll_number}` case-folded |
| `src/ground_truth/scorer.py` | Recursive `score_document` over the schema tree; array rows scored positionally; leaf paths in reports |
| `scripts/make_gt_template.py` | GT skeleton now nested via `null_shape` (arrays prefilled from `template_rows`) |
| `tests/` | `test_extraction_baseline.py` rewritten (5 tests: page-filtered prompt, nested parse shaping, boolean normalize, nested taxonomy, positional array rows) + `test_manifest_stability.py` key-fields test → **19/19 pass** |

**Design invariants kept:** zero-shot, transcribe-exactly (model outputs strings; booleans coerced to "true"/"false" at parse time), null-if-unreadable, never-crash-the-batch, provenance + conflict flags, PII stays in `data/`.

### 9.3 Single-pass document extraction (2026-09-12)

**Why:** the per-page flow wrote 4 near-empty `page_XXXX.json` files per person (each page only carries a subset of the schema) and merged them with lossy first-non-null-wins; the user asked for **one merged document per person and one JSON per person**, with per-image/per-page output deprecated.

**Design decision (recorded here for the next developer):** the "merge step" is implemented as **one model call over all 4 pages attached in order** (`Qwen2.5-VL` natively supports multiple images per message — this IS the multi-page-document path in transformers; no PDF or vertical stitching needed). Vertical stitching was rejected: the processor would resample the whole tall image into one pixel budget, shrinking every page and degrading exactly the handwritten digits the project cares about (acceptance criterion: quality must not degrade). Each page keeps its own full pixel budget, page order is preserved (asserted in T19), and one full-schema JSON is generated in one pass — so there is no cross-page merge and no `conflicts`.

**Changes (all verified: 23/23 pytest · T19–T21 CPU-side · `configs/schema.yaml` untouched):**

| File | Change |
|---|---|
| `src/extraction/vlm_extractor.py` | `build_extraction_prompt(schema, page=None, n_pages=None)` — `n_pages` = document mode (all sections, page-labelled, document intro/rules); `_ask(imgs: list, prompt, max_new_tokens)` — multi-image content list, `min/max_pixels` bracketing per-image areas (zero resampling); **`extract_document([paths], long_edge)`** — single pass, one JSON; B6 budget fix (2048/4096) |
| `src/extraction/run_extraction.py` | **`--mode document` (default)**: logs person→document mapping, one call per person, `record.json` (provenance credited to each section's page; `pages_covered`; `conflicts=[]`) + `document.json` sidecar (mode, person_index, model_input order, parse_ok); **no per-page files**. `--mode per_page` = original flow, unchanged (A/B + task-5 fallback) |
| `src/models/fields.py` | B5: null-ish literals → null in `_coerce_leaf` |
| `tests/test_extraction_baseline.py` | +4 tests (document prompt; null literals; runner document mode via real `main()`; runner per-page regression) → 23 total |
| `start.md`, this report | Phase 5 run/check/tune instructions updated to the new default |

**Unchanged by design:** `configs/schema.yaml` (extraction contract), scorer (consumes `record.json` only — unaffected), detection stage (own `page_*.json` under `regions/` — different stage, unaffected), leakage checker, manifest/ID invariants.

**Run it (needs ~17 GB free VRAM on one GPU — pause local inference first):**
```bash
CUDA_VISIBLE_DEVICES=<n> .venv/bin/python -m src.extraction.run_extraction \
  --doc 821c0b69fec00b012b39ecee5f4cbd8a1b7a1de22b486674abfe7bacf08dc644 --overwrite
# then inspect:
#   data/processed/extractions/821c…/record.json     ← the ONE per-person JSON
#   data/processed/extractions/821c…/document.json   ← sidecar (page order, person, parse_ok)
# A/B vs the old flow:
CUDA_VISIBLE_DEVICES=<n> .venv/bin/python -m src.extraction.run_extraction \
  --doc 821c… --mode per_page --overwrite
```

**Verification (T18, 2026-09-08):** deterministic fake model driven through the **real** `run_extraction.main()` — 46 leaf values merged across 4 pages, junk sections pruned, deliberate cross-page conflict flagged (`page_2_medical_metrics.pme_case.sugar`), first-non-null-wins, provenance at `page_4_declarations.part_one[0].answer` → page 3, `record.json` validated; GT round-trip: `?` exclusion, boolean `No`==`false`, positional array scoring, case-sensitive names vs case-insensitive `roll_number`. All fake artifacts removed; the user's 6 real v1 extractions untouched.

---

*Report generated by the 2026-09-07 audit, updated 2026-09-08 (B4 fix + T17 verification; §9 schema v2 change log + T18). All test evidence reproducible via §5 commands. No PII leaves `data/` (gitignored); nothing in this report contains patient data.*
