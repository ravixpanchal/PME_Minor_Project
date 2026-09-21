Here's your end-to-end operator's guide, stage by stage. Every stage: **run → inspect → tune → verify**. Your venv is already complete (torch 2.14 + transformers are installed — no extra `uv sync` needed for CPU stages).

---

## Stage 0 — One-time GPU prep (do this before the first extraction)

```bash
# 1. Model weights are NOT cached yet (12 KB stub) — download once (~16 GB):
hf download Qwen/Qwen2.5-VL-7B-Instruct
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct   # should be ~16 GB

# 2. Free-VRAM check — the 7B model (bf16) needs ~16–18 GB free on ONE card:
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

---

## Stage 1 — Phase 3: Assembly (CPU, ~10 s)

**Run:**
```bash
.venv/bin/python src/run_pipeline.py --phase assembly
```

**Check outputs:**
```bash
ls data/interim/high_res/                                  # 12 person-docs
ls data/interim/high_res/821c0b69fec0*/                    # 4 PNGs each (200 DPI)
.venv/bin/python -c "import pandas as pd; \
  print(pd.read_parquet('data/metadata/manifest.parquet').head())"
cat data/metadata/splits_applied.json                      # audit trail
```
Open any `page_0000.png` — it should be that person's header page (photo box, typed name).

**Tune (where to make changes):**
| Want to change | Edit |
|---|---|
| Person boundaries (pages per person) | `configs/splits.yaml` (keep `confirmed: true`) — **IDs change if boundaries change** |
| Render DPI cap (300), low-quality threshold (150) | `configs/pipeline.yaml` → `rendering` / `quality_checks` |

**Verify:** `.venv/bin/python src/test_contracts.py`

---

## Stage 2 — Phase 4: Preprocessing (CPU, ~15 s)

**Run:**
```bash
.venv/bin/python src/run_pipeline.py --phase preprocess        # uses existing manifest
# or redo both stages:      .venv/bin/python src/run_pipeline.py --phase all
# or dry-run first 3 pages (manifest untouched):
.venv/bin/python -m src.preprocessing.run_preprocessing --limit 3
```

**Check outputs:**
- Side-by-side the cleaning effect: open `data/interim/high_res/<doc>/page_0001.png` (raw) vs `data/interim/high_res_clean/<doc>/page_0001.png` (deskewed/CLAHE/denoised)
- VLM copy: `data/interim/vlm_res/<doc>/page_0001.png` → ~890×1280 (check: `.venv/bin/python -c "from PIL import Image; print(Image.open('data/interim/vlm_res/<doc>/page_0001.png').size)"`)
- The run log prints per-page flags (`deskewed: True, skew_deg: 0.91`, `borders_trimmed`, …) — that's your tuning signal

**Tune:** `configs/pipeline.yaml` → `preprocessing` block — `deskew: false`, `crop_to_content`, `contrast_equalization`, `denoise.enabled` + `params` (d, sigma_color, sigma_space); and `vlm_long_edge_px: 1280` (drop to 1024 to save VRAM later). Deskew angle window (0.3°–5.0°) lives in [cleaner.py](src/preprocessing/cleaner.py) as `MIN_SKEW_DEG`/`MAX_SKEW_DEG`.

**Verify:** `.venv/bin/python src/test_contracts.py` → all 3 checks (raw fidelity + cleaned plausibility + VLM size) must be green.

---

## Stage 3 — Phase 5: The actual extraction (GPU) 🔥

**Run — smallest first, then scale:**
```bash
# ONE document (4 pages) as the smoke test:
# default is --mode document: ONE model call over all 4 pages (in order)
# -> ONE consolidated JSON per person, no per-page files
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.extraction.run_extraction --ndocs 1

# Then the real batch:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.extraction.run_extraction --ndocs 6

# A/B: is preprocessing helping or hurting? (feeds ~2688px instead of 1280px)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.extraction.run_extraction \
  --ndocs 1 --source raw --overwrite

# A/B: old per-page flow (4 calls + first-non-null merge; writes page_XXXX.json)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.extraction.run_extraction \
  --ndocs 1 --mode per_page --overwrite
```
Useful flags: `--doc <document_id>` (one specific person), `--mode document|per_page` (default `document`), `--source vlm|clean|raw`, `--long-edge 2016` (resolution A/B), `--overwrite` (force redo; without it, docs with an existing `record.json` are skipped — idempotent).

**Check outputs (this is where you actually judge quality):**
```bash
ls data/processed/extractions/                                # one folder per doc
cat data/processed/extractions/<doc>/record.json              # THE per-person JSON: fields (full nested schema), provenance, pages_covered
cat data/processed/extractions/<doc>/document.json            # sidecar: mode, person_index, model_input (page order), parse_ok
```
(per-page mode only: `page_XXXX.json` sidecars per page.)
Then **open the page image** (`data/interim/vlm_res/<doc>/page_0000.png`) and compare against what the model wrote — especially the handwritten `vision_left` and `blood_pressure_systolic`.

**Tune:**
| Want to change | Edit |
|---|---|
| Field-level prompt hints (the highest-leverage knob) | `configs/schema.yaml` → `extraction_hint` per field (goes straight into the prompt) |
| Prompt structure / rules | `build_extraction_prompt()` in [vlm_extractor.py](src/extraction/vlm_extractor.py) |
| Which model | `configs/models.yaml` → `extraction.default_vlm` |
| Generation length / decoding | `max_new_tokens` (2048 per-page / 4096 document), `do_sample=False` in `vlm_extractor.py` → `_ask()` |
| Input resolution | `--source` / `--long-edge` flags |

**If you OOM:** lower `--long-edge` (for `--source raw/clean`), free VRAM, or use the other card.

---

## Stage 4 — Ground truth + Gate 1 scoring (CPU)

This is the loop that turns "looks plausible" into a number:

```bash
# 1. Scaffold GT skeletons (all fields null):
.venv/bin/python scripts/make_gt_template.py --ndocs 6

# 2. HAND-TYPE each data/ground_truth/gt/<doc>.json from the cleaned images
#    (open data/interim/high_res_clean/<doc>/page_*.png)
#    null = truly absent · '?' = unreadable (excluded from scoring)
#    ⚠️ do it BEFORE reading record.json — no anchoring on the model's answers

# 3. Reserve 2 docs as holdout (never in prompts/few-shots):
mv data/ground_truth/gt/<docA>.json data/ground_truth/eval_holdout/
echo <docA> > configs/holdout_ids.txt      # + docB on line 2
.venv/bin/python scripts/check_holdout_leakage.py    # must print clean

# 4. Score against the extraction:
.venv/bin/python -m src.ground_truth.scorer --tag run1
#    → console: typed vs handwritten accuracy tables + mismatch list
#    → file: data/processed/eval/gate1_report_run1.json
```

**The iteration loop:** read the MISMATCHES list → fix the `extraction_hint` / prompt → re-extract `--ndocs N --overwrite` → re-score with a **new tag** (`--tag run2`) so you can compare reports. Gate 1 passes when typed ≥ 90 % (the scorer prints PASS/FAIL).

---

## Stage 5 — Phase 6: Region detection (GPU, the "do we need a detector?" experiment)

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.detection.run_detection --limit 6
# A/B against preprocessing:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m src.detection.run_detection --limit 6 --source raw --overwrite
```
**Check:** `data/processed/regions/<doc>/page_0000.json` (normalized boxes + class) and, more importantly, the **visual overlays** `data/processed/regions/<doc>/vis/page_0000.png` — colored boxes per class (blue typed / orange handwritten / red discard). If all boxes pile in one corner, that's the known coordinate-drift bug signal.

Optional: `.venv/bin/python -m src.detection.pose_splits` → proposes person boundaries into `configs/splits_draft.yaml` (never auto-applied — you confirm).

---

## Stage 6 — Regression gates (run after any change, all CPU)

```bash
.venv/bin/python -m pytest tests/ -v                    # 23 tests
.venv/bin/python src/validate_schema.py                # config contracts
.venv/bin/python src/test_contracts.py                 # pixel-dimension gate
.venv/bin/python scripts/check_holdout_leakage.py      # leakage pre-flight
```

---

## Gotchas worth knowing

1. **One GPU job at a time** — extraction, detection, and your local inference all fight for the same ~24 GB; don't run two of them concurrently.
2. **Single-source assembly:** `run_assembly` and `make_splits.py` are hardwired to `data/raw/black_dataset.pdf` (the `SOURCE` constant + `splits.yaml`). To bring in `colored_data.pdf` (54 pp) you must repoint both — or ask me and I'll add proper multi-source support.
3. **Schema is the single source of truth:** add a field to `configs/schema.yaml` and it automatically flows into the extraction prompt, the GT template, and the scorer — nothing else to wire up.
4. **`splits.yaml` is a hard gate:** unconfirmed maps make assembly refuse to run (by design — "never merge on a guess").
5. **Raw renders are immutable:** never "fix" things inside `high_res/`; change preprocessing and re-run, the raw-vs-clean diff is your debugging path.

If you'd like, I can save this as a `RUNBOOK.md` in the repo, or walk through the first real `--ndocs 1` extraction live once you've freed up a GPU.