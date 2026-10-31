# Medical report extraction

Four-page candidate report extraction with a Streamlit dashboard and a persistent
doctor signature/stamp gallery. See [the GPU runbook and change inventory](docs/doctor-recognition.md)
for setup, commands, recognition decisions, limitations and validation results.

```bash
uv sync --extra vlm --group dev
# On the GPU machine, after configuring configs/splits.yaml:
uv run --extra vlm python -m src.run_pipeline --phase all
uv run python -m src.pme.sync
uv run streamlit run frontend/app.py
```

Use `--overwrite` when upgrading existing extraction records. Back up `data/`:
it contains sensitive reports, extracted crops and the persistent doctor gallery.
