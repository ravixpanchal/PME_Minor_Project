"""Load the explicit candidate subset from the latest extraction command."""
import json
from pathlib import Path


def selected_documents(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    ids = payload.get("document_ids")
    if not isinstance(ids, list) or not ids or any(
        not isinstance(doc, str) or not doc or Path(doc).name != doc or doc in {'.', '..'}
        for doc in ids
    ):
        raise ValueError("Selection must contain a nonempty document_ids list of directory names")
    return set(ids)
