"""Save localized evidence independently from OCR and identity decisions."""
from pathlib import Path

from PIL import Image, ImageDraw

from src.models.schemas import DoctorPageEvidence

COLORS = {"signature": "#1685ff", "stamp": "#f59e0b", "combined": "#14b86a"}


def save_evidence(image: Image.Image, crops: dict, boxes: dict, idx: int,
                  out_dir: Path, root: Path) -> DoctorPageEvidence:
    evidence = DoctorPageEvidence(page_index=idx, image_width=image.width,
                                  image_height=image.height)
    # Only validated boxes that actually produced a crop are persisted/drawn.
    evidence.boxes = {kind: boxes[kind] for kind in ("signature", "stamp") if crops[kind] is not None}
    if not evidence.boxes:
        return evidence
    evidence.boxes["combined"] = [
        min(b[0] for b in evidence.boxes.values()),
        min(b[1] for b in evidence.boxes.values()),
        max(b[2] for b in evidence.boxes.values()),
        max(b[3] for b in evidence.boxes.values()),
    ]
    directory = out_dir / "doctor_evidence" / f"page_{idx:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for kind in ("combined", "signature", "stamp"):
        if kind not in evidence.boxes:
            continue
        box = evidence.boxes[kind]
        pixels = [round(box[0]*image.width), round(box[1]*image.height),
                  round(box[2]*image.width), round(box[3]*image.height)]
        evidence.boxes_pixels[kind] = pixels
        draw.rectangle(pixels, outline=COLORS[kind], width=max(2, image.width//500))
        label_y = min(image.height-14, pixels[3]+4) if kind == "combined" else max(0, pixels[1]-14)
        draw.text((pixels[0]+3, label_y),
                  "Doctor evidence" if kind == "combined" else f"Doctor {kind}",
                  fill=COLORS[kind], stroke_width=1)
        crop = image.crop(tuple(pixels)) if kind == "combined" else crops[kind]
        path = directory / f"{kind}.png"
        crop.save(path)
        setattr(evidence, f"{kind}_path", path.relative_to(root).as_posix())
    path = directory / "annotated.png"
    annotated.save(path)
    evidence.annotated_path = path.relative_to(root).as_posix()
    return evidence
