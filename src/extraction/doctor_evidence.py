"""Save localized evidence independently from OCR and identity decisions."""
from pathlib import Path
import math

from PIL import Image, ImageDraw

from src.models.schemas import DoctorPageEvidence

COLORS = {"signature": "#1685ff", "stamp": "#f59e0b", "combined": "#14b86a"}


def combined_doctor_box(boxes, region=None, padding=0.008):
    """Prefer a localized enclosing doctor block; otherwise pad the union.

    Coordinates are full-page normalized values. A model box must contain all
    validated signature/stamp boxes and cannot cover more than 25% of the page.
    """
    def valid(box):
        return (isinstance(box,list) and len(box)==4
                and all(type(v) in (int,float) and math.isfinite(v) for v in box)
                and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
                and (box[2]-box[0])*(box[3]-box[1]) <= .25)
    parts = [box for box in boxes.values() if valid(box)]
    union = ([min(b[0] for b in parts),min(b[1] for b in parts),
              max(b[2] for b in parts),max(b[3] for b in parts)] if parts else None)
    if valid(region) and (union is None or (region[0] <= union[0] and region[1] <= union[1]
                                          and region[2] >= union[2] and region[3] >= union[3])):
        return region
    if union is None or (union[2]-union[0])*(union[3]-union[1]) > .5:
        return None
    return [max(0,union[0]-padding),max(0,union[1]-padding),
            min(1,union[2]+padding),min(1,union[3]+padding)]


def save_evidence(image: Image.Image, crops: dict, boxes: dict, idx: int,
                  out_dir: Path, root: Path, padding=0.008) -> DoctorPageEvidence:
    evidence = DoctorPageEvidence(page_index=idx, image_width=image.width,
                                  image_height=image.height)
    # Only validated boxes that actually produced a crop are persisted/drawn.
    evidence.boxes = {kind: boxes[kind] for kind in ("signature", "stamp") if crops[kind] is not None}
    combined = combined_doctor_box(evidence.boxes, boxes.get("doctor_region"), padding)
    if combined is None and not evidence.boxes:
        return evidence
    if combined is not None:
        evidence.boxes["combined"] = combined
    else:
        evidence.notes.append("Combined region too large; independent crops retained")
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
