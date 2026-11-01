"""Conservative, untrained visual matching with a persistent SQLite gallery.

No model is loaded here. The caller supplies its existing VLM for localization
and crop OCR. Scores are visual similarity measures, NOT identity probabilities.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from src.extraction.vlm_extractor import _first_json_object, _pin_size
from src.models.schemas import DoctorRecognition
from src.extraction.doctor_evidence import save_evidence
from src.utils.logger import setup_logger

log = setup_logger("doctor_recognition")


class RecognitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registry_dir: str = "data/processed/doctors"
    similarity_threshold: float = Field(default=0.90, ge=0.8, le=1)
    ambiguity_margin: float = Field(default=0.10, gt=0, le=1)
    min_inliers: int = Field(default=18, ge=8)
    min_ink_pixels: int = Field(default=100, ge=30)
    detection_long_edge: int = Field(default=2016, ge=672, le=2688)
    retry_empty_detection: bool = True
    min_sharpness: float = Field(default=12, ge=0)


def name_key(name: str | None) -> str:
    text = re.sub(r"^\s*(?:dr\.?|doctor)\s+", "", name or "", flags=re.I)
    return " ".join(re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE))


def valid_name(name: object) -> str | None:
    if not isinstance(name, str):
        return None
    name = name.strip()
    key = name_key(name)
    if len(key.replace(" ", "")) < 5 or len(key.split()) < 2:
        return None
    if any(word in key.split() for word in
           ("unknown", "unreadable", "null", "none", "medical", "officer", "examiner", "hospital")):
        return None
    return name


def crop_box(image: Image.Image, box: object) -> Image.Image | None:
    """Reject malformed/out-of-frame boxes instead of silently clamping."""
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    if (x2-x1)*(y2-y1) > .25:
        return None
    bounds = (round(x1*image.width), round(y1*image.height),
              round(x2*image.width), round(y2*image.height))
    if bounds[2]-bounds[0] < 24 or bounds[3]-bounds[1] < 16:
        return None
    return image.crop(bounds)


def normalize(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is None:
        return np.zeros((256, 512), np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    ink = ink[y:y+h, x:x+w]
    ratio = min(480/w, 224/h)
    ink = cv2.resize(ink, (max(1, round(w*ratio)), max(1, round(h*ratio))))
    canvas = np.zeros((256, 512), np.uint8)
    top, left = (256-ink.shape[0])//2, (512-ink.shape[1])//2
    canvas[top:top+ink.shape[0], left:left+ink.shape[1]] = ink
    return canvas


def usable(image: Image.Image, cfg: RecognitionConfig) -> bool:
    gray = np.asarray(image.convert("L"))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = np.count_nonzero(ink) / ink.size
    return (np.count_nonzero(ink) >= cfg.min_ink_pixels and .005 < density < .45
            and cv2.Laplacian(gray, cv2.CV_64F).var() >= cfg.min_sharpness)


def similarity(a: Image.Image, b: Image.Image, cfg: RecognitionConfig) -> float:
    """SIFT ratio matches + RANSAC geometry + aligned ink overlap.

    Generic stamp text must never be part of a signature crop. Reject weak
    geometry, small feature clusters and implausible scale/rotation changes.
    """
    a, b = normalize(a), normalize(b)
    sift = cv2.SIFT_create(nfeatures=800)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        return 0.0
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2
            for m, n in [pair] if m.distance < .65*n.distance]
    if len(good) < cfg.min_inliers:
        return 0.0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                            ransacReprojThreshold=3)
    if matrix is None or mask is None:
        return 0.0
    selected = mask.ravel().astype(bool)
    if selected.sum() < cfg.min_inliers or selected.mean() < .8:
        return 0.0
    scale = np.hypot(matrix[0, 0], matrix[1, 0])
    rotation = abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    if not .75 <= scale <= 1.33 or rotation > 20:
        return 0.0
    for points, ink in ((pa[selected], a), (pb[selected], b)):
        ys, xs = np.nonzero(ink)
        if not len(xs) or np.ptp(points[:, 0]) < .45*np.ptp(xs) or np.ptp(points[:, 1]) < .3*np.ptp(ys):
            return 0.0
    warped = cv2.warpAffine(a, matrix, (512, 256)) > 100
    target = b > 100
    kernel = np.ones((3, 3), np.uint8)
    aw = cv2.dilate(warped.astype(np.uint8), kernel).astype(bool)
    bw = cv2.dilate(target.astype(np.uint8), kernel).astype(bool)
    overlap = min((warped & bw).sum()/max(1, warped.sum()),
                  (target & aw).sum()/max(1, target.sum()))
    return float(overlap * selected.mean())


class DoctorRegistry:
    def __init__(self, root: Path, cfg: RecognitionConfig):
        self.root, self.cfg = root, cfg
        root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS doctors (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT UNIQUE NOT NULL,
                    signature BLOB NOT NULL, stamp BLOB NOT NULL, source_document TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "registry.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def identify(self, signature: Image.Image, stamp: Image.Image | None,
                 name: str | None, document_id: str) -> DoctorRecognition:
        from io import BytesIO
        with self.connect() as db:
            # Serialize lookup + enrollment across extraction workers.
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT * FROM doctors").fetchall()
            scores = []
            for row in rows:
                with Image.open(BytesIO(row["signature"])) as saved:
                    score = similarity(signature, saved, self.cfg)
                scores.append((score, row))
            scores.sort(key=lambda pair: pair[0], reverse=True)
            best = scores[0][0] if scores else 0.0
            second = scores[1][0] if len(scores) > 1 else 0.0
            result = DoctorRecognition(status="unmatched", confidence=best)
            if best >= self.cfg.similarity_threshold:
                row = scores[0][1]
                if best-second < self.cfg.ambiguity_margin:
                    return result.model_copy(update={"status": "ambiguous"})
                if name and name_key(name) != row["name_key"]:
                    return result.model_copy(update={"status": "identity_conflict"})
                # Do not add signature-only matches to the gallery (prevents drift).
                return DoctorRecognition(status="matched", confidence=best,
                                         doctor_id=row["id"], doctor_name=row["name"])
            if not name or stamp is None:
                return result
            # Same-name weak match may be a namesake or a changed signature.
            # Moderate visual resemblance may be an OCR variant: require review.
            if any(row["name_key"] == name_key(name) for row in rows) or best >= .6:
                return result.model_copy(update={"status": "needs_review"})
            identifier = hashlib.sha256(name_key(name).encode()).hexdigest()[:24]
            def png(image):
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue()
            db.execute("INSERT INTO doctors(id,name,name_key,signature,stamp,source_document) VALUES(?,?,?,?,?,?)",
                       (identifier, name, name_key(name), png(signature), png(stamp), document_id))
            return DoctorRecognition(status="registered", doctor_id=identifier,
                                     doctor_name=name, confidence=best)


class DoctorRecognizer:
    def __init__(self, extractor, root: Path, cfg: RecognitionConfig):
        self.extractor, self.root, self.cfg = extractor, root, cfg
        self.registry = DoctorRegistry(root / cfg.registry_dir, cfg)

    def ask_json(self, image, prompt, long_edge=1344):
        sent = _pin_size(image, long_edge)
        raw = self.extractor._ask([sent], prompt, max_new_tokens=1000)
        if getattr(self, "diagnostic_dir", None) is not None:
            self.request_number += 1
            (self.diagnostic_dir / f"response_{self.request_number:02d}.json").write_text(
                json.dumps({"image_size": list(sent.size), "prompt": prompt, "response": raw}, indent=2))
        obj = _first_json_object(raw)
        if obj is None:
            raise ValueError("Doctor localization/OCR returned no JSON object; inspect saved responses")
        result = json.loads(obj)
        if not isinstance(result, dict):
            raise ValueError("Invalid doctor evidence object")
        return result

    def locate(self, image):
        sent = _pin_size(image, self.cfg.detection_long_edge)
        obj = self.ask_json(image,
            'Locate only the Railway medical examiner doctor signature and stamp. '
            'Exclude candidate/witness signatures, printed labels and clinical handwriting. '
            f'This image is {sent.width} pixels wide and {sent.height} pixels high. '
            'Return {"coordinate_space":"pixels", "signature":[x1,y1,x2,y2] or null, '
            '"stamp":[x1,y1,x2,y2] or null, "same_doctor":true or false, '
            '"multiple_doctors":true or false}. Use absolute pixel coordinates in THIS image. '
            'Preserve both regions even if they overlap. Do not transcribe the signature as text. '
            'Use tight boxes; null only for an absent or unidentifiable region. '
            'same_doctor must be true only for an unambiguous signature/stamp pair.',
            long_edge=self.cfg.detection_long_edge)
        space = obj.get("coordinate_space", "normalized")
        for kind in ("signature", "stamp"):
            box = obj.get(kind)
            if isinstance(box, dict):
                box = box.get("box", box.get("bbox"))
            if isinstance(box, list) and len(box) == 4:
                if space == "pixels":
                    box = [v/(sent.width if i % 2 == 0 else sent.height) if type(v) in (int,float) else v for i,v in enumerate(box)]
                elif space == "normalized_1000":
                    box = [v/1000 if type(v) in (int,float) else v for v in box]
                elif space != "normalized":
                    box = None
            obj[kind] = box
        return obj

    def process(self, document_id: str, pages: dict[int, Path], out_dir: Path):
        candidates, errors, evidence = [], [], []
        multiple_doctors = False
        for idx in (2, 3):
            if idx not in pages:
                continue
            try:
                with Image.open(pages[idx]) as im:
                    image = im.convert("RGB")
                self.diagnostic_dir = out_dir / "doctor_evidence" / f"page_{idx:04d}"
                self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
                self.request_number = 0
                obj = self.locate(image)
                if self.cfg.retry_empty_detection and all(crop_box(image, obj.get(k)) is None for k in ("signature", "stamp")):
                    # A second, larger view of the bottom portion can expose small
                    # examiner marks. Coordinates must be mapped back to raw page.
                    top = int(image.height * .35)
                    local = image.crop((0, top, image.width, image.height))
                    retry = self.locate(local)
                    for kind in ("signature", "stamp"):
                        box = retry.get(kind)
                        if crop_box(local, box) is not None:
                            retry[kind] = [box[0], (top+box[1]*local.height)/image.height,
                                           box[2], (top+box[3]*local.height)/image.height]
                        else:
                            retry[kind] = None
                    obj = retry
                multiple_doctors |= obj.get("multiple_doctors") is True
                sig, stamp = crop_box(image, obj.get("signature")), crop_box(image, obj.get("stamp"))
                item = save_evidence(image, {"signature": sig, "stamp": stamp}, obj, idx, out_dir, self.root)
                evidence.append(item)
                if not item.boxes:
                    item.notes.append("No valid doctor boxes detected; inspect response JSON files in this page directory")
                log.info("doc %s page %d: signature_saved=%s stamp_saved=%s diagnostics=%s",
                         document_id[:12], idx+1, bool(item.signature_path), bool(item.stamp_path), self.diagnostic_dir)
                item.same_doctor = obj.get("same_doctor") is True
                item.signature_usable = sig is not None and usable(sig, self.cfg)
                item.stamp_usable = stamp is not None and usable(stamp, self.cfg)
                if sig is not None and stamp is not None:
                    x1, y1, x2, y2 = obj["signature"]
                    a1, b1, a2, b2 = obj["stamp"]
                    if min(x2, a2) > max(x1, a1) and min(y2, b2) > max(y1, b1):
                        item.signature_usable = False
                        item.notes.append("Overlapping stamp/signature retained for review; excluded from matching")
                if sig is not None and not item.signature_usable:
                    item.notes.append("Signature is not suitable for automatic matching")
                if stamp is not None and not item.stamp_usable:
                    item.notes.append("Stamp is not suitable for OCR/enrollment")
                # OCR failure must never discard crops or the signature candidate.
                candidate = dict(idx=idx, signature=sig if item.signature_usable else None,
                                 stamp=stamp if item.stamp_usable else None, name=None, text=None,
                                 paired=item.same_doctor, boxes=item.boxes, evidence=item)
                candidates.append(candidate)
                name, text = None, None
                if item.stamp_usable:
                    first = self.ask_json(stamp,
                        'OCR this doctor stamp. Return {"text": "verbatim visible stamp text", '
                        '"doctor_name": "full personal name" or null}. Do not infer missing letters '
                        'or use a designation/hospital as a name. Unreadable personal name must be null.')
                    item.stamp_text = first.get("text") if isinstance(first.get("text"), str) else None
                    second = self.ask_json(stamp,
                        'Read only the fully legible personal name printed inside this stamp. '
                        'Return {"doctor_name": "name" or null}. Never complete initials or guess letters.')
                    a, b = valid_name(first.get("doctor_name")), valid_name(second.get("doctor_name"))
                    text = first.get("text")
                    item.stamp_text = text if isinstance(text, str) else None
                    if not (a and b and name_key(a) == name_key(b)):
                        item.notes.append("Stamp personal name absent/unreadable or OCR reads disagree; not enrolled")
                    if a and b and name_key(a) == name_key(b) and isinstance(text, str) and name_key(a) in name_key(text):
                        name = a
                item.doctor_name = name
                item.stamp_text = text if isinstance(text, str) else None
                candidate.update(name=name, text=item.stamp_text)
            except Exception as exc:
                errors.append(f"Page {idx+1}: {type(exc).__name__}: {exc}")
                log.warning("doc %s page %d: %s", document_id[:12], idx+1, exc)
        names = {name_key(c["name"]) for c in candidates if c["name"]}
        candidates.sort(key=lambda c: (c["signature"] is not None,
                                      bool(c["name"] and c["paired"]),
                                      bool(c["evidence"].signature_path),
                                      bool(c["evidence"].stamp_path)), reverse=True)
        if not candidates:
            return DoctorRecognition(status="error" if errors else "missing", notes=errors, evidence=evidence)
        best = candidates[0]
        result = DoctorRecognition(status="missing_signature", doctor_name=best["name"])
        signatures = [c for c in candidates if c["signature"] is not None]
        if multiple_doctors:
            result = DoctorRecognition(status="ambiguous", notes=["Multiple doctors on report"])
        elif len(names) > 1:
            result = DoctorRecognition(status="identity_conflict", notes=["Stamp names disagree across pages"])
        elif len(signatures) == 2 and similarity(signatures[0]["signature"], signatures[1]["signature"], self.cfg) < self.cfg.similarity_threshold:
            result = DoctorRecognition(status="needs_review", notes=["Page 3/4 signatures disagree or cannot be reliably compared"])
        elif best["signature"] is not None:
            # Check any readable stamp name for contradictions, even if unpaired.
            observed_name = next((c["name"] for c in candidates if c["name"]), None)
            result = self.registry.identify(best["signature"], best["stamp"] if best["paired"] and best["name"] else None,
                                            observed_name, document_id)
        elif best["evidence"].signature_path:
            result = DoctorRecognition(status="needs_review", notes=["Signature extracted but unsuitable for matching"])
        selected = best["evidence"]
        for field in ("signature_path", "stamp_path", "combined_path", "annotated_path", "page_index", "boxes", "stamp_text"):
            setattr(result, field, getattr(selected, field))
        result.evidence = evidence
        result.notes.extend(errors)
        log.info("doc %s doctor: status=%s name=%s similarity=%.3f signature=%s stamp=%s notes=%s",
                 document_id[:12], result.status, result.doctor_name, result.confidence,
                 result.signature_path, result.stamp_path, result.notes)
        return result
