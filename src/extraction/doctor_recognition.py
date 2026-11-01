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
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.extraction.vlm_extractor import _first_json_object, _pin_size
from src.models.schemas import DoctorRecognition
from src.extraction.doctor_evidence import save_evidence
from src.extraction.signature_isolation import isolate_signature, overlapping_stamp
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
    layout_fallback_enabled: bool = False
    layout_search_regions: dict[int, list[float]] = Field(default_factory=dict)
    combined_padding: float = Field(default=.008, ge=0, le=.03)
    min_sharpness: float = Field(default=12, ge=0)


    @field_validator("layout_search_regions")
    @classmethod
    def valid_search_regions(cls, value):
        for page, box in value.items():
            if page not in (2,3) or len(box)!=4 or not all(math.isfinite(v) for v in box):
                raise ValueError("Layout search regions require page 2/3 and four finite coordinates")
            if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError("Layout search regions must be normalized")
        return value


def name_key(name: str | None) -> str:
    text = re.sub(r"^\s*(?:dr\.\s*|dr\s+|doctor\s+)", "", name or "", flags=re.I)
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


def crop_box(image: Image.Image, box: object, max_area=.25) -> Image.Image | None:
    """Reject malformed/out-of-frame boxes instead of silently clamping."""
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    if (x2-x1)*(y2-y1) > max_area:
        return None
    bounds = (round(x1*image.width), round(y1*image.height),
              round(x2*image.width), round(y2*image.height))
    if bounds[2]-bounds[0] < 24 or bounds[3]-bounds[1] < 16:
        return None
    return image.crop(bounds)


def normalize(image: Image.Image, return_mask=False) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    allowed = np.asarray(image.getchannel("A")) if image.mode == "RGBA" else np.full(gray.shape,255,np.uint8)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is None:
        empty = np.zeros((256,512),np.uint8)
        return (empty, np.zeros_like(empty)) if return_mask else empty
    x, y, w, h = cv2.boundingRect(points)
    ink = ink[y:y+h, x:x+w]
    allowed = allowed[y:y+h,x:x+w]
    ratio = min(480/w, 224/h)
    ink = cv2.resize(ink, (max(1, round(w*ratio)), max(1, round(h*ratio))))
    canvas = np.zeros((256, 512), np.uint8)
    top, left = (256-ink.shape[0])//2, (512-ink.shape[1])//2
    canvas[top:top+ink.shape[0], left:left+ink.shape[1]] = ink
    mask = np.full(canvas.shape,255,np.uint8)
    mask[top:top+ink.shape[0],left:left+ink.shape[1]] = cv2.resize(allowed,(ink.shape[1],ink.shape[0]),interpolation=cv2.INTER_NEAREST)
    return (canvas,mask) if return_mask else canvas


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
    a, mask_a = normalize(a, return_mask=True)
    b, mask_b = normalize(b, return_mask=True)
    sift = cv2.SIFT_create(nfeatures=800)
    ka, da = sift.detectAndCompute(a, cv2.erode(mask_a,np.ones((25,25),np.uint8)))
    kb, db = sift.detectAndCompute(b, cv2.erode(mask_b,np.ones((25,25),np.uint8)))
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
    common = (cv2.warpAffine(mask_a,matrix,(512,256),flags=cv2.INTER_NEAREST) > 250) & (mask_b > 250)
    if (warped & common).sum() < .5*max(1,warped.sum()) or (target & common).sum() < .5*max(1,target.sum()):
        return 0.0
    warped &= common
    target &= common
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
                CREATE TABLE IF NOT EXISTS doctor_names (
                    id TEXT PRIMARY KEY, name_key TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS doctor_name_evidence (
                    document_id TEXT NOT NULL, page_index INTEGER NOT NULL CHECK(page_index IN (2,3)),
                    name_id TEXT NOT NULL REFERENCES doctor_names(id), stamp_text TEXT,
                    raw_signature BLOB, stamp BLOB, ocr_region BLOB NOT NULL,
                    localization_method TEXT NOT NULL, paired INTEGER NOT NULL,
                    PRIMARY KEY(document_id,page_index));
                CREATE TABLE IF NOT EXISTS signature_templates (
                    doctor_id TEXT PRIMARY KEY REFERENCES doctors(id),
                    algorithm TEXT NOT NULL, matching_signature BLOB NOT NULL,
                    source_document TEXT NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "registry.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_stamp_name(self, name, document_id, evidence, root):
        """Persist stamp OCR immediately, independently from signature enrollment.

        A unique normalized NAME is a catalog entry, not proof that all people
        with that name share one identity. Matching never reads this evidence table.
        """
        if valid_name(name) is None or evidence.stamp_ocr_path is None:
            return None
        key = name_key(name)
        identifier = hashlib.sha256(key.encode()).hexdigest()[:24]
        def asset(path):
            return (root / path).read_bytes() if path else None
        signature = asset(evidence.signature_path)
        stamp = asset(evidence.stamp_path)
        region = asset(evidence.stamp_ocr_path)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO doctor_names(id,name_key,display_name) VALUES(?,?,?) "
                       "ON CONFLICT(name_key) DO NOTHING", (identifier,key,name))
            db.execute("INSERT INTO doctor_name_evidence VALUES(?,?,?,?,?,?,?,?,?) "
                       "ON CONFLICT(document_id,page_index) DO UPDATE SET "
                       "name_id=excluded.name_id,stamp_text=excluded.stamp_text,"
                       "raw_signature=excluded.raw_signature,stamp=excluded.stamp,"
                       "ocr_region=excluded.ocr_region,localization_method=excluded.localization_method,"
                       "paired=excluded.paired",
                       (document_id,evidence.page_index,identifier,evidence.stamp_text,
                        signature,stamp,region,evidence.localization_method,int(evidence.same_doctor)))
        return identifier

    def identify(self, signature: Image.Image, stamp: Image.Image | None,
                 name: str | None, document_id: str, raw_signature: Image.Image | None = None) -> DoctorRecognition:
        from io import BytesIO
        with self.connect() as db:
            # Serialize lookup + enrollment across extraction workers.
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT d.*, t.matching_signature, t.algorithm FROM doctors d "
                              "LEFT JOIN signature_templates t ON t.doctor_id=d.id").fetchall()
            def png(image):
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                return buffer.getvalue()
            # Old raw templates may contain printed text. Rebuild only from the
            # original source report with its paired, independently readable stamp.
            legacy = next((r for r in rows if r["name_key"] == name_key(name)
                           and r["matching_signature"] is None), None)
            if legacy and stamp is not None and legacy["source_document"] == document_id:
                db.execute("INSERT INTO signature_templates VALUES(?,?,?,?)",
                           (legacy["id"], "sift-ransac-masked-v2", png(signature), document_id))
                return DoctorRecognition(status="registered", doctor_id=legacy["id"],doctor_name=legacy["name"],
                                         notes=["Rebuilt legacy reference from isolated handwriting"] )
            scores = []
            for row in rows:
                if row["matching_signature"] is None or row["algorithm"] != "sift-ransac-masked-v2":
                    continue
                with Image.open(BytesIO(row["matching_signature"])) as saved:
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
            db.execute("INSERT INTO doctors(id,name,name_key,signature,stamp,source_document) VALUES(?,?,?,?,?,?)",
                       (identifier, name, name_key(name), png(raw_signature if raw_signature is not None else signature), png(stamp), document_id))
            db.execute("INSERT INTO signature_templates VALUES(?,?,?,?)",
                       (identifier, "sift-ransac-masked-v2", png(signature), document_id))
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

    def locate(self, image, page_index=None):
        sent = _pin_size(image, self.cfg.detection_long_edge)
        layout = (
            "Report page 3: the examiner block is near the bottom centre/right, usually left of the candidate photograph. "
            "Anchor on 'Initials of Railway Medical Examiner' and 'Office Seal'. "
            if page_index == 2 else
            "Report page 4: the examiner block is at the bottom right, BELOW the candidate signature/thumbprint section. "
            "Anchor on 'Signature of the Medical Examiner'. " if page_index == 3 else "")
        obj = self.ask_json(image,
            layout +
            'Locate only the Railway medical examiner doctor signature and stamp. '
            'Exclude candidate/witness signatures, printed labels and clinical handwriting. '
            f'This image is {sent.width} pixels wide and {sent.height} pixels high. '
            'Return {"coordinate_space":"pixels", "signature":[x1,y1,x2,y2] or null, '
            '"stamp":[x1,y1,x2,y2] or null, "doctor_region":[x1,y1,x2,y2] or null, '
            '"same_doctor":true or false, '
            '"multiple_doctors":true or false}. Use absolute pixel coordinates in THIS image. '
            'Preserve both regions even if they overlap. Do not transcribe the signature as text. '
            'Use tight boxes; null only for an absent or unidentifiable region. '
            'Include ALL handwritten signature strokes even when they cross printed stamp/form text. '
            'The stamp box must include the English and Hindi name lines and designation/location line. '
            'Do not stop at form borders: stamped text can extend outside them. '
            'doctor_region must be ONE rectangular crop enclosing the entire examiner signature, '
            'both stamp name lines and the designation/location line (such as DMO/BRC or Sr. DMO/BRC), '
            'with a small margin. Preserve long upper and lower pen strokes. '
            'Do not include candidate signature/thumbprint, photograph or unrelated clinical notes. '
            'Nearby examiner-label text inside this rectangle is allowed. Do not crop only the English name. '
            'same_doctor must be true only for an unambiguous signature/stamp pair.',
            long_edge=self.cfg.detection_long_edge)
        space = obj.get("coordinate_space", "normalized")
        for kind in ("signature", "stamp", "doctor_region"):
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
                localization_method = "detected"
                localization_notes = []
                try:
                    obj = self.locate(image, page_index=idx)
                except (ValueError, TypeError) as exc:
                    obj = {}
                    errors.append(f"Page {idx+1}: Full-page localization failed: {exc}")
                    localization_notes.append(f"Full-page localization failed: {exc}")
                if self.cfg.retry_empty_detection and all(crop_box(image, obj.get(k)) is None for k in ("signature", "stamp")):
                    region = (self.cfg.layout_search_regions.get(idx)
                              if self.cfg.layout_fallback_enabled else None)
                    if region is None:
                        region = [0, .35, 1, 1]
                    bounds = (round(region[0]*image.width),round(region[1]*image.height),
                              round(region[2]*image.width),round(region[3]*image.height))
                    local = image.crop(bounds)
                    try:
                        retry = self.locate(local, page_index=idx)
                        retry["multiple_doctors"] = (retry.get("multiple_doctors") is True or obj.get("multiple_doctors") is True)
                        for kind in ("signature", "stamp", "doctor_region"):
                            box = retry.get(kind)
                            # A stamp can occupy most of a close-up. The 25% rule
                            # applies to the full report, not the search window.
                            if crop_box(local, box, max_area=1) is not None:
                                raw_box = [(bounds[0]+box[0]*local.width)/image.width,
                                           (bounds[1]+box[1]*local.height)/image.height,
                                           (bounds[0]+box[2]*local.width)/image.width,
                                           (bounds[1]+box[3]*local.height)/image.height]
                                retry[kind] = raw_box if crop_box(image,raw_box) is not None else None
                            else:
                                retry[kind] = None
                        # Preserve any valid full-page detection when refinement
                        # cannot replace it. Pairing must refer to one view only.
                        if any(crop_box(image,retry.get(k)) is not None for k in ("signature","stamp")):
                            obj = retry
                            localization_method = "region_refinement"
                    except (ValueError, TypeError) as exc:
                        localization_notes.append(f"Close-up localization failed: {exc}")
                if (self.cfg.layout_fallback_enabled and idx in self.cfg.layout_search_regions
                    and all(crop_box(image,obj.get(k)) is None for k in ("signature","stamp","doctor_region"))):
                    obj["doctor_region"] = self.cfg.layout_search_regions[idx]
                    obj["same_doctor"] = False
                    localization_method = "layout_search"
                    localization_notes.append("Configured RRB examiner search region; individual signature/stamp boundaries were NOT detected")
                multiple_doctors |= obj.get("multiple_doctors") is True
                sig, stamp = crop_box(image, obj.get("signature")), crop_box(image, obj.get("stamp"))
                item = save_evidence(image, {"signature": sig, "stamp": stamp}, obj, idx, out_dir, self.root, padding=self.cfg.combined_padding)
                evidence.append(item)
                item.localization_method = localization_method
                item.notes.extend(localization_notes)
                if not item.boxes:
                    item.notes.append("No valid doctor boxes detected; inspect response JSON files in this page directory")
                log.info("doc %s page %d: signature_saved=%s stamp_saved=%s diagnostics=%s",
                         document_id[:12], idx+1, bool(item.signature_path), bool(item.stamp_path), self.diagnostic_dir)
                item.same_doctor = obj.get("same_doctor") is True
                item.signature_usable = sig is not None and usable(sig, self.cfg)
                item.stamp_usable = stamp is not None and usable(stamp, self.cfg)
                matching_signature = None
                if sig is not None:
                    try:
                        isolation = self.ask_json(sig,
                            'Identify every PRINTED or STAMPED text line and form rule in this signature crop. '
                            'Do not box freehand signature strokes as text. '
                            'Return {"coordinate_space":"normalized", "text_boxes":[[x1,y1,x2,y2],...], '
                            '"complete":true or false, "handwriting_present":true or false}. '
                            'Coordinates MUST be normalized 0..1 relative to THIS CROP. '
                            'complete means all printed/stamped text and lines are boxed. '
                            'Use [] only if there is no printed text. Do not invent or reconstruct handwriting.')
                        boxes = isolation.get("text_boxes")
                        if isolation.get("coordinate_space", "normalized") != "normalized":
                            raise ValueError("Signature isolation needs normalized coordinates")
                        overlap = overlapping_stamp(obj["signature"], obj["stamp"] if stamp is not None else None)
                        cleaned, mask, fraction = isolate_signature(sig, boxes, overlap)
                        item.signature_text_boxes = boxes + ([overlap] if overlap else [])
                        item.signature_retained_ink_ratio = fraction
                        item.signature_isolation_complete = (isolation.get("complete") is True
                                                             and isolation.get("handwriting_present") is True)
                        matching_path = self.diagnostic_dir / "signature_matching.png"
                        mask_path = self.diagnostic_dir / "signature_mask.png"
                        cleaned.save(matching_path)
                        mask.save(mask_path)
                        item.signature_matching_path = matching_path.relative_to(self.root).as_posix()
                        item.signature_mask_path = mask_path.relative_to(self.root).as_posix()
                        item.signature_usable = (item.signature_isolation_complete and fraction >= .1
                                                 and usable(cleaned, self.cfg))
                        if item.signature_usable:
                            matching_signature = cleaned
                        if overlap:
                            item.notes.append("Overlapping original retained; stamp pixels excluded from signature matching")
                    except Exception as exc:
                        item.signature_usable = False
                        item.notes.append(f"Signature isolation failed: {exc}")
                    if not item.signature_usable:
                        item.notes.append("Original signature saved; insufficient isolated handwriting for automatic matching")
                if stamp is not None and not item.stamp_usable:
                    item.notes.append("Stamp is not suitable for OCR/enrollment")
                # OCR failure must never discard crops or the signature candidate.
                candidate = dict(idx=idx, signature=matching_signature, raw_signature=sig,
                                 stamp=stamp if item.stamp_usable else None, name=None, text=None,
                                 paired=item.same_doctor, boxes=item.boxes, evidence=item)
                candidates.append(candidate)
                name, text = None, None
                # Reading a stamp name must not depend on finding a tight
                # signature/stamp box. OCR the enclosing examiner region when
                # necessary, while preventing fallback-only gallery enrollment.
                ocr_image = stamp
                item.stamp_ocr_path = item.stamp_path
                if (ocr_image is None and item.combined_path
                    and crop_box(image, obj.get("doctor_region")) is not None):
                    with Image.open(self.root / item.combined_path) as saved:
                        ocr_image = saved.convert("RGB")
                    item.stamp_ocr_path = item.combined_path
                    item.notes.append("Stamp OCR used enclosing examiner region; not a separately detected stamp crop")
                if ocr_image is not None:
                    first = self.ask_json(ocr_image,
                        'Read only the printed doctor name in the STAMP within this examiner region. '
                        'Ignore candidate names, candidate signatures and handwritten notes. Return {"text": "verbatim visible stamp text", '
                        '"doctor_name": "personal name from the English line when available" or null}. '
                        'Preserve initials; never expand them. Do not concatenate Hindi and English names. '
                        'Do not infer missing letters '
                        'or use a designation/hospital as a name. Unreadable personal name must be null.')
                    item.stamp_text = first.get("text") if isinstance(first.get("text"), str) else None
                    second = self.ask_json(ocr_image,
                        'Read only the fully legible personal name printed inside this stamp. '
                        'Return {"doctor_name": "name" or null}. Prefer the English personal name line when available. '
                        'Never concatenate bilingual versions, complete initials or guess letters.')
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
                log.info("doc %s page %d: stamp_name=%s OCR_source=%s localization=%s",
                         document_id[:12],idx+1,name,item.stamp_ocr_path,item.localization_method)
            except Exception as exc:
                errors.append(f"Page {idx+1}: {type(exc).__name__}: {exc}")
                log.warning("doc %s page %d: %s", document_id[:12], idx+1, exc)
        names = {name_key(c["name"]) for c in candidates if c["name"]}
        candidates.sort(key=lambda c: (c["signature"] is not None,
                                      bool(c["name"] and c["paired"]),
                                      bool(c["name"]),
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
        elif (len(signatures) == 2 and not all(c["name"] and c["paired"] for c in signatures)
              and similarity(signatures[0]["signature"], signatures[1]["signature"], self.cfg) < self.cfg.similarity_threshold):
            result = DoctorRecognition(status="needs_review", notes=["Page 3/4 signatures disagree or cannot be reliably compared"])
        elif best["signature"] is not None:
            # Check any readable stamp name for contradictions, even if unpaired.
            observed_name = next((c["name"] for c in candidates if c["name"]), None)
            result = self.registry.identify(best["signature"], best["stamp"] if best["paired"] and best["name"] else None,
                                            observed_name, document_id, raw_signature=best["raw_signature"])
        elif best["evidence"].signature_path:
            result = DoctorRecognition(status="needs_review", notes=["Signature extracted but unsuitable for matching"])
        selected = best["evidence"]
        for field in ("signature_path", "stamp_path", "combined_path", "annotated_path", "page_index", "boxes", "stamp_text"):
            setattr(result, field, getattr(selected, field))
        if result.status == "missing_signature" and errors and not any(c["evidence"].boxes for c in candidates):
            result.status = "error"
        result.stamp_doctor_name = selected.doctor_name or next((c["name"] for c in candidates if c["name"]), None)
        if result.status == "missing_signature" and result.stamp_doctor_name:
            result.status = "stamp_read"
            result.doctor_name = None
        for candidate in candidates:
            if candidate["name"]:
                saved_id = self.registry.save_stamp_name(candidate["name"], document_id,
                                                          candidate["evidence"], self.root)
                if name_key(candidate["name"]) == name_key(result.stamp_doctor_name) and saved_id:
                    result.stamp_name_id = saved_id
                    result.stamp_name_saved = True
        result.evidence = evidence
        result.notes.extend(errors)
        log.info("doc %s doctor: status=%s recognized_name=%s stamp_name=%s similarity=%.3f signature=%s stamp=%s notes=%s",
                 document_id[:12], result.status, result.doctor_name, result.stamp_doctor_name, result.confidence,
                 result.signature_path, result.stamp_path, result.notes)
        return result
