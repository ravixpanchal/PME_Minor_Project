from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.extraction.doctor_recognition import (
    DoctorRecognizer, DoctorRegistry, RecognitionConfig, crop_box, name_key,
    similarity, usable, valid_name,
)
from src.models.schemas import ExtractionRecord


def ink(seed=1):
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(seed)
    for _ in range(30):
        points = [(int(x), int(y)) for x, y in rng.integers([20, 20], [480, 160], (3, 2))]
        draw.line(points, fill="black", width=2)
    return image


def test_crop_rejects_invalid_coordinates():
    image = ink()
    for box in ([0, 0, 1, 1], [0, 0, float('nan'), .2], [-.1, 0, .3, .2],
                [.5, .5, .2, .6], [0, 0, .001, .001], None):
        assert crop_box(image, box) is None
    assert crop_box(image, [.1, .1, .5, .5]).width == 200


def test_quality_and_actual_visual_match():
    cfg = RecognitionConfig()
    assert not usable(Image.new("RGB", (100, 50), "white"), cfg)
    assert usable(ink(), cfg)
    assert similarity(ink(), ink(), cfg) >= cfg.similarity_threshold
    assert similarity(ink(1), ink(2), cfg) < cfg.similarity_threshold


def test_names():
    assert name_key("Dr. A. Kumar") == name_key("A Kumar")
    assert valid_name("Medical Officer") is None
    assert valid_name("unreadable") is None
    assert valid_name("Dr. A Kumar") == "Dr. A Kumar"


def test_persistent_enrollment_match_and_conflict(tmp_path):
    cfg = RecognitionConfig()
    registry = DoctorRegistry(tmp_path, cfg)
    first = registry.identify(ink(), ink(), "Dr. A Kumar", "doc1")
    assert first.status == "registered"
    registry = DoctorRegistry(tmp_path, cfg)
    match = registry.identify(ink(), None, None, "doc2")
    assert match.status == "matched"
    assert match.doctor_name == first.doctor_name
    assert match.doctor_id == first.doctor_id
    assert registry.identify(ink(), ink(), "Dr. B Shah", "doc3").status == "identity_conflict"
    with patch('src.extraction.doctor_recognition.similarity', return_value=.1):
        assert registry.identify(ink(), ink(), "A Kumar", "doc4").status == "needs_review"
        assert registry.identify(ink(), None, None, "doc5").status == "unmatched"
    with registry.connect() as db:
        assert db.execute('SELECT count(*) FROM doctors').fetchone()[0] == 1


def test_ambiguous_matches_do_not_identify(tmp_path):
    registry = DoctorRegistry(tmp_path, RecognitionConfig())
    registry.identify(ink(), ink(), "A Kumar", "doc1")
    with patch('src.extraction.doctor_recognition.similarity', return_value=0):
        registry.identify(ink(2), ink(), "B Shah", "doc2")
    with patch('src.extraction.doctor_recognition.similarity', return_value=.95):
        assert registry.identify(ink(), None, None, "doc3").status == "ambiguous"


def test_backwards_compatible_record():
    assert ExtractionRecord(document_id="old", model="fake", fields={}).doctor_recognition is None


class FakeExtractor:
    def __init__(self, responses):
        self.responses = iter(responses)
    def _ask(self, *args, **kwargs):
        return next(self.responses)


def test_missing_and_malformed_detection(tmp_path):
    path = tmp_path / 'page.png'
    ink().save(path)
    recognizer = DoctorRecognizer(FakeExtractor(['{"signature":null,"stamp":null}']), tmp_path,
                                 RecognitionConfig(registry_dir='gallery', retry_empty_detection=False))
    result = recognizer.process('doc', {2: path}, tmp_path/'doc')
    assert result.status == 'missing_signature'
    assert result.doctor_name is None
    recognizer.extractor = FakeExtractor(['garbage'])
    assert recognizer.process('doc', {3: path}, tmp_path/'doc').status == 'error'


def test_only_doctor_pages_inspected(tmp_path):
    recognizer = DoctorRecognizer(FakeExtractor([]), tmp_path, RecognitionConfig(registry_dir='gallery'))
    assert recognizer.process('doc', {0: Path('never-open')}, tmp_path/'doc').status == 'missing'


def test_localization_ocr_enroll_then_signature_only(tmp_path):
    page = Image.new('RGB', (1200, 600), 'white')
    page.paste(ink(), (50, 50))
    page.paste(ink(2), (650, 50))
    path = tmp_path/'page.png'
    page.save(path)
    detection = '{"signature":[0.04,0.08,0.46,0.39],"stamp":[0.54,0.08,0.96,0.39],"same_doctor":true}'
    extractor = FakeExtractor([detection, '{"doctor_name":"Dr. A Kumar","text":"Dr. A Kumar MBBS"}',
                               '{"doctor_name":"A Kumar"}'])
    cfg = RecognitionConfig(registry_dir='gallery')
    recognizer = DoctorRecognizer(extractor, tmp_path, cfg)
    first = recognizer.process('doc1', {2:path}, tmp_path/'doc1')
    assert first.status == 'registered'
    assert (tmp_path/first.signature_path).is_file()
    assert (tmp_path/first.stamp_path).is_file()
    recognizer = DoctorRecognizer(FakeExtractor(['{"signature":[0.04,0.08,0.46,0.39],"stamp":null}']), tmp_path, cfg)
    second = recognizer.process('doc2', {3:path}, tmp_path/'doc2')
    assert second.status == 'matched'
    assert second.doctor_name == first.doctor_name
    with recognizer.registry.connect() as db:
        assert db.execute('SELECT count(*) FROM doctors').fetchone()[0] == 1


def test_disagreeing_ocr_never_enrolls(tmp_path):
    page = Image.new('RGB', (1200,600), 'white')
    page.paste(ink(), (50,50))
    page.paste(ink(2), (650,50))
    path = tmp_path/'page.png'
    page.save(path)
    responses = ['{"signature":[0.04,0.08,0.46,0.39],"stamp":[0.54,0.08,0.96,0.39],"same_doctor":true}',
                 '{"doctor_name":"A Kumar","text":"A Kumar"}', '{"doctor_name":"B Shah"}']
    recognizer = DoctorRecognizer(FakeExtractor(responses), tmp_path, RecognitionConfig(registry_dir='gallery'))
    assert recognizer.process('doc', {2:path}, tmp_path/'doc').status == 'unmatched'
    with recognizer.registry.connect() as db:
        assert db.execute('SELECT count(*) FROM doctors').fetchone()[0] == 0


def evidence_page(tmp_path):
    page = Image.new('RGB', (1200, 600), 'white')
    page.paste(ink(), (50, 50))
    page.paste(ink(2), (650, 50))
    path = tmp_path/'page.png'
    page.save(path)
    return path


def test_annotation_coordinates_and_combined_crop(tmp_path):
    from src.extraction.doctor_evidence import save_evidence
    image = Image.new('RGB', (1000, 600), 'white')
    boxes = {'signature': [.1, .2, .3, .4], 'stamp': [.4, .3, .6, .5]}
    crops = {kind: crop_box(image, box) for kind, box in boxes.items()}
    item = save_evidence(image, crops, boxes, 2, tmp_path/'doc', tmp_path)
    assert item.boxes['combined'] == [.1, .2, .6, .5]
    assert item.boxes_pixels['combined'] == [100, 120, 600, 300]
    with Image.open(tmp_path/item.combined_path) as combined:
        assert combined.size == (500, 180)
    with Image.open(tmp_path/item.annotated_path) as annotated:
        assert annotated.size == image.size
        assert annotated.getpixel((100, 120)) != (255, 255, 255)


def test_ocr_failure_preserves_extracted_crops(tmp_path):
    path = evidence_page(tmp_path)
    detection = '{"signature":[0.04,0.08,0.46,0.39],"stamp":[0.54,0.08,0.96,0.39],"same_doctor":true}'
    recognizer = DoctorRecognizer(FakeExtractor([detection, 'invalid OCR']), tmp_path,
                                 RecognitionConfig(registry_dir='gallery'))
    result = recognizer.process('doc', {2:path}, tmp_path/'doc')
    assert result.status == 'unmatched'
    assert result.doctor_name is None
    assert len(result.evidence) == 1
    for value in (result.signature_path, result.stamp_path, result.annotated_path, result.combined_path):
        assert (tmp_path/value).is_file()
    assert result.notes


def test_overlap_saved_but_never_matched(tmp_path):
    path = evidence_page(tmp_path)
    detection = '{"signature":[0.04,0.08,0.46,0.39],"stamp":[0.20,0.08,0.46,0.39],"same_doctor":true}'
    recognizer = DoctorRecognizer(FakeExtractor([detection, '{"doctor_name":null}', '{"doctor_name":null}']),
                                 tmp_path, RecognitionConfig(registry_dir='gallery'))
    result = recognizer.process('doc', {2:path}, tmp_path/'doc')
    assert result.status == 'needs_review'
    assert result.signature_path and result.stamp_path
    assert not result.evidence[0].signature_usable
    assert result.evidence[0].notes
    with recognizer.registry.connect() as db:
        assert db.execute('SELECT count(*) FROM doctors').fetchone()[0] == 0


def test_multiple_doctors_preserves_both_pages(tmp_path):
    path = evidence_page(tmp_path)
    detection = '{"signature":[0.04,0.08,0.46,0.39],"stamp":null,"multiple_doctors":true}'
    recognizer = DoctorRecognizer(FakeExtractor([detection, detection]), tmp_path,
                                 RecognitionConfig(registry_dir='gallery'))
    result = recognizer.process('doc', {2:path,3:path}, tmp_path/'doc')
    assert result.status == 'ambiguous'
    assert len(result.evidence) == 2
    assert all((tmp_path/e.annotated_path).is_file() for e in result.evidence)


def test_reject_invalid_evidence_contract():
    from pydantic import ValidationError
    from src.models.schemas import DoctorPageEvidence
    with pytest.raises(ValidationError):
        DoctorPageEvidence(page_index=2, image_width=100, image_height=100,
                           boxes={'signature':[0,0,float('nan'),.5]})


def test_declared_pixel_boxes_are_scaled_to_raw_image(tmp_path):
    path = evidence_page(tmp_path)
    detection = '{"coordinate_space":"pixels","signature":[84,84,924,392],"stamp":null}'
    recognizer = DoctorRecognizer(FakeExtractor([detection]),tmp_path,RecognitionConfig(registry_dir='gallery'))
    result = recognizer.process('doc',{2:path},tmp_path/'doc')
    assert result.evidence[0].boxes['signature'] == pytest.approx([84/2016,84/1008,924/2016,392/1008])
    assert (tmp_path/result.signature_path).is_file()
    response = tmp_path/'doc/doctor_evidence/page_0002/response_01.json'
    assert 'coordinate_space' in response.read_text()


def test_empty_full_page_retries_lower_region_and_remaps_boxes(tmp_path):
    path = evidence_page(tmp_path)
    responses = ['{"signature":null,"stamp":null}',
                 '{"coordinate_space":"normalized","signature":[0.1,0.1,0.4,0.3],"stamp":null}']
    recognizer = DoctorRecognizer(FakeExtractor(responses),tmp_path,RecognitionConfig(registry_dir='gallery'))
    result = recognizer.process('doc',{2:path},tmp_path/'doc')
    assert result.evidence[0].boxes['signature'] == pytest.approx([.1,.415,.4,.545])
    assert (tmp_path/'doc/doctor_evidence/page_0002/response_02.json').is_file()
