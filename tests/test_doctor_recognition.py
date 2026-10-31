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
                                 RecognitionConfig(registry_dir='gallery'))
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
