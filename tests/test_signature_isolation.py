from io import BytesIO
import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.extraction.signature_isolation import isolate_signature, overlapping_stamp
from src.extraction.doctor_recognition import DoctorRegistry, RecognitionConfig


def test_stamp_and_form_text_masked_original_preserved():
    original = Image.new('RGB',(400,200),'white')
    draw = ImageDraw.Draw(original)
    draw.line([(20,20),(80,75),(150,30),(220,180)],fill='black',width=4)
    draw.text((20,130),'DR. EXAMPLE STAMP',fill='black')
    before = np.asarray(original).copy()
    cleaned,mask,fraction = isolate_signature(original,[[0,.60,1,.80]])
    assert np.array_equal(np.asarray(original),before)
    assert cleaned.mode == 'RGBA'
    assert cleaned.convert('RGB').getpixel((30,140)) == (255,255,255)
    assert mask.getpixel((30,140)) == 0
    assert 0 < fraction < 1
    buffer=BytesIO();cleaned.save(buffer,format='PNG');buffer.seek(0)
    assert Image.open(buffer).getchannel('A').getpixel((30,140)) == 0


def test_stamp_intersection_is_in_signature_coordinates():
    assert overlapping_stamp([.2,.1,.6,.6],[.1,.4,.8,.8]) == pytest.approx([0,.6,1,1])
    assert overlapping_stamp([.1,.1,.2,.2],[.3,.3,.4,.4]) is None


def test_malformed_text_masks_fail_closed():
    for boxes in (None,[[0,0,float('nan'),1]], [[0,.5,1,.2]], [[-1,0,1,1]]):
        with pytest.raises(ValueError):
            isolate_signature(Image.new('RGB',(100,100),'white'),boxes)


def test_legacy_raw_gallery_not_used_for_identification(tmp_path):
    registry = DoctorRegistry(tmp_path,RecognitionConfig())
    buffer=BytesIO();Image.new('RGB',(100,100),'white').save(buffer,format='PNG')
    with registry.connect() as db:
        db.execute('INSERT INTO doctors(id,name,name_key,signature,stamp,source_document) VALUES(?,?,?,?,?,?)',
                   ('legacy','Dr. A Robin','a robin',buffer.getvalue(),buffer.getvalue(),'original'))
    signature = Image.new('RGB',(100,100),'white')
    assert registry.identify(signature,None,None,'old-report').status == 'unmatched'
    result=registry.identify(signature,signature,'Dr. A Robin','original')
    assert result.status == 'registered'
    assert result.doctor_id == 'legacy'
