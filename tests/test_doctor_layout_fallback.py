from PIL import Image, ImageDraw
from src.extraction.doctor_recognition import DoctorRecognizer, RecognitionConfig, crop_box


class Fake:
    def __init__(self,responses):
        self.responses=iter(responses)
    def _ask(self,*args,**kwargs):
        return next(self.responses)


def page(tmp_path):
    im=Image.new('RGB',(1200,1800),'white')
    draw=ImageDraw.Draw(im)
    draw.text((650,1500),'DR. A. ROBIN',fill='black')
    draw.text((650,1540),'DMO / BRC',fill='black')
    path=tmp_path/'report.png'
    im.save(path)
    return path


def config():
    return RecognitionConfig(registry_dir='gallery',layout_fallback_enabled=True,
                             layout_search_regions={2:[.38,.66,.99,.95],3:[.48,.65,.99,.95]})


def test_empty_detections_do_not_prevent_stamp_ocr(tmp_path):
    null='{"signature":null,"stamp":null,"doctor_region":null}'
    first='{"text":"DR. A. ROBIN DMO / BRC","doctor_name":"Dr. A. Robin"}'
    second='{"doctor_name":"A Robin"}'
    recognizer=DoctorRecognizer(Fake([null,null,first,second]),tmp_path,config())
    result=recognizer.process('doc',{2:page(tmp_path)},tmp_path/'doc')
    assert result.status=='stamp_read'
    assert result.stamp_doctor_name=='Dr. A. Robin'
    assert result.doctor_name is None
    assert result.signature_path is None and result.stamp_path is None
    item=result.evidence[0]
    assert item.localization_method=='layout_search'
    assert item.stamp_ocr_path==item.combined_path
    assert (tmp_path/item.stamp_ocr_path).is_file()
    assert (tmp_path/item.annotated_path).is_file()
    with recognizer.registry.connect() as db:
        assert db.execute('SELECT count(*) FROM doctors').fetchone()[0]==0


def test_closeup_stamp_may_cover_most_of_search_window(tmp_path):
    null='{"signature":null,"stamp":null}'
    closeup='{"coordinate_space":"normalized","signature":null,"stamp":[0.05,0.1,0.95,0.95]}'
    reads=['{"doctor_name":"Dr. A Robin","text":"Dr. A Robin"}','{"doctor_name":"A Robin"}']
    recognizer=DoctorRecognizer(Fake([null,closeup,*reads]),tmp_path,config())
    result=recognizer.process('doc',{2:page(tmp_path)},tmp_path/'doc')
    assert result.status=='stamp_read'
    assert result.stamp_path
    assert result.evidence[0].localization_method=='region_refinement'
    box=result.evidence[0].boxes['stamp']
    assert box[0]>=.38 and box[1]>=.66 and box[2]<=.99 and box[3]<=.95


def test_region_without_readable_name_never_fabricates_doctor(tmp_path):
    null='{"signature":null,"stamp":null}'
    recognizer=DoctorRecognizer(Fake([null,null,'{"text":null,"doctor_name":null}','{"doctor_name":null}']),tmp_path,config())
    result=recognizer.process('doc',{2:page(tmp_path)},tmp_path/'doc')
    assert result.stamp_doctor_name is None
    assert result.doctor_name is None
    assert result.doctor_id is None
    assert result.combined_path


def test_area_limit_applies_to_page_not_closeup():
    im=Image.new('RGB',(400,300),'white')
    box=[.05,.05,.95,.95]
    assert crop_box(im,box) is None
    assert crop_box(im,box,max_area=1) is not None
