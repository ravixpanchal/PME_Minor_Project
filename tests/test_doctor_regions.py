import pytest
from PIL import Image
from src.extraction.doctor_evidence import combined_doctor_box, save_evidence
from src.extraction.run_extraction import select_documents


def test_combined_block_contains_full_signature_and_stamp():
    parts={'signature':[.6,.65,.8,.88],'stamp':[.5,.8,.95,.93]}
    region=[.49,.64,.96,.94]
    assert combined_doctor_box(parts,region) == region
    # A box containing only the name must not clip off the signature's top.
    box=combined_doctor_box(parts,[.5,.8,.95,.93])
    assert box == pytest.approx([.492,.642,.958,.938])


def test_bad_combined_block_falls_back_and_clamps_margin():
    assert combined_doctor_box({'signature':[0,.8,.2,1]},[0,0,1,1]) == pytest.approx([0,.792,.208,1])
    assert combined_doctor_box({},[0,0,float('nan'),1]) is None


def test_combined_region_can_be_saved_without_inventing_split_boxes(tmp_path):
    item=save_evidence(Image.new('RGB',(1000,1400),'white'),
                       {'signature':None,'stamp':None},{'doctor_region':[.5,.7,.9,.95]},
                       2,tmp_path/'doc',tmp_path)
    assert item.signature_path is None and item.stamp_path is None
    assert (tmp_path/item.combined_path).is_file()
    assert (tmp_path/item.annotated_path).is_file()
    assert item.boxes == {'combined':[.5,.7,.9,.95]}


def test_mixed_doctor_trial_selects_requested_manifest_positions():
    ids=[f'doc{i}' for i in range(1,51)]
    assert select_documents(ids,[1,10,27]) == ['doc1','doc10','doc27']
    for bad in ([0],[51],[1,1]):
        with pytest.raises(ValueError):
            select_documents(ids,bad)


def test_rejected_large_union_never_discards_independent_crops(tmp_path):
    image=Image.new('RGB',(1000,1400),'white')
    boxes={'signature':[.05,.05,.25,.25],'stamp':[.75,.75,.95,.95]}
    crops={kind:image.crop((int(box[0]*1000),int(box[1]*1400),int(box[2]*1000),int(box[3]*1400))) for kind,box in boxes.items()}
    item=save_evidence(image,crops,boxes,2,tmp_path/'doc',tmp_path)
    assert item.combined_path is None
    assert (tmp_path/item.signature_path).is_file()
    assert (tmp_path/item.stamp_path).is_file()
