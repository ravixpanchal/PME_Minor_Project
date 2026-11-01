import pytest
import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.models.fields import iter_leaves
from src.database.importer import import_directory
from src.database.models import Base, Candidate, ExtractionRun, MedicalExamination
from src.utils.config_loader import config


@pytest.fixture(autouse=True)
def rendered_pages(monkeypatch, tmp_path):
    from PIL import Image
    from src.database import importer
    directory = tmp_path / "images"
    directory.mkdir()
    pages = []
    for idx in range(4):
        path = directory / f"page_{idx:04d}.png"
        Image.new("RGB", (80, 100), "white").save(path)
        pages.append((idx+1, path))
    monkeypatch.setattr(importer, "find_report_images", lambda *args, **kwargs: pages)


def record(name="TEST CANDIDATE", status="registered"):
    return dict(document_id='doc1', model='fake', fields={
        'page_1_basic_info': {'general_details': {'candidate_name': name, 'father_name': 'FATHER', 'date_of_birth': '21-02-1996'}},
        'page_3_fitness_classification': {'bp': '120/80', 'fit_in_class': 'A1'},
        'page_4_declarations': {'email': 'test@example.com', 'mobile_number': '1234567890', 'declaration_date': '22-02-2025'},
    }, doctor_recognition={'status': status, 'doctor_name': 'Dr. A Kumar'})


def test_record_sidecars_and_refresh(tmp_path):
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    target = tmp_path/'doc1'
    target.mkdir()
    (target/'record.json').write_text(json.dumps(record()))
    (target/'document.json').write_text('{"fields": {}}')
    (target/'page_0000.json').write_text('{"fields": {}}')
    with Session(engine) as session:
        assert import_directory(session, tmp_path) == (1, 0)
        assert import_directory(session, tmp_path) == (1, 0)
        exam = session.scalar(select(MedicalExamination))
        assert exam.physical.bp == '120/80'
        assert exam.examining_doctor.doctor_name == 'Dr. A Kumar'
        assert exam.candidate.email == 'test@example.com'
        # No roll number must not merge two unrelated candidates via NULL == NULL.
        (tmp_path/'doc2').mkdir()
        other = record('SECOND')
        other['document_id'] = 'doc2'
        (tmp_path/'doc2'/'record.json').write_text(json.dumps(other))
        assert import_directory(session, tmp_path) == (2, 0)
        assert len(list(session.scalars(select(Candidate)))) == 2
        updated = record(status='identity_conflict')
        updated['fields']['page_3_fitness_classification']['bp'] = '110/70'
        (target/'record.json').write_text(json.dumps(updated))
        assert import_directory(session, tmp_path) == (2, 0)
        run = session.scalar(select(ExtractionRun).where(ExtractionRun.document_id == 'doc1'))
        assert run.examination.physical.bp == '110/70'
        assert run.examination.examining_doctor is None
        assert len(list(session.scalars(select(ExtractionRun)))) == 2
        assert len(list(session.scalars(select(Candidate)))) == 2


def test_dashboard_fields_in_extraction_schema():
    paths = {path for path, _ in iter_leaves(config.schema.fields)}
    for suffix in ('color_perception', 'night_vision', 'heart', 'lungs', 'bp', 'hearing'):
        assert f'page_3_fitness_classification.{suffix}' in paths
    for suffix in ('email', 'mobile_number', 'declaration_date'):
        assert f'page_4_declarations.{suffix}' in paths


def test_additive_schema_upgrade_and_evidence_refresh(tmp_path):
    from sqlalchemy import inspect
    from src.database.models import DoctorEvidence, DoctorRecognitionResult, DoctorRegistryLink
    engine = create_engine('sqlite:///:memory:')
    # Simulate the old schema with populated candidates and no new tables.
    old_tables = [table for table in Base.metadata.sorted_tables
                  if table.name not in {'doctor_registry_links','doctor_recognition_results','doctor_evidence','doctor_signature_isolation'}]
    Base.metadata.create_all(engine, tables=old_tables)
    with Session(engine) as session:
        session.add(Candidate(candidate_name='EXISTING'))
        session.commit()
    Base.metadata.create_all(engine)
    assert 'doctor_evidence' in inspect(engine).get_table_names()
    data = record()
    data['doctor_recognition'].update(doctor_id='registry-id', evidence=[{
        'page_index':2, 'image_width':1000, 'image_height':600,
        'boxes':{'signature':[.1,.1,.3,.3]},
        'boxes_pixels':{'signature':[100,60,300,180]},
        'signature_path':'data/processed/doc/signature.png',
        'annotated_path':'data/processed/doc/annotated.png',
        'signature_usable':True,
        'signature_matching_path':'data/processed/doc/signature_matching.png',
        'signature_isolation_complete':True,
    }])
    target = tmp_path/'record.json'
    target.write_text(json.dumps(data))
    with Session(engine) as session:
        assert import_directory(session,tmp_path) == (1,0)
        run = session.scalar(select(ExtractionRun))
        run_id, exam_id = run.id, run.examination_id
        assert session.get(DoctorRegistryLink,'registry-id').doctor_id == run.examination.examining_doctor_id
        assert session.get(DoctorRecognitionResult,run_id).status == 'registered'
        assert session.scalar(select(DoctorEvidence)).boxes_pixels['signature'] == [100,60,300,180]
        isolated = session.scalar(select(DoctorEvidence)).isolation.payload
        assert isolated['signature_matching_path'] == 'data/processed/doc/signature_matching.png'
        assert isolated['signature_isolation_complete'] is True
        data['doctor_recognition']['evidence'][0]['page_index'] = 3
        data['fields']['page_3_fitness_classification']['bp'] = '110/70'
        target.write_text(json.dumps(data))
        assert import_directory(session,tmp_path) == (1,0)
        run = session.scalar(select(ExtractionRun))
        assert (run.id,run.examination_id) == (run_id,exam_id)
        assert len(list(session.scalars(select(DoctorEvidence)))) == 1
        assert session.scalar(select(DoctorEvidence)).page_index == 3
        assert session.scalar(select(Candidate).where(Candidate.candidate_name=='EXISTING'))
        assert import_directory(session,tmp_path) == (1,0)


def test_identity_roll_conflict_rolls_back_existing_record(tmp_path):
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    target = tmp_path/'record.json'
    original = record()
    target.write_text(json.dumps(original))
    with Session(engine) as session:
        assert import_directory(session,tmp_path) == (1,0)
        run = session.scalar(select(ExtractionRun))
        exam_id = run.examination_id
        # A previously missing roll number can be filled in.
        original['fields']['page_1_basic_info']['narrative_details'] = {'roll_number':'ROLL1'}
        target.write_text(json.dumps(original))
        assert import_directory(session,tmp_path) == (1,0)
        assert session.scalar(select(Candidate)).roll_number == 'ROLL1'
        # A conflicting correction must not partially delete the old clinical data.
        original['fields']['page_1_basic_info']['narrative_details']['roll_number'] = 'ROLL2'
        target.write_text(json.dumps(original))
        assert import_directory(session,tmp_path) == (0,1)
        run = session.scalar(select(ExtractionRun))
        assert run.examination_id == exam_id
        assert run.examination.physical.bp == '120/80'
        assert session.scalar(select(Candidate)).roll_number == 'ROLL1'


def test_distinct_registry_ids_with_same_name_are_not_merged(tmp_path):
    from src.database.models import DoctorRegistryLink
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    (tmp_path/'doc1').mkdir()
    (tmp_path/'doc2').mkdir()
    first = record()
    first['doctor_recognition']['doctor_id'] = 'doctor-one'
    second = record('OTHER CANDIDATE')
    second['document_id'] = 'doc2'
    second['doctor_recognition']['doctor_id'] = 'doctor-two'
    (tmp_path/'doc1'/'record.json').write_text(json.dumps(first))
    (tmp_path/'doc2'/'record.json').write_text(json.dumps(second))
    with Session(engine) as session:
        assert import_directory(session,tmp_path) == (2,0)
        a = session.get(DoctorRegistryLink,'doctor-one')
        b = session.get(DoctorRegistryLink,'doctor-two')
        assert a.doctor_id != b.doctor_id
        assert a.doctor.doctor_name == b.doctor.doctor_name


def test_selection_import_excludes_stale_invalid_records(tmp_path):
    from src.database.selection import selected_documents
    (tmp_path/'doc1').mkdir()
    (tmp_path/'old').mkdir()
    (tmp_path/'doc1/record.json').write_text(json.dumps(record()))
    (tmp_path/'old/record.json').write_text('{}')
    selection = tmp_path/'last_selection.json'
    selection.write_text(json.dumps({'document_ids':['doc1']}))
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert import_directory(session,tmp_path,document_ids=selected_documents(selection)) == (1,0)
        assert len(list(session.scalars(select(ExtractionRun)))) == 1


def test_stamp_ocr_without_signature_is_saved_without_gallery_identity(tmp_path):
    from src.database.models import DoctorRecognitionResult, DoctorEvidence
    data=record(status='stamp_read')
    data['doctor_recognition'].update(doctor_name=None,stamp_doctor_name='Dr. A Robin',evidence=[{
        'page_index':2,'image_width':1200,'image_height':1800,
        'localization_method':'layout_search',
        'stamp_ocr_path':'data/processed/doc/doctor_evidence/page_0002/combined.png',
        'doctor_name':'Dr. A Robin',
    }])
    (tmp_path/'record.json').write_text(json.dumps(data))
    engine=create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert import_directory(session,tmp_path)==(1,0)
        result=session.scalar(select(DoctorRecognitionResult))
        assert result.status=='stamp_read'
        assert result.observed_name=='Dr. A Robin'
        assert result.registry_id is None
        assert result.extraction_run.examination.examining_doctor.doctor_name=='Dr. A Robin'
        assert session.scalar(select(DoctorEvidence)).isolation.payload['localization_method']=='layout_search'
