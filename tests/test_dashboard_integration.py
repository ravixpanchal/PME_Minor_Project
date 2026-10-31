import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.models.fields import iter_leaves
from src.pme.importer import import_directory
from src.pme.models import Base, Candidate, ExtractionRun, MedicalExamination
from src.utils.config_loader import config


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
        other = record('SECOND')
        other['document_id'] = 'doc2'
        (target/'other.json').write_text(json.dumps(other))
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
