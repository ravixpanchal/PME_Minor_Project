from __future__ import annotations

from pathlib import Path
from .config import get_extraction_dir

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    Candidate,
    Declaration,
    Doctor,
    DoctorEvidence,
    DoctorRecognitionResult,
    DoctorRegistryLink,
    ExtractionRun,
    FitnessClassification,
    MedicalExamination,
    PhysicalExamination,
    PmeCase,
    PmeCaseMeasurement,
    VisionExamination,
)
from .schemas import PMEExtraction


def clean(value):
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def parse_metrics(text):
    output = {}

    if not text:
        return output

    for item in text.split(";"):
        if ":" not in item:
            continue

        key, value = item.split(":", 1)

        key = key.strip().lower()
        value = value.strip()

        if key:
            output[key] = value

    return output


def store_doctor_result(session, run, recognition):
    """Backfill or replace queryable evidence, retaining raw JSON for audit."""
    if run.doctor_result is not None:
        run.doctor_result = None
        session.flush()
    if recognition is None:
        return
    identity = None
    if recognition.doctor_id and recognition.status in {"registered", "matched"}:
        identity = session.get(DoctorRegistryLink, recognition.doctor_id)
        if identity is None and run.examination.examining_doctor is not None:
            doctor = run.examination.examining_doctor
            bound = session.scalar(select(DoctorRegistryLink).where(
                DoctorRegistryLink.doctor_id == doctor.id))
            if bound is not None:
                # Legacy imports may have merged namesakes; split the mapping
                # during backfill without modifying the other examination.
                doctor = Doctor(doctor_name=recognition.doctor_name or doctor.doctor_name,
                                doctor_role=doctor.doctor_role)
                session.add(doctor)
                run.examination.examining_doctor = doctor
                session.flush()
            identity = DoctorRegistryLink(registry_id=recognition.doctor_id, doctor=doctor)
            session.add(identity)
            session.flush()
    result = DoctorRecognitionResult(
        registry_id=identity.registry_id if identity else None,
        status=recognition.status, similarity=recognition.confidence,
        algorithm=recognition.algorithm, observed_name=recognition.doctor_name,
        notes=recognition.notes)
    run.doctor_result = result
    result.evidence = [DoctorEvidence(**item.model_dump()) for item in recognition.evidence]
    session.flush()


def import_one(session: Session, data: PMEExtraction):
    # ---------------------------------------------------------
    # Prevent duplicate imports
    # ---------------------------------------------------------
    existing = session.scalar(
        select(ExtractionRun).where(
            ExtractionRun.document_id == data.document_id
        )
    )

    previous_candidate = existing.examination.candidate if existing else None
    if existing:
        if existing.raw_json == data.model_dump(mode="json"):
            if existing.doctor_result is None and data.doctor_recognition is not None:
                store_doctor_result(session, existing, data.doctor_recognition)
            return existing.examination, False
        # Replace dependent clinical data while retaining examination/run IDs.
        examination = existing.examination
        for relation in ("pme_case", "vision", "physical", "fitness"):
            setattr(examination, relation, None)
        examination.declarations.clear()
        examination.examining_doctor = None
        session.flush()

    # ---------------------------------------------------------
    # Extract pages
    # ---------------------------------------------------------
    page1 = data.page1
    page2 = data.page2
    page3 = data.page3
    page4 = data.page4

    general = page1.general_details
    narrative = page1.narrative_details
    physical_traits = page1.physical_traits

    # ---------------------------------------------------------
    # Candidate validation
    # ---------------------------------------------------------
    roll_number = clean(narrative.roll_number)
    if not roll_number and not clean(general.candidate_name):
        raise ValueError(
            f"{data.document_id}: candidate_name or roll_number is required"
        )

    # ---------------------------------------------------------
    # Find or create candidate
    # ---------------------------------------------------------
    candidate = previous_candidate or (session.scalar(select(Candidate).where(
        Candidate.roll_number == roll_number)) if roll_number else None)

    if not candidate:
        candidate = Candidate(
            roll_number=roll_number,
            candidate_name=clean(general.candidate_name),
            father_name=clean(general.father_name),
            date_of_birth=general.date_of_birth,
            mobile_number=clean(general.mobile_number or page4.mobile_number),
            email=clean(general.email or page4.email),
            recruitment_cen=clean(narrative.recruitment_cen),
        )

        session.add(candidate)
        session.flush()

    else:
        candidate_updates = {
            "candidate_name": general.candidate_name,
            "father_name": general.father_name,
            "date_of_birth": general.date_of_birth,
            "mobile_number": general.mobile_number or page4.mobile_number,
            "email": general.email or page4.email,
            "recruitment_cen": narrative.recruitment_cen,
        }

        if roll_number and roll_number != candidate.roll_number:
            owner = session.scalar(select(Candidate).where(Candidate.roll_number == roll_number))
            if owner is not None or candidate.roll_number:
                raise ValueError("Roll number conflicts with stored candidate identity; review required")
            candidate.roll_number = roll_number

        for attribute, value in candidate_updates.items():
            if value in (None, ""):
                continue

            if attribute == "date_of_birth":
                setattr(candidate, attribute, value)
            else:
                setattr(candidate, attribute, clean(value))

        session.flush()

    # ---------------------------------------------------------
    # Medical examination
    # ---------------------------------------------------------
    examination = existing.examination if existing else MedicalExamination(candidate=candidate)
    examination.dv_date = general.dv_date
    examination.medical_examination_date = general.medical_examination_date
    examination.candidate_declaration_date = page4.declaration_date
    examination.medical_class = clean(narrative.class_name)
    examination.identification_marks = clean(physical_traits.permanent_physical_identification_marks)

    session.add(examination)
    session.flush()

    # ---------------------------------------------------------
    # Page 2 - PME Case
    # ---------------------------------------------------------
    pme_case = PmeCase(
        examination=examination,
        urine=clean(page2.pme_case.urine),
        sugar=clean(page2.pme_case.sugar),
        alb=clean(page2.pme_case.alb),
        handwritten_remarks=clean(page2.handwritten_remarks),
        chest_xray_result=clean(page2.chest_xray_result),
        advice=clean(page2.advice),
    )

    session.add(pme_case)
    session.flush()

    # ---------------------------------------------------------
    # Page 2 - vision/table rows
    # ---------------------------------------------------------
    for row in page2.right_side_table:
        measurement = PmeCaseMeasurement(
            pme_case_id=pme_case.id,
            row_label=clean(row.row) or "UNKNOWN",
            column_1=clean(row.side_r),
            column_2=clean(row.side_l),
        )

        session.add(measurement)

    # ---------------------------------------------------------
    # Page 3 - handwritten metrics
    # ---------------------------------------------------------
    metrics = parse_metrics(
        page3.handwritten_text_below_vision_table
    )

    # ---------------------------------------------------------
    # Page 3 - eyes
    # ---------------------------------------------------------
    right_eye = page3.accuracy_of_vision.get("right_eye")
    left_eye = page3.accuracy_of_vision.get("left_eye")

    vision = VisionExamination(
        examination=examination,

        right_distance_uncorrected=(
            right_eye.distance_uncorrected
            if right_eye
            else None
        ),

        right_distance_corrected=(
            right_eye.distance_corrected
            if right_eye
            else None
        ),

        right_near_uncorrected=(
            right_eye.near_uncorrected
            if right_eye
            else None
        ),

        right_near_corrected=(
            right_eye.near_corrected
            if right_eye
            else None
        ),

        right_glasses_s=(
            right_eye.glasses_power_s
            if right_eye
            else None
        ),

        right_glasses_c=(
            right_eye.glasses_power_c
            if right_eye
            else None
        ),

        right_glasses_a=(
            right_eye.glasses_power_a
            if right_eye
            else None
        ),

        left_distance_uncorrected=(
            left_eye.distance_uncorrected
            if left_eye
            else None
        ),

        left_distance_corrected=(
            left_eye.distance_corrected
            if left_eye
            else None
        ),

        left_near_uncorrected=(
            left_eye.near_uncorrected
            if left_eye
            else None
        ),

        left_near_corrected=(
            left_eye.near_corrected
            if left_eye
            else None
        ),

        left_glasses_s=(
            left_eye.glasses_power_s
            if left_eye
            else None
        ),

        left_glasses_c=(
            left_eye.glasses_power_c
            if left_eye
            else None
        ),

        left_glasses_a=(
            left_eye.glasses_power_a
            if left_eye
            else None
        ),

        color_perception=clean(
            page3.color_perception
            or metrics.get("colour perception")
            or metrics.get("color perception")
        ),

        binocular_vision=clean(
            page3.binocular_vision
        ),

        night_vision=clean(
            page3.night_vision
            or metrics.get("night vision")
        ),

        field_of_vision=clean(
            page3.field_of_vision
        ),
    )

    session.add(vision)

    # ---------------------------------------------------------
    # Page 3 - physical examination
    # ---------------------------------------------------------
    physical = PhysicalExamination(
        examination=examination,

        urine=clean(
            page3.urine
            or metrics.get("urine")
        ),

        hearing=clean(
            page3.hearing
            or metrics.get("hearing")
        ),

        pr=clean(
            page3.pr
            or metrics.get("pr")
        ),

        bp=clean(
            page3.bp
            or metrics.get("bp")
        ),

        cvs=clean(page3.cvs),
        rs=clean(page3.rs),
        pia=clean(page3.pia),
        spine=clean(page3.spine),

        heart=clean(
            page3.heart
            or metrics.get("heart")
        ),

        lungs=clean(
            page3.lungs
            or metrics.get("lungs")
        ),

        general_physical_examination=clean(
            page3.general_physical_examination
        ),
    )

    session.add(physical)

    # ---------------------------------------------------------
    # Page 3 - fitness classification
    # ---------------------------------------------------------
    fitness = FitnessClassification(
        examination=examination,
        fit_in_class=clean(page3.fit_in_class),
        unfit_in_class=clean(page3.unfit_in_class),
    )

    session.add(fitness)

    # ---------------------------------------------------------
    # Page 4 - declarations
    # ---------------------------------------------------------
    declaration_groups = (
        ("Part 1", page4.part_one),
        ("Part 2", page4.part_two),
    )

    for part_name, items in declaration_groups:
        for item in items:
            declaration = Declaration(
                examination=examination,
                part=part_name,
                question_no=(
                    clean(item.question_no)
                    or "UNKNOWN"
                ),
                answer=clean(item.answer),
            )

            session.add(declaration)

    # ---------------------------------------------------------
    # Doctor
    # ---------------------------------------------------------
    recognition = data.doctor_recognition
    doctor_name = (recognition.doctor_name if recognition and recognition.status in
                   {"matched", "registered", "missing_signature"} else None)
    if recognition is None:  # Legacy exports.
        doctor_name = clean(page3.doctor_name or page4.doctor_name)

    if doctor_name:
        link = session.get(DoctorRegistryLink, recognition.doctor_id) if recognition and recognition.doctor_id else None
        statement = select(Doctor).where(
            Doctor.doctor_name == doctor_name,
            Doctor.doctor_role == "Railway Medical Examiner")
        if recognition and recognition.doctor_id:
            # Never collapse different gallery identities merely because names agree.
            statement = statement.where(~Doctor.id.in_(select(DoctorRegistryLink.doctor_id)))
        doctor = link.doctor if link else session.scalar(statement)

        if not doctor:
            doctor = Doctor(
                doctor_name=doctor_name,
                doctor_role="Railway Medical Examiner",
            )

            session.add(doctor)
            session.flush()

        examination.examining_doctor = doctor

    # ---------------------------------------------------------
    # Extraction run
    # ---------------------------------------------------------
    extracted_at = next(
        (
            provenance.extracted_at
            for provenance in data.provenance.values()
            if provenance.extracted_at
        ),
        None,
    )

    extraction_run = existing or ExtractionRun(examination=examination, document_id=data.document_id)
    extraction_run.model_name = data.model
    extraction_run.extracted_at = extracted_at
    extraction_run.raw_json = data.model_dump(mode="json")
    session.add(extraction_run)
    session.flush()
    store_doctor_result(session, extraction_run, recognition)

    return examination, True


def import_directory(
    session: Session,
    directory: Path,
):
    successful = 0
    failed = 0

    for path in sorted(directory.rglob("*.json")):
        # Sidecars are diagnostics, not complete extraction records.
        if path.name == "document.json" or path.name.startswith("page_"):
            continue
        try:
            with session.begin_nested():
                _, created = import_one(
                    session,
                    PMEExtraction.from_file(path),
                )

            session.commit()

            successful += 1

            status = (
                "imported"
                if created
                else "already imported"
            )

            print(
                f"[OK] {path.name} {status}"
            )

        except Exception as error:
            session.rollback()
            failed += 1

            print(
                f"[ERROR] {path.name}: {error}"
            )

    return successful, failed


def sync_extractions(
    session: Session,
    directory: Path | None = None,
) -> tuple[int, int]:
    """Import every extraction currently available in the pipeline output."""
    extraction_dir = directory or get_extraction_dir()
    if not extraction_dir.exists():
        raise FileNotFoundError(
            f"Extraction directory does not exist: {extraction_dir}"
        )
    return import_directory(session, extraction_dir)