import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from src.database.migrations import allow_missing_roll_numbers


def test_upgrade_keeps_rows_references_and_unique_indexes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path/'legacy.sqlite3'}")
    with engine.begin() as db:
        db.exec_driver_sql('CREATE TABLE candidates (id INTEGER PRIMARY KEY, roll_number VARCHAR(50) NOT NULL, candidate_name TEXT)')
        db.exec_driver_sql('CREATE UNIQUE INDEX ix_candidates_roll_number ON candidates(roll_number)')
        db.exec_driver_sql('CREATE TABLE examinations (id INTEGER PRIMARY KEY, candidate_id INTEGER REFERENCES candidates(id))')
        db.exec_driver_sql("INSERT INTO candidates VALUES(1,'ROLL1','SYNTHETIC')")
        db.exec_driver_sql('INSERT INTO examinations VALUES(1,1)')
    allow_missing_roll_numbers(engine)
    allow_missing_roll_numbers(engine)
    assert next(c for c in inspect(engine).get_columns('candidates') if c['name']=='roll_number')['nullable']
    with engine.begin() as db:
        assert db.exec_driver_sql('SELECT candidate_id FROM examinations').scalar_one() == 1
        assert db.exec_driver_sql('SELECT candidate_name FROM candidates').scalar_one() == 'SYNTHETIC'
        assert db.exec_driver_sql('PRAGMA foreign_key_check').all() == []
        assert db.exec_driver_sql('PRAGMA foreign_keys').scalar_one() == 1
        db.exec_driver_sql("INSERT INTO candidates VALUES(2,NULL,'NO ROLL')")
    with pytest.raises(IntegrityError), engine.begin() as db:
        db.exec_driver_sql("INSERT INTO candidates VALUES(3,'ROLL1','DUPLICATE')")
    engine.dispose()
