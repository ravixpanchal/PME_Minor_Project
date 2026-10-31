"""Idempotent upgrade for older databases requiring a readable roll number."""
import re

from sqlalchemy import inspect


def allow_missing_roll_numbers(engine):
    inspector = inspect(engine)
    if not inspector.has_table("candidates"):
        return
    column = next(c for c in inspector.get_columns("candidates") if c["name"] == "roll_number")
    if column["nullable"]:
        return
    dialect = engine.dialect.name
    if dialect != "sqlite":
        statements = {
            "postgresql": "ALTER TABLE candidates ALTER COLUMN roll_number DROP NOT NULL",
            "mysql": "ALTER TABLE candidates MODIFY COLUMN roll_number VARCHAR(50) NULL",
        }
        if dialect not in statements:
            raise RuntimeError(f"Nullable roll-number migration unsupported for {dialect}")
        with engine.begin() as connection:
            connection.exec_driver_sql(statements[dialect])
        return
    # SQLite cannot alter nullability. Rebuild only this table, preserving its
    # original DDL, data, constraints, indexes, triggers and referencing rows.
    with engine.connect() as connection:
        ddl = connection.exec_driver_sql("SELECT sql FROM sqlite_master WHERE type='table' AND name='candidates'").scalar_one()
        extras = connection.exec_driver_sql("SELECT sql FROM sqlite_master WHERE tbl_name='candidates' AND type IN ('index','trigger') AND sql IS NOT NULL").scalars().all()
        columns = [row[1] for row in connection.exec_driver_sql('PRAGMA table_info("candidates")')]
        quoted = ', '.join('"'+name.replace('"','""')+'"' for name in columns)
        replacement, count = re.subn(r'((?:"roll_number"|\broll_number)\s+[^,\n]*?)\s+NOT\s+NULL', r'\1', ddl, count=1, flags=re.I)
        if count != 1:
            raise RuntimeError("Cannot safely identify candidates.roll_number constraint")
        replacement = re.sub(r'^CREATE TABLE\s+(?:"candidates"|candidates)', 'CREATE TABLE "candidates_nullable_upgrade"', replacement, count=1, flags=re.I)
        connection.commit()
        connection.exec_driver_sql('PRAGMA foreign_keys=OFF')
        connection.commit()
        try:
            connection.exec_driver_sql('BEGIN IMMEDIATE')
            connection.exec_driver_sql(replacement)
            connection.exec_driver_sql(f'INSERT INTO candidates_nullable_upgrade ({quoted}) SELECT {quoted} FROM candidates')
            connection.exec_driver_sql('DROP TABLE candidates')
            connection.exec_driver_sql('ALTER TABLE candidates_nullable_upgrade RENAME TO candidates')
            for statement in extras:
                connection.exec_driver_sql(statement)
            if connection.exec_driver_sql('PRAGMA foreign_key_check').first() is not None:
                raise RuntimeError("Foreign key validation failed during candidates upgrade")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.exec_driver_sql('PRAGMA foreign_keys=ON')
            connection.commit()
