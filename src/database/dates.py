"""Shared parsing of literal report dates; original strings remain in raw JSON."""
from datetime import date, datetime


def report_date(value):
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y",
                "%d-%m-%y", "%d/%m/%y", "%d.%m.%y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported report date: {value!r}")
