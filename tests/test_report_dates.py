from datetime import date
import pytest
from src.database.schemas import GeneralDetails, Page4


@pytest.mark.parametrize('raw, expected', [
    ('29/7/25', date(2025,7,29)), ('30/07/25', date(2025,7,30)),
    ('12-08-26', date(2026,8,12)), ('05/08/23', date(2023,8,5)),
    ('29-7-25', date(2025,7,29)), ('2025-07-29', date(2025,7,29)),
    ('29.07.2025', date(2025,7,29)), ('21/2/96', date(1996,2,21)),
])
def test_common_report_dates(raw,expected):
    assert Page4(declaration_date=raw).declaration_date == expected
    assert GeneralDetails(date_of_birth=raw).date_of_birth == expected


def test_invalid_dates_are_not_guessed():
    with pytest.raises(ValueError):
        Page4(declaration_date='31/2/25')
    assert Page4(declaration_date=' ').declaration_date is None
