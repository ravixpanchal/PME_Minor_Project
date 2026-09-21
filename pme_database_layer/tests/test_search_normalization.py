from pme.repository import normalize_search_text


def test_normalize_search_text_ignores_case_and_extra_whitespace():
    assert normalize_search_text("  PrAkAsH   ChAnDrA  ") == "prakash chandra"
