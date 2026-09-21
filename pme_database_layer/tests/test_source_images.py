from pathlib import Path

from pme.source_images import get_original_image_paths


def test_get_original_image_paths_returns_all_images_in_name_order(
    tmp_path: Path,
):
    image_dir = tmp_path / "high_res" / "document-123"
    image_dir.mkdir(parents=True)
    (image_dir / "page_0001.png").touch()
    (image_dir / "page_0000.png").touch()
    (image_dir / "notes.txt").touch()

    assert get_original_image_paths(
        "document-123",
        interim_dir=tmp_path,
    ) == [
        image_dir / "page_0000.png",
        image_dir / "page_0001.png",
    ]


def test_get_original_image_paths_returns_empty_when_missing(
    tmp_path: Path,
):
    assert get_original_image_paths(
        "missing-document",
        interim_dir=tmp_path,
    ) == []
