import pytest

from src.database.dataset_loader import DatasetError, DatasetLoader


def write_csv(tmp_path, content):
    path = tmp_path / "labels.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_and_deduplicates_labels(tmp_path):
    path = write_csv(
        tmp_path,
        "image_name,medicine_label\n"
        "a.jpg,AMOXICILLIN\n"
        "b.jpg,amoxicillin\n"
        "c.jpg,IBUPROFEN\n",
    )
    dataset = DatasetLoader(path, tmp_path / "images").load()
    assert dataset.label_names == ["AMOXICILLIN", "IBUPROFEN"]
    assert len(dataset) == 2
    assert dataset.duplicate_rows == 1


def test_normalizes_labels_and_ignores_blank_rows(tmp_path):
    path = write_csv(
        tmp_path,
        "image_name,medicine_label\n"
        "a.jpg,  amoxicillin  \n"
        "b.jpg,\n"
        ",IBUPROFEN\n",
    )
    dataset = DatasetLoader(path, tmp_path / "images").load()
    assert dataset.label_names == ["AMOXICILLIN"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError):
        DatasetLoader(tmp_path / "missing.csv", tmp_path).load()


def test_wrong_header_raises(tmp_path):
    path = write_csv(tmp_path, "image_name,brand_name\na.jpg,AMOXIL\n")
    with pytest.raises(DatasetError):
        DatasetLoader(path, tmp_path).load()


def test_missing_image_reporting(tmp_path):
    path = write_csv(tmp_path, "image_name,medicine_label\nmissing.jpg,AMOXICILLIN\n")
    dataset = DatasetLoader(path, tmp_path / "images").load(check_images=True)
    assert dataset.missing_images == ["missing.jpg"]
